import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.base_model import BaseModel


# =========================================================
# GRADIENT REVERSAL
# =========================================================
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# =========================================================
# MODEL
# =========================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class DomainClassifier(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, z):
        return self.net(z)


# =========================================================
# NETWORK
# =========================================================
class FRAE_DANNNet(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.domain = DomainClassifier(latent_dim)
        self.register_buffer("cov_source", torch.zeros(latent_dim, latent_dim))
        self.register_buffer("cov_target", torch.zeros(latent_dim, latent_dim))

    def forward(self, x, alpha=0.0):
        z = self.encoder(x)
        recon = self.decoder(z)
        dom = self.domain(grad_reverse(z, alpha))
        return recon, z, dom


# =========================================================
# MAIN MODEL
# =========================================================
class FRAE_DANN(BaseModel):

    def __init__(self, args, train=True, test=False):
        super().__init__(args=args, train=train, test=test)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=1e-4
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.05
        )

        # FIXED AMP
        self.scaler = torch.amp.GradScaler("cuda")

        self.block_size = None

    # -----------------------------------------------------
    def init_model(self):
        self.block_size = int(self.data.input_dim)

        return FRAE_DANNNet(
            input_dim=self.data.input_dim,
            latent_dim=32
        )

    # -----------------------------------------------------
    def train(self, epoch):

        is_cov_epoch = (epoch == self.args.epochs + 1)

        if is_cov_epoch:
            print("\n[INFO] COVARIANCE COMPUTATION PHASE")
            self.model.eval()
            torch.set_grad_enabled(False)

            cov_x_source = torch.zeros((32, 32), device=self.device)
            cov_x_target = torch.zeros_like(cov_x_source)

            num_source = 0
            num_target = 0
        else:
            self.model.train()

        for batch in self.train_loader:

            x = batch[0].to(self.device).float()

            if is_cov_epoch:
                recon, z, _ = self.model(x)

                is_target = ["target" in n for n in batch[3]]
                is_source = np.logical_not(is_target)

                # safe covariance accumulation (NO None crashes)
                if z is not None:
                    if any(is_source):
                        cov_x_source += torch.cov(z[torch.tensor(is_source)].T).detach()
                        num_source += sum(is_source)

                    if any(is_target):
                        cov_x_target += torch.cov(z[torch.tensor(is_target)].T).detach()
                        num_target += sum(is_target)

                continue

            self.optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                recon, z, dom = self.model(x, alpha=0.5)

                loss = F.mse_loss(recon, x)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.scheduler.step()

        # finalize covariance safely
        if is_cov_epoch:
            cov_x_source /= max(num_source, 1)
            cov_x_target /= max(num_target, 1)

            self.model.cov_source.copy_(cov_x_source)
            self.model.cov_target.copy_(cov_x_target)

            print("[INFO] Covariance computed safely")

    # -----------------------------------------------------
    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x, reduction="none")