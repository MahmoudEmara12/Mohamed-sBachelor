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


class FRAE_DANNNet(nn.Module):
    def __init__(self, input_dim, latent_dim=32):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.domain = DomainClassifier(latent_dim)

    def forward(self, x, alpha=0.0):
        z = self.encoder(x)
        recon = self.decoder(z)

        dom = self.domain(grad_reverse(z, alpha))

        return recon, z, dom


# =========================================================
# MAIN CLASS
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

        # FIXED AMP (no warning)
        self.scaler = torch.amp.GradScaler("cuda")

        self.latent_dim = 32
        self.mean = None
        self.inv_cov = None

    # -----------------------------------------------------
    def init_model(self):
        return FRAE_DANNNet(
            input_dim=self.data.input_dim,
            latent_dim=self.latent_dim
        )

    # -----------------------------------------------------
    def train(self, epoch):

        self.model.train()
        total_loss = 0

        for batch in self.train_loader:

            x = batch[0].to(self.device).float()

            self.optimizer.zero_grad()

            alpha = min(epoch / max(self.args.epochs, 1), 1.0)

            with torch.amp.autocast("cuda"):

                recon, z, dom = self.model(x, alpha)

                # reconstruction
                recon_loss = F.mse_loss(recon, x)

                # domain loss (safe fallback if labels exist)
                if len(batch) > 2:
                    domain_labels = torch.zeros(x.size(0), dtype=torch.long, device=self.device)
                    dom_loss = F.cross_entropy(dom, domain_labels)
                else:
                    dom_loss = 0.0

                loss = recon_loss + 0.1 * dom_loss

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        self.scheduler.step()

        print(f"[Epoch {epoch}] loss={total_loss / len(self.train_loader):.6f}")

    # -----------------------------------------------------
    # MAHALANOBIS BUILD (CLEAN VERSION)
    # -----------------------------------------------------
    def build_covariance(self):

        self.model.eval()
        feats = []

        with torch.no_grad():
            for batch in self.train_loader:
                x = batch[0].to(self.device).float()
                _, z, _ = self.model(x)
                feats.append(z.cpu())

        feats = torch.cat(feats, dim=0)

        self.mean = feats.mean(0)

        centered = feats - self.mean
        cov = (centered.T @ centered) / (len(feats) - 1)

        shrink = 0.05
        cov = (1 - shrink) * cov + shrink * torch.eye(cov.size(0))

        self.inv_cov = torch.linalg.pinv(cov)

        self.mean = self.mean.to(self.device)
        self.inv_cov = self.inv_cov.to(self.device)

        print("[INFO] Mahalanobis ready")

    # -----------------------------------------------------
    def test(self):

        self.model.eval()
        self.build_covariance()

        all_scores = []

        with torch.no_grad():
            for batch in self.test_loader:

                x = batch[0].to(self.device).float()
                recon, z, _ = self.model(x)

                diff = z - self.mean

                score = torch.sum(
                    (diff @ self.inv_cov) * diff,
                    dim=1
                )

                all_scores.extend(score.cpu().numpy())

        print("[INFO] Test done")
        return all_scores