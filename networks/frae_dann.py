import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy
from sklearn import metrics
import csv
from tqdm import tqdm

from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.base_model import BaseModel

from networks.criterion.mahala import (
    cov_v,
    loss_function_mahala,
    calc_inv_cov
)

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
        return grad_output.neg() * ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# =========================================================
# AUGMENTATION
# =========================================================
def augment(x, frames, n_mels):
    x = x + torch.randn_like(x) * 0.02

    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(0.9, 1.1)
    x = x * scale

    mask_len = np.random.randint(0, max(1, n_mels // 6))
    if mask_len > 0:
        f0 = np.random.randint(0, n_mels - mask_len + 1)
        x[:, :, f0:f0 + mask_len] = 0

    mask_len_t = np.random.randint(0, max(1, frames // 6))
    if mask_len_t > 0:
        t0 = np.random.randint(0, frames - mask_len_t + 1)
        x[:, t0:t0 + mask_len_t, :] = 0

    return x


# =========================================================
# MODEL COMPONENTS
# =========================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden=256, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim)
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.net(z)


class DomainClassifier(nn.Module):
    def __init__(self, latent_dim, n_domains=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_domains)
        )

    def forward(self, z):
        return self.net(z)


class FRAE_DANNNet(nn.Module):
    def __init__(self, input_dim, frames, n_mels, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(input_dim, 256, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.domain_clf = DomainClassifier(latent_dim)

        self.freq_weights = nn.Parameter(torch.linspace(0.8, 1.2, n_mels))

    def forward(self, x, alpha=0.0):
        z = self.encoder(x)
        recon = self.decoder(z)
        rev_z = grad_reverse(z, alpha)
        dom = self.domain_clf(rev_z)
        return recon, z, dom


# =========================================================
# LOSS
# =========================================================
def contrastive_loss(z1, z2, temperature=0.5):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    reps = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(reps, reps.T) / temperature

    bsz = z1.size(0)
    labels = torch.arange(bsz, device=z1.device)
    labels = torch.cat([labels + bsz, labels])

    mask = torch.eye(2 * bsz, device=z1.device).bool()
    sim = sim.masked_fill(mask, -1e4)

    return F.cross_entropy(sim, labels)


# =========================================================
# MAIN MODEL (DCASE-COMPATIBLE)
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

        self.scaler = torch.cuda.amp.GradScaler()

        self.block_size = None

    # -----------------------------------------------------
    def init_model(self):
        self.block_size = self.data.height
        return FRAE_DANNNet(
            input_dim=self.data.input_dim,
            frames=self.args.frames,
            n_mels=self.args.n_mels
        )

    # -----------------------------------------------------
    def train(self, epoch):

        # === covariance pass exactly like baseline ===
        is_cov_epoch = (epoch == self.args.epochs + 1)

        if is_cov_epoch:
            print("\n[INFO] COVARIANCE COMPUTATION PHASE")
            self.model.eval()
            torch.set_grad_enabled(False)

            cov_x_source = torch.zeros((self.block_size, self.block_size), device=self.device)
            cov_x_target = torch.zeros_like(cov_x_source)

            num_source = 0
            num_target = 0
        else:
            self.model.train()

        train_loader = self.train_loader

        for batch in tqdm(train_loader):

            data = batch[0].to(self.device).float()
            if data.shape[0] <= 1:
                continue

            data_name_list = batch[3]

            is_target = ["target" in n for n in data_name_list]
            is_source = np.logical_not(is_target).tolist()

            machine_id = torch.argmax(batch[2], dim=1).to(self.device)

            recon, z = self.model(data)

            if is_cov_epoch:
                score_2d, cov_s, cov_t = loss_function_mahala(
                    recon_x=recon,
                    x=data,
                    block_size=self.block_size,
                    update_cov=True,
                    reduction=False,
                    is_source_list=is_source,
                    is_target_list=is_target
                )

                cov_x_source += cov_v(cov_s, 1)
                cov_x_target += cov_v(cov_t, 1)

                num_source += sum(is_source)
                num_target += sum(is_target)

                continue

            # === normal training ===
            score = self.loss_fn(recon, data)
            loss = score.mean()

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.scheduler.step()

        # finalize covariance like baseline
        if is_cov_epoch:
            cov_x_source /= max(num_source - 1, 1)
            cov_x_target /= max(num_target - 1, 1)

            self.model.cov_source.data = cov_x_source
            self.model.cov_target.data = cov_x_target

            calc_inv_cov(self.model, self.device)

    # -----------------------------------------------------
    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")

    # -----------------------------------------------------
    def test(self):

        anm_score_figdata = None
        mode = self.data.mode

        if not os.path.exists(self.model_path):
            print("Model not found")
            return

        self.model.load_state_dict(torch.load(self.model_path))
        self.model.eval()

        inv_cov_s, inv_cov_t = calc_inv_cov(self.model, self.device)

        decision_threshold = self.calc_decision_threshold(
            score_distr_file_path=self.mse_score_distr_file_path
        )

        for idx, loader in enumerate(self.test_loader):

            section_name = f"section_{self.data.section_id_list[idx]}"
            anomaly_list = []
            decision_list = []

            y_pred, y_true = [], []

            for batch in loader:

                data = batch[0].to(self.device).float()
                recon, _ = self.model(data)

                loss_s, num = loss_function_mahala(
                    recon, data, self.block_size,
                    cov=inv_cov_s, use_precision=True, reduction=False
                )

                loss_t, num = loss_function_mahala(
                    recon, data, self.block_size,
                    cov=inv_cov_t, use_precision=True, reduction=False
                )

                score = min(
                    self.loss_reduction_1d(loss_s).mean().item(),
                    self.loss_reduction_1d(loss_t).mean().item()
                )

                y_pred.append(score)

                basename = batch[3][0]
                anomaly_list.append([basename, score])

                decision_list.append([basename, int(score > decision_threshold)])

                if mode:
                    y_true.append(batch[1][0].item())

            result_dir = self.result_dir if self.args.dev else self.eval_data_result_dir

            save_csv(result_dir / f"anomaly_score_{section_name}.csv", anomaly_list)
            save_csv(result_dir / f"decision_{section_name}.csv", decision_list)


def save_csv(path, data):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(data)