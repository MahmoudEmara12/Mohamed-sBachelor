import os
import glob
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from sklearn import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.base_model import BaseModel


# =========================================================
# CONFORMER BLOCK — unchanged
# =========================================================
class ConformerBlock(nn.Module):
    def __init__(self, dim=64, heads=4, ff_mult=2, dropout=0.1):
        super().__init__()

        self.ff1 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads,
            dropout=dropout, batch_first=True,
        )

        self.conv = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, 1),
        )

        self.ff2 = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)

        a = self.attn_norm(x)
        a, _ = self.attn(a, a, a)
        x = x + a

        c = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + c

        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


# =========================================================
# ENCODER
# =========================================================
class Encoder(nn.Module):
    def __init__(self, frames, n_mels, dim=64, latent_dim=32, depth=2):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels

        self.input_proj = nn.Linear(n_mels, dim)
        self.blocks = nn.ModuleList([
            ConformerBlock(dim) for _ in range(depth)
        ])

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.embedding = nn.Sequential(
            nn.Linear(dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, self.frames, self.n_mels)

        x = self.input_proj(x)
        for b in self.blocks:
            x = b(x)

        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        return self.embedding(x)


# =========================================================
# DECODER
# =========================================================
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, z):
        return self.net(z)


# =========================================================
# MODEL
# =========================================================
class ConformerNet(nn.Module):
    def __init__(self, input_dim, frames, n_mels, latent_dim=32):
        super().__init__()
        self.encoder = Encoder(frames, n_mels, 64, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.freq_weights = nn.Parameter(torch.linspace(0.8, 1.2, n_mels))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# =========================================================
# AUGMENTATION — unchanged
# =========================================================
def augment(x, frames, n_mels):
    x = x + torch.randn_like(x) * 0.01
    scale = torch.empty(x.size(0), 1, device=x.device).uniform_(0.9, 1.1)
    x = x * scale
    shift = np.random.randint(-1, 2)

    if shift != 0:
        x = torch.roll(
            x.view(x.size(0), frames, n_mels),
            shifts=shift,
            dims=2
        ).view(x.size(0), -1)

    return x


# =========================================================
# CONFORMER (BASELINE-COMPATIBLE WRAPPER)
# =========================================================
class Conformer(BaseModel):

    def __init__(self, args, train=True, test=False):
        super().__init__(args=args, train=train, test=test)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=1e-4,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(getattr(self.args, "epochs", 50), 1),
            eta_min=self.args.learning_rate * 0.05,
        )

        self.scaler = torch.cuda.amp.GradScaler()

        self.mean_s = None
        self.inv_cov_s = None
        self.mean_t = None
        self.inv_cov_t = None

    # -----------------------------------------------------
    def init_model(self):
        return ConformerNet(
            input_dim=self.data.input_dim,
            frames=self.args.frames,
            n_mels=self.args.n_mels,
            latent_dim=32,
        )

    # =====================================================
    # LOSS (BASELINE STYLE)
    # =====================================================
    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")

    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    # =====================================================
    # TRAIN (baseline-compatible logging structure)
    # =====================================================
    def train(self, epoch):

        self.model.train()

        total_loss = 0
        y_pred = []

        for batch in self.train_loader:

            x = batch[0].to(self.device).float()

            x1 = augment(x, self.args.frames, self.args.n_mels)
            x2 = augment(x, self.args.frames, self.args.n_mels)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():

                recon1, z1 = self.model(x1)
                _, z2 = self.model(x2)

                x2d = x.view(x.size(0), self.args.frames, self.args.n_mels)
                r2d = recon1.view(x.size(0), self.args.frames, self.args.n_mels)

                w = F.softmax(self.model.freq_weights, dim=0).unsqueeze(0).unsqueeze(0)

                score = ((x2d - r2d) ** 2) * w
                score_1d = self.loss_reduction_1d(score)
                recon_loss = self.loss_reduction(score_1d, x.size(0))

                latent_loss = F.mse_loss(z1, z2)

                loss = recon_loss + 0.05 * latent_loss

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            y_pred.append(loss.item())

        self.scheduler.step()

        print(f"[Epoch {epoch}] loss={total_loss / len(self.train_loader):.5f}")

        return False

    # =====================================================
    # COVARIANCE (kept simple but aligned)
    # =====================================================
    def _build_cov(self, feats, shrinkage=0.05):
        mean = feats.mean(dim=0)
        centered = feats - mean
        cov = (centered.T @ centered) / max(centered.size(0) - 1, 1)
        cov = (1 - shrinkage) * cov + shrinkage * torch.eye(cov.size(0))
        return mean, torch.linalg.pinv(cov)

    def build_covariance(self):

        self.model.eval()
        feats = []

        with torch.inference_mode():
            for batch in self.train_loader:
                x = batch[0].to(self.device).float()
                _, z = self.model(x)
                feats.append(z.cpu())

        feats = torch.cat(feats, dim=0)
        self.mean_s, self.inv_cov_s = self._build_cov(feats)
        self.mean_t, self.inv_cov_t = self._build_cov(feats)

        self.mean_s = self.mean_s.to(self.device)
        self.inv_cov_s = self.inv_cov_s.to(self.device)
        self.mean_t = self.mean_t.to(self.device)
        self.inv_cov_t = self.inv_cov_t.to(self.device)

    # =====================================================
    # TEST (baseline-style scoring)
    # =====================================================
    def test(self):

        self.model.eval()
        self.build_covariance()

        all_scores = []
        all_labels = []
        all_names = []

        with torch.inference_mode():

            for section_loader in self.test_loader:
                for batch in section_loader:

                    x = batch[0].to(self.device).float()
                    recon, z = self.model(x)

                    x2d = x.view(x.size(0), self.args.frames, self.args.n_mels)
                    r2d = recon.view(x.size(0), self.args.frames, self.args.n_mels)

                    w = F.softmax(self.model.freq_weights, dim=0).unsqueeze(0).unsqueeze(0)

                    mse = ((x2d - r2d) ** 2 * w).mean(dim=(1, 2))

                    diff = z - self.mean_s
                    mahal = torch.sum((diff @ self.inv_cov_s) * diff, dim=1)

                    score = torch.minimum(mse, mahal)

                    all_scores.extend(score.cpu().numpy())
                    all_labels.extend(batch[1].cpu().numpy() if len(batch) > 1 else np.zeros(x.size(0)))
                    all_names.extend(batch[3] if len(batch) > 3 else [f"s{i}" for i in range(x.size(0))])

        pd.DataFrame({
            "name": all_names,
            "score": all_scores,
            "label": all_labels
        }).to_csv("conformer_scores.csv", index=False)

        print("[OK] Conformer evaluation complete → conformer_scores.csv")


def save_csv(path, data):
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(data)