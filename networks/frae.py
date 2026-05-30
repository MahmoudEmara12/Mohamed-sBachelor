import os
import pickle
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
# FRAE V4 CORE (UNCHANGED)
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        r = x
        x = F.gelu(self.norm1(self.fc1(x)))
        x = self.norm2(self.fc2(x))
        return F.gelu(x + r)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden=256, latent_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            ResBlock(hidden),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            ResBlock(hidden),
            nn.Linear(hidden, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class FRAENetV4(nn.Module):
    def __init__(self, input_dim, frames, n_mels, latent_dim=16):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels

        self.encoder = Encoder(input_dim, 256, latent_dim)
        self.decoder = Decoder(latent_dim, 256, input_dim)

        self.freq_weights = nn.Parameter(torch.linspace(0.8, 1.5, n_mels))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# =========================================================
# AUGMENTATION (UNCHANGED)
# =========================================================
def augment(x, frames, n_mels):
    x2 = x.clone()

    scale = torch.empty(x2.size(0), 1, device=x.device).uniform_(0.85, 1.15)
    x2 = x2 * scale
    x2 = x2 + torch.randn_like(x2) * 0.03

    roll = np.random.randint(-1, 2)
    if roll != 0:
        x3 = x2.view(x2.size(0), frames, n_mels)
        x3 = torch.roll(x3, roll, dims=2)
        x2 = x3.view(x2.size(0), -1)

    return x2


# =========================================================
# FRAE V4 (BASELINE-COMPATIBLE WRAPPER)
# =========================================================
class FRAEV4(BaseModel):

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

        self.mean = None
        self.inv_cov = None

    # -----------------------------------------------------
    def init_model(self):
        return FRAENetV4(
            input_dim=self.data.input_dim,
            frames=self.args.frames,
            n_mels=self.args.n_mels,
            latent_dim=getattr(self.args, "latent_dim", 16),
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
    # TRAIN (baseline-aligned)
    # =====================================================
    def train(self, epoch):

        self.model.train()

        total_loss = 0.0
        y_pred = []

        for batch in self.train_loader:

            x = batch[0].to(self.device, non_blocking=True).float()

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

                latent_loss = 0.00002 * torch.mean(torch.abs(z1 - z2))

                loss = recon_loss + latent_loss

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
    # COVARIANCE
    # =====================================================
    def build_covariance(self):

        self.model.eval()
        feats = []

        with torch.inference_mode():
            for batch in self.train_loader:
                x = batch[0].to(self.device).float()
                _, z = self.model(x)
                feats.append(z.cpu())

        feats = torch.cat(feats, dim=0)

        self.mean = feats.mean(dim=0)
        centered = feats - self.mean

        cov = (centered.T @ centered) / max(centered.size(0) - 1, 1)

        shrinkage = 0.05
        eye = torch.eye(cov.size(0))

        cov = (1 - shrinkage) * cov + shrinkage * eye

        self.inv_cov = torch.linalg.pinv(cov)

        self.mean = self.mean.to(self.device)
        self.inv_cov = self.inv_cov.to(self.device)

    # =====================================================
    # TEST (BASELINE FORMAT)
    # =====================================================
    def test(self):

        self.model.eval()
        self.build_covariance()

        all_scores = []
        all_labels = []
        all_domains = []
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

                    diff = z - self.mean
                    mahal = torch.sum((diff @ self.inv_cov) * diff, dim=1)

                    score = torch.minimum(mse, mahal)

                    all_scores.extend(score.cpu().numpy())
                    all_labels.extend(batch[1].cpu().numpy() if len(batch) > 1 else np.zeros(x.size(0)))
                    all_domains.extend(batch[2].cpu().numpy() if len(batch) > 2 else np.zeros(x.size(0)))
                    all_names.extend(batch[3] if len(batch) > 3 else [f"s{i}" for i in range(x.size(0))])

        pd.DataFrame({
            "anon": all_names,
            "score": all_scores,
            "label": all_labels,
            "domain": all_domains
        }).to_csv("frae_v4_scores.csv", index=False)

        print("[OK] FRAE V4 evaluation complete → frae_v4_scores.csv")


# =========================================================
FRAE = FRAEV4
FRAEV2 = FRAEV4
FRAEV3 = FRAEV4