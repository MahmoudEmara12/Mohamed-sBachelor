import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.base_model import BaseModel


# =========================================================
# GRADIENT REVERSAL LAYER
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
# AUGMENTATION (SpecAugment style)
# =========================================================
def augment(x, frames, n_mels):

    # x shape:
    # (B, frames, n_mels)

    # Gaussian noise
    x = x + torch.randn_like(x) * 0.02

    # Random amplitude scaling
    scale = torch.empty(
        x.size(0),
        1,
        1,
        device=x.device
    ).uniform_(0.9, 1.1)

    x = x * scale

    # Frequency masking
    mask_len = np.random.randint(
        0,
        max(1, n_mels // 6)
    )

    if mask_len > 0:

        f0 = np.random.randint(
            0,
            n_mels - mask_len + 1
        )

        x[:, :, f0:f0 + mask_len] = 0

    # Time masking
    mask_len_t = np.random.randint(
        0,
        max(1, frames // 6)
    )

    if mask_len_t > 0:

        t0 = np.random.randint(
            0,
            frames - mask_len_t + 1
        )

        x[:, t0:t0 + mask_len_t, :] = 0

    return x


# =========================================================
# ENCODER
# =========================================================
class Encoder(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden=256,
        latent_dim=32
    ):
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

        z = self.net(x)

        return F.normalize(z, dim=1)


# =========================================================
# DECODER
# =========================================================
class Decoder(nn.Module):

    def __init__(
        self,
        latent_dim,
        output_dim
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.GELU(),

            nn.Linear(128, output_dim)
        )

    def forward(self, z):
        return self.net(z)


# =========================================================
# DOMAIN CLASSIFIER
# =========================================================
class DomainClassifier(nn.Module):

    def __init__(
        self,
        latent_dim,
        n_domains=2
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),

            nn.Linear(64, n_domains)
        )

    def forward(self, z):
        return self.net(z)


# =========================================================
# FRAE + DOMAIN ADVERSARIAL NETWORK
# =========================================================
class FRAE_DANNNet(nn.Module):

    def __init__(
        self,
        input_dim,
        frames,
        n_mels,
        latent_dim=32
    ):
        super().__init__()

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden=256,
            latent_dim=latent_dim
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            output_dim=input_dim
        )

        self.domain_clf = DomainClassifier(
            latent_dim=latent_dim
        )

        self.freq_weights = nn.Parameter(
            torch.linspace(0.8, 1.2, n_mels)
        )

    def forward(self, x, alpha=0.0):

        z = self.encoder(x)

        recon = self.decoder(z)

        rev_z = grad_reverse(z, alpha)

        domain_logits = self.domain_clf(rev_z)

        return recon, z, domain_logits


# =========================================================
# INFO-NCE CONTRASTIVE LOSS
# =========================================================
def contrastive_loss(
    z1,
    z2,
    temperature=0.5
):

    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    representations = torch.cat(
        [z1, z2],
        dim=0
    )

    similarity_matrix = torch.matmul(
        representations,
        representations.T
    )

    similarity_matrix = (
        similarity_matrix / temperature
    )

    batch_size = z1.size(0)

    labels = torch.arange(
        batch_size,
        device=z1.device
    )

    labels = torch.cat([
        labels + batch_size,
        labels
    ])

    mask = torch.eye(
        2 * batch_size,
        dtype=torch.bool,
        device=z1.device
    )

    similarity_matrix = similarity_matrix.masked_fill(
        mask,
        -1e4
    )

    loss = F.cross_entropy(
        similarity_matrix,
        labels
    )

    return loss


# =========================================================
# BASEMODEL WRAPPER
# =========================================================
class FRAE_DANN(BaseModel):

    def __init__(
        self,
        args,
        train=True,
        test=False
    ):
        super().__init__(
            args=args,
            train=train,
            test=test
        )

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

        self.best_hmean = 0.0
        self.early_stop_patience = 20

        # Mahalanobis parameters
        self.mean_s = None
        self.inv_cov_s = None

        self.mean_t = None
        self.inv_cov_t = None

    # -----------------------------------------------------
    # INIT MODEL
    # -----------------------------------------------------
    def init_model(self):

        return FRAE_DANNNet(
            input_dim=self.data.input_dim,
            frames=self.args.frames,
            n_mels=self.args.n_mels,
            latent_dim=32
        )

    # -----------------------------------------------------
    # BUILD COVARIANCE
    # -----------------------------------------------------
    @staticmethod
    def _build_covariance(
        feats,
        shrinkage=0.1
    ):

        mean = feats.mean(dim=0)

        centered = feats - mean

        cov = (
            centered.T @ centered
        ) / max(centered.size(0) - 1, 1)

        eye = torch.eye(
            cov.size(0),
            device=cov.device
        )

        cov = (
            (1 - shrinkage) * cov
            + shrinkage * eye
        )

        inv_cov = torch.linalg.pinv(cov)

        return mean, inv_cov

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------
    def train(self, epoch):

        self.model.train()

        total_loss = 0.0

        alpha = (
            epoch / self.args.epochs
        ) ** 2

        for batch in self.train_loader:

            x = batch[0].to(self.device).float()

            d = batch[2].to(self.device).long()

            batch_size = x.size(0)

            # Reshape into spectrogram
            x_spec = x.view(
                batch_size,
                self.args.frames,
                self.args.n_mels
            )

            # Two augmented views
            x1 = augment(
                x_spec.clone(),
                self.args.frames,
                self.args.n_mels
            )

            x2 = augment(
                x_spec.clone(),
                self.args.frames,
                self.args.n_mels
            )

            x1 = x1.view(batch_size, -1)
            x2 = x2.view(batch_size, -1)

            self.optimizer.zero_grad(
                set_to_none=True
            )

            with torch.cuda.amp.autocast():

                recon1, z1, dom_logits1 = self.model(
                    x1,
                    alpha
                )

                _, z2, _ = self.model(
                    x2,
                    alpha
                )

                # Reconstruction loss
                x2d = x.view(
                    batch_size,
                    self.args.frames,
                    self.args.n_mels
                )

                r2d = recon1.view(
                    batch_size,
                    self.args.frames,
                    self.args.n_mels
                )

                w = F.softmax(
                    self.model.freq_weights,
                    dim=0
                )

                w = w.unsqueeze(0).unsqueeze(0)

                recon_loss = (
                    ((x2d - r2d) ** 2) * w
                ).mean()

                # Contrastive loss
                cont_loss = contrastive_loss(
                    z1,
                    z2
                )

                # Domain adversarial loss
                dom_loss = F.cross_entropy(
                    dom_logits1,
                    d.view(-1)
                )

                # Final loss
                loss = (
                    recon_loss
                    + 0.3 * cont_loss
                    + 0.1 * dom_loss
                )

            self.scaler.scale(loss).backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                5.0
            )

            self.scaler.step(self.optimizer)

            self.scaler.update()

            total_loss += loss.item()

        self.scheduler.step()

        avg_loss = (
            total_loss / len(self.train_loader)
        )

        print(
            f"[Epoch {epoch}] "
            f"loss={avg_loss:.5f}"
        )

        return False

    # -----------------------------------------------------
    # BUILD MAHALANOBIS
    # -----------------------------------------------------
    def build_covariance(self):

        self.model.eval()

        feats_s = []
        feats_t = []

        with torch.inference_mode():

            for batch in self.train_loader:

                x = batch[0].to(self.device).float()

                _, z, _ = self.model(x)

                z_cpu = z.cpu()

                # Domains
                if len(batch) > 2:

                    d = batch[2]

                    if isinstance(d, torch.Tensor):
                        d = d.cpu().numpy()
                    else:
                        d = np.zeros(z_cpu.size(0))

                else:
                    d = np.zeros(z_cpu.size(0))

                for i, dom in enumerate(d):

                    if dom == 1:
                        feats_t.append(z_cpu[i])
                    else:
                        feats_s.append(z_cpu[i])

        # Safety fallback
        if len(feats_s) == 0:
            feats_s = feats_t.copy()

        if len(feats_t) == 0:
            feats_t = feats_s.copy()

        feats_s = torch.stack(feats_s)
        feats_t = torch.stack(feats_t)

        self.mean_s, self.inv_cov_s = (
            self._build_covariance(feats_s)
        )

        self.mean_t, self.inv_cov_t = (
            self._build_covariance(feats_t)
        )

        self.mean_s = self.mean_s.to(self.device)
        self.inv_cov_s = self.inv_cov_s.to(self.device)

        self.mean_t = self.mean_t.to(self.device)
        self.inv_cov_t = self.inv_cov_t.to(self.device)

        print("[INFO] Source covariance built")
        print("[INFO] Target covariance built")

    # -----------------------------------------------------
    # TEST
    # -----------------------------------------------------
    def test(self):

        self.model.eval()

        self.build_covariance()

        all_mse = []
        all_mahal = []

        all_labels = []
        all_domains = []
        all_names = []

        with torch.inference_mode():

            for section_loader in self.test_loader:

                for batch in section_loader:

                    x = batch[0].to(self.device).float()

                    recon, z, _ = self.model(x)

                    # Reconstruction score
                    x2d = x.view(
                        x.size(0),
                        self.args.frames,
                        self.args.n_mels
                    )

                    r2d = recon.view(
                        x.size(0),
                        self.args.frames,
                        self.args.n_mels
                    )

                    w = F.softmax(
                        self.model.freq_weights,
                        dim=0
                    )

                    w = w.unsqueeze(0).unsqueeze(0)

                    mse = (
                        ((x2d - r2d) ** 2) * w
                    ).mean(dim=(1, 2))

                    # Mahalanobis source
                    diff_s = z - self.mean_s

                    mahal_s = torch.sum(
                        (diff_s @ self.inv_cov_s)
                        * diff_s,
                        dim=1
                    )

                    # Mahalanobis target
                    diff_t = z - self.mean_t

                    mahal_t = torch.sum(
                        (diff_t @ self.inv_cov_t)
                        * diff_t,
                        dim=1
                    )

                    # Minimum distance
                    mahal = torch.minimum(
                        mahal_s,
                        mahal_t
                    )

                    all_mse.extend(
                        mse.cpu().numpy()
                    )

                    all_mahal.extend(
                        mahal.cpu().numpy()
                    )

                    y = (
                        batch[1].cpu().numpy()
                        if len(batch) > 1
                        else np.zeros(x.size(0))
                    )

                    d = (
                        batch[2].cpu().numpy()
                        if len(batch) > 2
                        else np.zeros(x.size(0))
                    )

                    names = (
                        batch[3]
                        if len(batch) > 3
                        else [
                            f"s{i}"
                            for i in range(x.size(0))
                        ]
                    )

                    all_labels.extend(y)
                    all_domains.extend(d)
                    all_names.extend(names)

        # Save scores
        os.makedirs(
            "results",
            exist_ok=True
        )

        df = pd.DataFrame({
            "anon": all_names,
            "mse_score": all_mse,
            "mahal_score": all_mahal,
            "label": all_labels,
            "domain": all_domains
        })

        df.to_csv(
            "frae_all_scores.csv",
            index=False
        )

        print(
            "\n[OK] FRAE+DANN evaluation complete"
        )

        print(
            "Saved → frae_all_scores.csv"
        )

        print(
            f"MSE range: "
            f"{min(all_mse):.6f} "
            f"→ "
            f"{max(all_mse):.6f}"
        )

        print(
            f"Mahal range: "
            f"{min(all_mahal):.6f} "
            f"→ "
            f"{max(all_mahal):.6f}"
        )