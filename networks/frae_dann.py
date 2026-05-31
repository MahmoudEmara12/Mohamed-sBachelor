import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import scipy

from sklearn import metrics
from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.base_model import BaseModel


# =========================================================
# SMALL HELPERS
# =========================================================
def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    return x


def _to_1d_int_array(x):
    x = _to_numpy(x)

    if x.ndim == 0:
        return np.array([int(x)])

    if x.ndim > 1:
        # Handles one-hot labels/domains if they ever appear
        if x.shape[-1] > 1:
            x = np.argmax(x, axis=-1)
        else:
            x = np.squeeze(x, axis=-1)

    return x.astype(int).reshape(-1)


def _safe_roc_auc(y_true, y_score, max_fpr=None):
    y_true = np.asarray(y_true).reshape(-1)
    y_score = np.asarray(y_score).reshape(-1)

    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return np.nan

    try:
        if max_fpr is None:
            return metrics.roc_auc_score(y_true, y_score)
        return metrics.roc_auc_score(y_true, y_score, max_fpr=max_fpr)
    except Exception:
        return np.nan


def _safe_hmean(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values = values[values > 0]

    if len(values) == 0:
        return np.nan

    return scipy.stats.hmean(np.maximum(values, sys.float_info.epsilon))


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

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=torch.cuda.is_available()
        )

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
            epoch / max(self.args.epochs, 1)
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

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(
                enabled=torch.cuda.is_available()
            ):

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
            self.scaler.unscale_(self.optimizer)

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

        # Save model exactly like baseline
        torch.save(
            self.model.state_dict(),
            self.model_path
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": avg_loss
            },
            self.checkpoint_path
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
                    d = _to_1d_int_array(batch[2])
                else:
                    d = np.zeros(z_cpu.size(0), dtype=int)

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
                        _to_1d_int_array(batch[1])
                        if len(batch) > 1
                        else np.zeros(x.size(0), dtype=int)
                    )

                    d = (
                        _to_1d_int_array(batch[2])
                        if len(batch) > 2
                        else np.zeros(x.size(0), dtype=int)
                    )

                    names = (
                        batch[3]
                        if len(batch) > 3
                        else [
                            f"s{i}"
                            for i in range(x.size(0))
                        ]
                    )

                    all_labels.extend(y.tolist())
                    all_domains.extend(d.tolist())
                    all_names.extend(names)

        # Save scores
        result_dir = (
            self.result_dir
            if self.args.dev
            else self.eval_data_result_dir
        )
        result_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        save_path = result_dir / (
            f"frae_all_scores_"
            f"{self.args.dataset}_"
            f"seed{self.args.seed}"
            f"{self.model_name_suffix}"
            f"{self.eval_suffix}.csv"
        )

        df = pd.DataFrame({
            "filename": all_names,
            "mse_score": all_mse,
            "mahal_score": all_mahal,
            "label": all_labels,
            "domain": all_domains
        })

        df.to_csv(
            save_path,
            index=False
        )

        # =====================================================
        # METRICS: SOURCE / TARGET AUC, pAUC, HARMONIC MEAN
        # =====================================================
        y_true = np.asarray(all_labels, dtype=int)
        y_pred = np.asarray(all_mahal, dtype=float)
        domains = np.asarray(all_domains)

        # Support either numeric domains or string domains
        if domains.dtype.kind in {"U", "S", "O"}:
            domains_str = np.array([str(x).lower() for x in domains])
            source_mask = (domains_str == "source")
            target_mask = (domains_str == "target")
        else:
            source_mask = (domains == 0)
            target_mask = (domains == 1)

        max_fpr = getattr(self.args, "max_fpr", 0.1)

        # Source AUC / pAUC
        y_true_s_auc = y_true[source_mask | (y_true == 1)]
        y_pred_s_auc = y_pred[source_mask | (y_true == 1)]

        auc_s = _safe_roc_auc(y_true_s_auc, y_pred_s_auc)
        p_auc_s = _safe_roc_auc(
            y_true[source_mask],
            y_pred[source_mask],
            max_fpr=max_fpr
        )

        # Target AUC / pAUC
        y_true_t_auc = y_true[target_mask | (y_true == 1)]
        y_pred_t_auc = y_pred[target_mask | (y_true == 1)]

        auc_t = _safe_roc_auc(y_true_t_auc, y_pred_t_auc)
        p_auc_t = _safe_roc_auc(
            y_true[target_mask],
            y_pred[target_mask],
            max_fpr=max_fpr
        )

        # Overall pAUC
        p_auc = _safe_roc_auc(
            y_true,
            y_pred,
            max_fpr=max_fpr
        )

        # Means over available metrics
        perf_values = np.array(
            [auc_s, auc_t, p_auc, p_auc_s, p_auc_t],
            dtype=float
        )
        perf_values = perf_values[np.isfinite(perf_values)]

        arithmetic_mean = (
            float(np.mean(perf_values))
            if len(perf_values) > 0
            else np.nan
        )
        harmonic_mean = _safe_hmean(perf_values)

        print(
            "\n[OK] FRAE+DANN evaluation complete"
        )
        print(
            f"Saved → {save_path}"
        )

        print(
            f"AUC (source): {auc_s:.6f}"
        )
        print(
            f"pAUC (source): {p_auc_s:.6f}"
        )
        print(
            f"AUC (target): {auc_t:.6f}"
        )
        print(
            f"pAUC (target): {p_auc_t:.6f}"
        )
        print(
            f"pAUC (all): {p_auc:.6f}"
        )
        print(
            f"Arithmetic mean: {arithmetic_mean:.6f}"
        )
        print(
            f"Harmonic mean: {harmonic_mean:.6f}"
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

        return {
            "save_path": str(save_path),
            "auc_source": auc_s,
            "auc_target": auc_t,
            "pauc_all": p_auc,
            "pauc_source": p_auc_s,
            "pauc_target": p_auc_t,
            "arithmetic_mean": arithmetic_mean,
            "harmonic_mean": harmonic_mean,
        }