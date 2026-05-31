import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from networks.base_model import BaseModel


# =========================================================
# CONFORMER BLOCK  — unchanged
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
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Conv1d(dim, dim, kernel_size=1),
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

        attn_input = self.attn_norm(x)
        attn_out, _ = self.attn(attn_input, attn_input, attn_input)
        x = x + attn_out

        conv_out = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + conv_out

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
            ConformerBlock(dim=dim, heads=4, ff_mult=2, dropout=0.1)
            for _ in range(depth)
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
        for block in self.blocks:
            x = block(x)
        x = self.pool(x.transpose(1, 2)).squeeze(-1)
        z = self.embedding(x)
        # F.normalize removed: unconstrained latent codes
        # give the decoder more freedom and allow Mahalanobis
        # to work correctly in the residual space.
        return z


# =========================================================
# DECODER  — unchanged
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
# CONFORMER NETWORK  — unchanged
# =========================================================
class ConformerNet(nn.Module):

    def __init__(self, input_dim, frames, n_mels, latent_dim=32):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels

        self.encoder = Encoder(
            frames=frames, n_mels=n_mels,
            dim=64, latent_dim=latent_dim, depth=2,
        )
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim)
        self.freq_weights = nn.Parameter(torch.linspace(0.8, 1.2, n_mels))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# =========================================================
# AUGMENTATION  — unchanged
# =========================================================
def augment(x, frames, n_mels):
    x = x + torch.randn_like(x) * 0.01
    scale = torch.empty(x.size(0), 1, device=x.device).uniform_(0.9, 1.1)
    x = x * scale
    shift = np.random.randint(-1, 2)
    if shift != 0:
        x = torch.roll(
            x.view(x.size(0), frames, n_mels), shifts=shift, dims=2
        ).view(x.size(0), -1)
    return x


# =========================================================
# CONFORMER MODEL
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

        # Per-domain Mahalanobis — fitted on reconstruction residuals
        self.mean_s    = None;  self.inv_cov_s = None
        self.mean_t    = None;  self.inv_cov_t = None

    # ----------------------------------------------------------
    def init_model(self):
        return ConformerNet(
            input_dim  = self.data.input_dim,
            frames     = self.args.frames,
            n_mels     = self.args.n_mels,
            latent_dim = 32,
        )

    # ----------------------------------------------------------
    # TRAIN  — unchanged
    # ----------------------------------------------------------
    def train(self, epoch):
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            x  = batch[0].to(self.device).float()
            x1 = augment(x, self.args.frames, self.args.n_mels)
            x2 = augment(x, self.args.frames, self.args.n_mels)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                recon1, z1 = self.model(x1)
                _,      z2 = self.model(x2)

                x2d = x.view(x.size(0), self.args.frames, self.args.n_mels)
                r2d = recon1.view(x.size(0), self.args.frames, self.args.n_mels)
                w   = F.softmax(self.model.freq_weights, dim=0).unsqueeze(0).unsqueeze(0)

                recon_loss  = ((x2d - r2d) ** 2 * w).mean()
                latent_loss = F.mse_loss(z1, z2)
                loss        = recon_loss + 0.05 * latent_loss

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        self.scheduler.step()
        print(f"[Epoch {epoch}] loss={total_loss / len(self.train_loader):.5f}")

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------
    @staticmethod
    def _build_cov(feats, shrinkage=0.05):
        if feats.size(0) < 2:
            feats = feats.repeat(2, 1)
        mean     = feats.mean(dim=0)
        centered = feats - mean
        cov      = (centered.T @ centered) / max(centered.size(0) - 1, 1)
        cov      = (1 - shrinkage) * cov + shrinkage * torch.eye(cov.size(0))
        return mean, torch.linalg.pinv(cov)

    @staticmethod
    def _oversample(arr, target_count):
        n = len(arr)
        if n < 2 or n >= target_count:
            return arr
        rng       = np.random.default_rng(seed=42)
        synthetic = []
        for _ in range(target_count - n):
            i, j  = rng.choice(n, size=2, replace=False)
            alpha = rng.uniform(0.0, 1.0)
            synthetic.append(alpha * arr[i] + (1 - alpha) * arr[j])
        return np.vstack([arr, np.array(synthetic, dtype=np.float32)])

    def _domain_from_batch(self, batch, batch_size):
        """Return int array (0=source, 1=target) for every sample in batch."""
        domains = None
        if len(batch) > 2:
            d = batch[2]
            try:
                domains = (d.cpu().numpy() if isinstance(d, torch.Tensor)
                           else np.array(d)).astype(int).reshape(-1)
            except Exception:
                domains = None
        if domains is None or len(np.unique(domains)) == 1:
            if len(batch) > 3:
                try:
                    domains = np.array(
                        [1 if "target" in str(f).lower() else 0
                         for f in batch[3]], dtype=int)
                except Exception:
                    domains = None
        if domains is None or len(domains) != batch_size:
            domains = np.zeros(batch_size, dtype=int)
        return domains

    def _supplemental_residuals(self):
        """
        Load supplemental WAV files, run them through the model,
        and return (recon - input) residuals as a CPU tensor.
        Returns None if the folder is not found or librosa is missing.
        """
        machine = (
            getattr(self.args, "dataset", "")
            .replace("DCASE2025T2", "").replace("DCASE2024T2", "").replace("DCASE2023T2", "")
        )
        candidates = [
            os.path.join("data", "dcase2025t2", "eval_data", "raw", machine, "supplemental"),
            os.path.join("data", "dcase2025t2", "dev_data",  "raw", machine, "supplemental"),
        ]
        supp_dir = next((d for d in candidates if os.path.isdir(d)), None)
        if supp_dir is None:
            print("[INFO] Supplemental folder not found; skipping.")
            return None

        wav_files = sorted(glob.glob(os.path.join(supp_dir, "*.wav")))
        if not wav_files:
            return None

        try:
            import librosa
        except ImportError:
            print("[WARN] librosa not installed; skipping supplemental data.")
            return None

        sr         = getattr(self.args, "sample_rate", 16000)
        n_fft      = getattr(self.args, "n_fft", 1024)
        hop_length = getattr(self.args, "hop_length", 512)
        power      = getattr(self.args, "power", 2.0)

        raw = []
        for wav_path in wav_files:
            try:
                y, _ = librosa.load(wav_path, sr=sr, mono=True)
                mel  = librosa.feature.melspectrogram(
                    y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
                    n_mels=self.args.n_mels, power=power,
                )
                log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
                for t in range(log_mel.shape[1] - self.args.frames + 1):
                    raw.append(log_mel[:, t:t + self.args.frames].T.flatten())
            except Exception as e:
                print(f"[WARN] {os.path.basename(wav_path)}: {e}")

        if not raw:
            return None

        feat_t = torch.from_numpy(np.array(raw, dtype=np.float32))
        self.model.eval()
        res_list = []
        with torch.inference_mode():
            for i in range(0, len(feat_t), 256):
                chunk = feat_t[i:i + 256].to(self.device)
                recon, _ = self.model(chunk)
                res_list.append((recon - chunk).cpu())

        result = torch.cat(res_list, dim=0)
        print(f"[INFO] Supplemental: {result.size(0)} residual vectors from {len(wav_files)} files.")
        return result

    # ----------------------------------------------------------
    # BUILD COVARIANCE — fitted on reconstruction residuals
    # ----------------------------------------------------------
    def build_covariance(self):
        """
        Collect (reconstruction - input) residuals from training data,
        split by domain, then fit two independent Mahalanobis models.

        Working in the 640-dimensional residual space (instead of the
        32-dimensional latent space) gives the covariance far more
        information about how each domain's reconstruction errors differ,
        which is exactly what the selective Mahalanobis scoring needs.
        """
        self.model.eval()
        res_s, res_t = [], []

        with torch.inference_mode():
            for batch in self.train_loader:
                if not batch:
                    continue
                try:
                    x        = batch[0].to(self.device).float()
                    recon, _ = self.model(x)
                    residual  = (recon - x).cpu()       # (B, input_dim=640)
                except Exception as e:
                    print(f"[WARN] batch skipped: {e}")
                    continue

                domains = self._domain_from_batch(batch, x.size(0))
                for i, dom in enumerate(domains):
                    if i < residual.size(0):
                        (res_t if dom == 1 else res_s).append(residual[i])

        # --- source fallback ---
        if len(res_s) == 0:
            print("[WARN] No source residuals; using target as fallback.")
            res_s = res_t if res_t else [torch.zeros(self.data.input_dim)]
        res_s = torch.stack(res_s)
        print(f"[INFO] Source residuals: {res_s.size(0)}")

        # --- target fallback ---
        if len(res_t) == 0:
            print("[WARN] No target residuals; using last 100 source vectors.")
            res_t = res_s[-min(100, res_s.size(0)):]
        else:
            res_t = torch.stack(res_t)
        print(f"[INFO] Target residuals before oversampling: {res_t.size(0)}")

        # --- supplemental → source pool ---
        supp = self._supplemental_residuals()
        if supp is not None:
            res_s = torch.cat([res_s, supp], dim=0)
            print(f"[INFO] Source after supplemental: {res_s.size(0)}")

        # --- oversample target ---
        target_count = min(res_s.size(0) // 4, 5000)
        res_t        = torch.from_numpy(
            self._oversample(res_t.numpy(), target_count)).float()
        print(f"[INFO] Target after oversampling: {res_t.size(0)}")

        self.mean_s, self.inv_cov_s = self._build_cov(res_s, shrinkage=0.05)
        self.mean_t, self.inv_cov_t = self._build_cov(res_t, shrinkage=0.05)
        for attr in ("mean_s", "inv_cov_s", "mean_t", "inv_cov_t"):
            setattr(self, attr, getattr(self, attr).to(self.device))
        print("[INFO] Residual-based selective Mahalanobis covariances built.")

    # ----------------------------------------------------------
    # TEST
    # ----------------------------------------------------------
    def test(self):
        self.model.eval()
        self.build_covariance()

        all_mse, all_mahal, all_labels, all_domains, all_names = [], [], [], [], []

        with torch.inference_mode():
            for section_loader in self.test_loader:
                for batch in section_loader:
                    x        = batch[0].to(self.device).float()
                    recon, _ = self.model(x)

                    # frequency-weighted MSE per sample
                    x2d = x.view(x.size(0), self.args.frames, self.args.n_mels)
                    r2d = recon.view(x.size(0), self.args.frames, self.args.n_mels)
                    w   = F.softmax(self.model.freq_weights, dim=0).unsqueeze(0).unsqueeze(0)
                    mse = ((x2d - r2d) ** 2 * w).mean(dim=(1, 2))

                    # selective Mahalanobis on residual per sample
                    try:
                        residual = recon - x                # (B, input_dim)
                        diff_s   = residual - self.mean_s
                        mahal_s  = torch.sum((diff_s @ self.inv_cov_s) * diff_s, dim=1)
                        diff_t   = residual - self.mean_t
                        mahal_t  = torch.sum((diff_t @ self.inv_cov_t) * diff_t, dim=1)
                        mahal    = torch.minimum(mahal_s, mahal_t)
                    except Exception as e:
                        print(f"[WARN] Mahalanobis failed ({e}); using MSE.")
                        mahal = mse

                    y     = batch[1].cpu().numpy() if len(batch) > 1 else np.zeros(x.size(0))
                    d     = batch[2].cpu().numpy() if len(batch) > 2 else np.zeros(x.size(0))
                    names = batch[3] if len(batch) > 3 else [f"s{i}" for i in range(x.size(0))]

                    all_mse.extend(mse.cpu().numpy())
                    all_mahal.extend(mahal.cpu().numpy())
                    all_labels.extend(y);  all_domains.extend(d);  all_names.extend(names)

        pd.DataFrame({
            "anon": all_names, "mse_score": all_mse,
            "mahal_score": all_mahal, "label": all_labels, "domain": all_domains,
        }).to_csv("conformer_scores.csv", index=False)

        print("\n[OK] Conformer complete  →  conformer_scores.csv")
        print(f"MSE   range: {min(all_mse):.6f} → {max(all_mse):.6f}")
        print(f"Mahal range: {min(all_mahal):.6f} → {max(all_mahal):.6f}")