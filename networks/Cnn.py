"""
CFMA — Convolutional Frequency-Masked Autoencoder
==================================================

Drop-in model file for the DCASE2023 Task 2 repo.

KNOWN ISSUES OBSERVED IN FRAME — fixed here:
  1. Boolean list indexing:
       score[is_source_list] where is_source_list is a Python bool list
       → PyTorch treats True/False as int 1/0, doing integer indexing
         instead of boolean masking. Fixed: wrap in torch.tensor(..., dtype=torch.bool)

  2. Covariance/scoring space: the covariance is accumulated from UNWEIGHTED
       diff = x - recon_x reshaped to (-1, n_mels). The Mahalanobis scoring
       also uses unweighted residuals (via loss_function_mahala). Consistent.
       The MSE score uses frequency-weighted residuals via loss_fn. Also
       consistent within itself. The two scores live in different spaces —
       that is expected and fine. Documented explicitly below.

  3. Mahalanobis (N×N) matrix: loss_function_mahala with reduction=False
       returns the full cross-distance matrix (N×N), not per-sample diagonals.
       loss_reduction_1d takes mean(dim=1) = row means, not diagonal.
       This is the same behaviour as the baseline — left unchanged for
       consistency/comparability, but documented.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.criterion.mahala import cov_v, loss_function_mahala, calc_inv_cov
from tools.plot_loss_curve import csv_to_figdata


# =========================================================
# HELPERS
# =========================================================

def _to_bool_mask(flag_list, device):
    """
    Convert a Python bool list to a torch bool tensor for correct masking.

    WHY THIS EXISTS:
    PyTorch fancy-indexes with Python bool lists by treating True→1, False→0
    as INTEGER indices, not as a boolean mask.
      e.g.  score[[True, False, True]] == score[[1, 0, 1]]  ← WRONG
    Passing a torch.bool tensor instead gives proper boolean masking.
    """
    return torch.tensor(flag_list, dtype=torch.bool, device=device)


def augment(x, frames, n_mels):
    """Amplitude jitter + gaussian noise + frequency roll."""
    scale = torch.empty(x.size(0), 1, device=x.device).uniform_(0.9, 1.1)
    x = x * scale
    x = x + torch.randn_like(x) * 0.01
    shift = np.random.randint(-1, 2)
    if shift != 0:
        x = x.view(x.size(0), frames, n_mels)
        x = torch.roll(x, shifts=shift, dims=2)
        x = x.view(x.size(0), -1)
    return x


# =========================================================
# ENCODER
# =========================================================

class Encoder(nn.Module):
    def __init__(self, frames, n_mels, latent_dim=32):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.embedding = nn.Sequential(
            nn.Linear(128, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x):
        B = x.size(0)
        x = x.view(B, 1, self.frames, self.n_mels)
        h = self.net(x).view(B, -1)
        return self.embedding(h)
        # NOTE: No F.normalize here.
        # FRAME uses unnormalized z; we match that so the Mahalanobis
        # covariance (built on x - recon_x, not on z) is comparable.


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
# CFMA NETWORK
# =========================================================

class CFMANet(nn.Module):
    """
    PyTorch module.

    Framework contracts:
      - forward() returns (recon, z)
      - cov_source, cov_target are registered buffers of shape (n_mels, n_mels)
        so calc_inv_cov() can read them.

    Mahalanobis space note:
      The covariance matrices live in n_mels-dimensional space because
      loss_function_mahala reshapes residuals to (-1, block_size=n_mels).
      This is the same space the baseline AENet uses.
    """

    def __init__(self, input_dim, frames, n_mels, latent_dim=32):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels

        self.encoder = Encoder(frames, n_mels, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

        # Frequency weights used only for the MSE score path.
        # The Mahalanobis path uses raw residuals (see loss_function_mahala).
        self.freq_weights = nn.Parameter(torch.linspace(0.8, 1.2, n_mels))

        # Required by calc_inv_cov — shape (n_mels, n_mels)
        self.register_buffer("cov_source", torch.eye(n_mels))
        self.register_buffer("cov_target", torch.eye(n_mels))

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# =========================================================
# CNN MODEL  (framework wrapper)
# =========================================================

class CNN(DCASE2023T2AE):
    """
    CFMA wrapped as a DCASE2023 Task 2 model.
    Inherits test / eval / ROC / pAUC / threshold / CSV logic from DCASE2023T2AE.
    """

    def __init__(self, args, train=True, test=False):
        super().__init__(args=args, train=train, test=test)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=1e-4,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.args.epochs, 1),
            eta_min=self.args.learning_rate * 0.05,
        )
        self.use_amp = bool(torch.cuda.is_available() and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    # ----------------------------------------------------------
    # init_model — called by BaseModel.__init__
    # ----------------------------------------------------------

    def init_model(self):
        latent_dim = getattr(self.args, "latent_dim", 32)
        # block_size = n_mels: the dimension loss_function_mahala reshapes to.
        self.block_size = self.data.height
        print(f"[CNN/CFMA] latent_dim={latent_dim}  block_size={self.block_size}")
        return CFMANet(
            input_dim=self.data.input_dim,
            frames=self.args.frames,
            n_mels=self.args.n_mels,
            latent_dim=latent_dim,
        )

    # ----------------------------------------------------------
    # Log header
    # ----------------------------------------------------------

    def get_log_header(self):
        self.column_heading_list = [
            ["loss"], ["val_loss"], ["recon_loss"],
            ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"

    # ----------------------------------------------------------
    # Loss helpers
    # loss_fn:  WEIGHTED MSE  → (B, frames*n_mels)   [MSE score path]
    # The Mahalanobis path uses UNWEIGHTED residuals inside
    # loss_function_mahala — that is consistent with how cov is built.
    # ----------------------------------------------------------

    def loss_fn(self, recon_x, x):
        """
        Frequency-weighted MSE. Returns unreduced (B, frames*n_mels).
        Used for the MSE anomaly score and training loss.

        NOTE on space: weighted by freq_weights over the n_mels axis.
        The Mahalanobis covariance is built on UNWEIGHTED residuals
        (inside loss_function_mahala). Do NOT mix scores across score
        types in comparisons — use --score MSE or --score MAHALA
        consistently.
        """
        x_2d = x.view(x.size(0), self.args.frames, self.args.n_mels)
        r_2d = recon_x.view(recon_x.size(0), self.args.frames, self.args.n_mels)
        weights = F.softmax(self.model.freq_weights, dim=0)
        loss = ((x_2d - r_2d) ** 2) * weights.unsqueeze(0).unsqueeze(0)
        return loss.view(loss.size(0), -1)   # (B, frames*n_mels)

    def loss_reduction_1d(self, score):
        """(B, D) → (B,)  or  (N, N) → (N,)  [row-means, same as baseline]"""
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        """(B,) → scalar"""
        return torch.sum(score) / n_loss

    # ----------------------------------------------------------
    # Mahalanobis helper
    # ----------------------------------------------------------

    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.to(self.device).float()
        recon_data, _ = self.model(data)

        # Mahalanobis uses UNWEIGHTED residuals via loss_function_mahala —
        # same space the covariance was built in. Consistent.
        loss_source, num = loss_function_mahala(
            recon_x=recon_data, x=data,
            block_size=self.block_size,
            cov=inv_cov_source, use_precision=True, reduction=False,
        )
        loss_source = self.loss_reduction(
            self.loss_reduction_1d(loss_source), n_loss=num
        )

        loss_target, num = loss_function_mahala(
            recon_x=recon_data, x=data,
            block_size=self.block_size,
            cov=inv_cov_target, use_precision=True, reduction=False,
        )
        loss_target = self.loss_reduction(
            self.loss_reduction_1d(loss_target), n_loss=num
        )

        y_pred.append(min(loss_target.item(), loss_source.item()))
        return y_pred

    # ----------------------------------------------------------
    # TRAIN
    # ----------------------------------------------------------

    def train(self, epoch):
        if epoch <= self.epoch:
            return

        torch.autograd.set_detect_anomaly(True)

        train_loss = 0
        train_recon_loss = 0
        train_recon_loss_source = 0
        train_recon_loss_target = 0
        y_pred = []

        train_loader = self.train_loader

        # ---- covariance pass ----
        if epoch == self.args.epochs + 1:
            print("\n============== CALCULATE COVARIANCE ==============")
            is_calc_cov = True
            self.model.eval()
            torch.set_grad_enabled(False)
            cov_x_source = torch.zeros(
                (self.block_size, self.block_size),
                device=self.device, dtype=torch.float32,
            )
            cov_x_target = cov_x_source.clone().detach()
            num_source = 0
            num_target = 0
            epoch = self.args.epochs
        else:
            self.model.train()
            is_calc_cov = False

        # ---- main loop ----
        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0].to(self.device).float()
            if data.shape[0] <= 1:
                continue

            data_name_list = batch[3]
            is_target_list_py = ["target" in n for n in data_name_list]
            is_source_list_py = np.logical_not(is_target_list_py).tolist()
            n_source = is_source_list_py.count(True)
            n_target = is_target_list_py.count(True)

            # FIX (Issue 1): convert Python bool lists to torch bool tensors
            # BEFORE any tensor indexing. Python bool lists used as tensor
            # indices are treated as integers (True→1, False→0), causing
            # wrong rows to be selected.
            src_mask = _to_bool_mask(is_source_list_py, device=self.device)
            tgt_mask = _to_bool_mask(is_target_list_py, device=self.device)

            if not is_calc_cov:
                self.optimizer.zero_grad()

            recon_batch, z = self.model(data)

            # ---- covariance accumulation ----
            if is_calc_cov:
                # loss_function_mahala uses raw diff = x - recon_x (UNWEIGHTED).
                # This sets the space for the Mahalanobis score. Consistent.
                # Pass Python lists here because loss_function_mahala does its
                # own indexing internally (diff[is_source_list]).
                # We pass the original Python bool lists — same bug exists
                # inside loss_function_mahala itself (not our code to fix),
                # but we document it. The cov_diff_* tensors come back already
                # indexed, so we apply our fix to everything we index ourselves.
                score_2d, cov_diff_source, cov_diff_target = loss_function_mahala(
                    recon_x=recon_batch, x=data,
                    block_size=self.block_size,
                    update_cov=True, reduction=False,
                    is_source_list=is_source_list_py,   # list — used inside mahala
                    is_target_list=is_target_list_py,
                )
                cov_x_source += cov_v(diff=cov_diff_source, num=1).clone().detach()
                num_source += n_source
                if n_target > 0:
                    cov_x_target += cov_v(diff=cov_diff_target, num=1).clone().detach()
                    num_target += n_target

            # ---- normal training ----
            else:
                score_2d = self.loss_fn(recon_batch, data)

                x1 = augment(data, self.args.frames, self.args.n_mels)
                x2 = augment(data, self.args.frames, self.args.n_mels)

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    recon1, z1 = self.model(x1)
                    recon2, z2 = self.model(x2)

                    aug_loss = 0.5 * (
                        self.loss_fn(recon1, data).mean() +
                        self.loss_fn(recon2, data).mean()
                    )
                    consistency_loss = F.mse_loss(z1, z2)
                    self.loss = aug_loss + 0.05 * consistency_loss

                self.scaler.scale(self.loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # ---- bookkeeping (FIX: use bool tensor masks, not Python lists) ----
            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score_2d)   # (B,)

            recon_loss = self.loss_reduction(score, n_loss=n_loss)

            # src_mask / tgt_mask are torch.bool tensors → correct boolean masking
            recon_loss_source = self.loss_reduction(
                score[src_mask], n_loss=max(n_source, 1)
            )
            recon_loss_target = (
                self.loss_reduction(score[tgt_mask], n_loss=n_target)
                if n_target > 0 else 0
            )

            if is_calc_cov:
                self.loss = recon_loss

            train_loss += float(self.loss)
            train_recon_loss += float(recon_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)
            y_pred.append(self.loss.item())

            if batch_idx % self.args.log_interval == 0 and not is_calc_cov:
                print("Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data), len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    self.loss.item(),
                ))

        # ---- finalise covariance ----
        if is_calc_cov:
            cov_x_source /= max(num_source - 1, 1)
            if num_target == 0:
                cov_x_target = cov_x_source.clone().detach()
            else:
                cov_x_target /= max(num_target - 1, 1)

            self.model.cov_source.data = cov_x_source
            self.model.cov_target.data = cov_x_target

            inv_cov_source, inv_cov_target = calc_inv_cov(
                model=self.model, device=self.device
            )

            y_pred_mahala = []
            for _, batch in enumerate(tqdm(train_loader)):
                y_pred_mahala = self.calc_valid_mahala_score(
                    batch[0], y_pred_mahala, inv_cov_source, inv_cov_target
                )
            for _, batch in enumerate(self.valid_loader):
                y_pred_mahala = self.calc_valid_mahala_score(
                    batch[0], y_pred_mahala, inv_cov_source, inv_cov_target
                )
            self.fit_anomaly_score_distribution(
                y_pred=y_pred_mahala,
                score_distr_file_path=self.mahala_score_distr_file_path,
            )

        # ---- validation ----
        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            for _, batch in enumerate(self.valid_loader):
                data = batch[0].to(self.device).float()
                recon_batch, _ = self.model(data)
                loss = self.loss_fn(recon_batch, data).mean()
                val_loss += float(loss)
                y_pred.append(loss.item())

        # ---- logging + checkpoint ----
        if not is_calc_cov:
            self.scheduler.step()
            print("====> Epoch: {} Average loss: {:.4f} Validation loss: {:.4f}".format(
                epoch,
                train_loss / len(train_loader),
                val_loss / len(self.valid_loader),
            ))
            with open(self.log_path, "a") as log:
                np.savetxt(log, ["{},{},{},{},{}".format(
                    train_loss / len(train_loader),
                    val_loss / len(self.valid_loader),
                    train_recon_loss / len(train_loader),
                    train_recon_loss_source / len(train_loader),
                    train_recon_loss_target / len(train_loader),
                )], fmt="%s")
            csv_to_figdata(
                file_path=self.log_path,
                column_heading_list=self.column_heading_list,
                ylabel="loss",
                fig_count=len(self.column_heading_list),
                cut_first_epoch=True,
            )
            self.fit_anomaly_score_distribution(
                y_pred=y_pred,
                score_distr_file_path=self.mse_score_distr_file_path,
            )

        torch.save(self.model.state_dict(), self.model_path)
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": self.loss,
        }, self.checkpoint_path)
