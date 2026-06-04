import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.criterion.mahala import cov_v, loss_function_mahala, calc_inv_cov
from tools.plot_loss_curve import csv_to_figdata


# =========================================================
# RESIDUAL BLOCK
# =========================================================
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = self.norm1(x)
        x = F.gelu(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = torch.clamp(x, -3, 3)

        return x + residual


# =========================================================
# ENCODER
# =========================================================
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


# =========================================================
# DECODER
# =========================================================
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


# =========================================================
# CORE NETWORK
# =========================================================
class FRAENetV4(nn.Module):

    def __init__(self, input_dim, block_size, frames, n_mels, latent_dim=16):
        super().__init__()

        self.frames = frames
        self.n_mels = n_mels
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden=256,
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden=256,
            output_dim=input_dim,
        )

        self.freq_weights = nn.Parameter(torch.linspace(0.8, 1.5, n_mels))

        # Required by the repo's Mahalanobis utilities
        self.register_buffer("cov_source", torch.eye(block_size))
        self.register_buffer("cov_target", torch.eye(block_size))

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        x = torch.clamp(x, -5, 5)
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def freq_weighted_mse(self, x, recon):
        x_2d = x.view(x.size(0), self.frames, self.n_mels)
        r_2d = recon.view(recon.size(0), self.frames, self.n_mels)

        w = F.softmax(self.freq_weights, dim=0)
        w = w.unsqueeze(0).unsqueeze(0)

        sq_err = (x_2d - r_2d) ** 2
        return (sq_err * w).sum(dim=-1).mean()


# =========================================================
# VICREG LOSS
# =========================================================
def vicreg_loss(z1, z2):

    inv_loss = F.mse_loss(z1, z2)
    eps = 1e-4
    std_z1 = torch.sqrt(z1.var(dim=0, unbiased=False) + eps)
    std_z2 = torch.sqrt(z2.var(dim=0, unbiased=False) + eps)

    var_loss = (
        F.relu(1.0 - std_z1).mean() +
        F.relu(1.0 - std_z2).mean()
    )

    def covariance_loss(z):
        if z.size(0) < 2:
            return torch.tensor(0.0, device=z.device)

        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (z.size(0) - 1)

        off_diag = cov - torch.diag(torch.diagonal(cov))
        return (off_diag ** 2).mean()

    cov_loss = covariance_loss(z1) + covariance_loss(z2)

    return (
        25.0 * inv_loss +
        15.0 * var_loss +
        1.0 * cov_loss
    )


# =========================================================
# AUGMENTATION
# =========================================================
def augment(x, frames, n_mels):
    x2 = x.clone()

    scale = torch.empty(x2.size(0), 1, device=x.device).uniform_(0.85, 1.15)
    x2 = x2 * scale
    x2 = torch.clamp(x2, -5, 5)

    x2 = x2 + torch.randn_like(x2) * 0.01  # ↓ reduce noise

    x2 = torch.nan_to_num(x2, nan=0.0, posinf=1.0, neginf=-1.0)

    return x2
def safe_input(x):
    x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return torch.clamp(x, -5, 5)
    


# =========================================================
# FRAE V4
# =========================================================
class FRAEV4(DCASE2023T2AE):

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

        self.use_amp = bool(torch.cuda.is_available() and self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    # =====================================================
    # INIT MODEL
    # =====================================================
    def init_model(self):
        self.block_size = self.data.height
        print(f"[INFO] latent_dim = {getattr(self.args, 'latent_dim', 16)}")

        return FRAENetV4(
            input_dim=self.data.input_dim,
            block_size=self.block_size,
            frames=self.args.frames,
            n_mels=self.args.n_mels,
            latent_dim=getattr(self.args, "latent_dim", 16),
        )

    def get_log_header(self):
        self.column_heading_list = [
            ["loss"],
            ["val_loss"],
            ["recon_loss"],
            ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"

    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    def loss_fn(self, recon_x, x):
        """
        Frequency-weighted MSE.
        Returns unreduced tensor with shape (B, D).
        """
        x_2d = x.view(x.size(0), self.args.frames, self.args.n_mels)
        recon_2d = recon_x.view(recon_x.size(0), self.args.frames, self.args.n_mels)

        weights = F.softmax(self.model.freq_weights, dim=0)
        loss = ((x_2d - recon_2d) ** 2) * weights.unsqueeze(0).unsqueeze(0)

        return loss.view(loss.size(0), -1)

    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.to(self.device).float()
        recon_data, _ = self.model(data)

        loss_source, num = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_source,
            use_precision=True,
            reduction=False,
        )
        loss_source = self.loss_reduction(
            score=self.loss_reduction_1d(loss_source),
            n_loss=num,
        )

        loss_target, num = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_target,
            use_precision=True,
            reduction=False,
        )
        loss_target = self.loss_reduction(
            score=self.loss_reduction_1d(loss_target),
            n_loss=num,
        )

        y_pred.append(0.5 * (loss_source.item() + loss_target.item()))
        return y_pred

    # =====================================================
    # TRAIN
    # Baseline protocol + FRAE v4 objective
    # =====================================================
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

        if epoch == self.args.epochs + 3:
            print("\n============== CALCULATE COVARIANCE ==============")
            is_calc_cov = True
            self.model.eval()

            cov_x_source = torch.zeros(
                (self.block_size, self.block_size),
                device=self.device,
                dtype=torch.float32,
            )
            cov_x_target = cov_x_source.clone().detach()
            num_source = 0
            num_target = 0
            epoch = self.args.epochs
        else:
            self.model.train()
            is_calc_cov = False

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0].to(self.device).float()

            if data.shape[0] <= 1:
                continue

            data_name_list = batch[3]
            is_target_list = ["target" in data_name for data_name in data_name_list]
            is_source_list = np.logical_not(is_target_list).tolist()
            n_source = is_source_list.count(True)
            n_target = is_target_list.count(True)

            if not is_calc_cov:
                self.optimizer.zero_grad(set_to_none=True)

            if is_calc_cov:
                with torch.no_grad():
                    recon_batch, _ = self.model(data)

                    score_2d, cov_diff_source, cov_diff_target = loss_function_mahala(
                        recon_x=recon_batch,
                        x=data,
                        block_size=self.block_size,
                        update_cov=True,
                        reduction=False,
                        is_source_list=is_source_list,
                        is_target_list=is_target_list,
                    )

                    cov_x_source_batch = cov_v(
                        diff=cov_diff_source,
                        num=1,
                    )
                    cov_x_source += cov_x_source_batch.clone().detach()
                    num_source += n_source

                    if n_target > 0:
                        cov_x_target_batch = cov_v(
                            diff=cov_diff_target,
                            num=1,
                        )
                        cov_x_target += cov_x_target_batch.clone().detach()
                        num_target += n_target

            else:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    recon_batch, _ = self.model(data)

                    # Baseline-style score for logging / distribution fitting
                    score_2d = self.loss_fn(recon_batch, data)

                    n_loss = len(score_2d)
                    score = self.loss_reduction_1d(score=score_2d)

                    recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
                    recon_loss_source = self.loss_reduction(
                        score=score[is_source_list],
                        n_loss=n_source,
                    )
                    if n_target > 0:
                        recon_loss_target = self.loss_reduction(
                            score=score[is_target_list],
                            n_loss=n_target,
                        )
                    else:
                        recon_loss_target = 0

                    x1 = safe_input(augment(data, self.args.frames, self.args.n_mels))
                    x2 = safe_input(augment(data, self.args.frames, self.args.n_mels))

                    z1 = self.model.encoder(x1)
                    z2 = self.model.encoder(x2)
                    z1 = torch.nan_to_num(z1, nan=0.0, posinf=1.0, neginf=-1.0)
                    z2 = torch.nan_to_num(z2, nan=0.0, posinf=1.0, neginf=-1.0)
                    

                    vic_loss = vicreg_loss(z1.float(), z2.float())
                    latent_penalty = torch.mean(torch.abs(torch.clamp(z1, -5, 5)))
                    latent_penalty = 1e-5 * latent_penalty

                    if n_source > 0 and n_target > 0:
                        x_source = data[is_source_list]
                        x_target = data[is_target_list]
                        z_source = self.model.encoder(x_source)
                        z_target = self.model.encoder(x_target)
                        domain_loss = F.mse_loss(
                            z_source.mean(dim=0),
                             z_target.mean(dim=0)
                        )
                    else:
                        domain_loss = 0.0

                    progress = min(
                        epoch / max(getattr(self.args, "epochs", 50), 1),
                        1.0
                    )
                    vic_weight = 0.0 if epoch < 3 else 0.002 * progress

                    train_obj = (
                        recon_loss +
                        vic_weight * vic_loss +
                        0.0002 * latent_penalty
                    )

                self.loss = recon_loss

                self.scaler.scale(train_obj).backward()
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=5.0,
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                train_loss += float(self.loss)
                train_recon_loss += float(recon_loss)
                train_recon_loss_source += float(recon_loss_source)
                train_recon_loss_target += float(recon_loss_target)

                y_pred.append(self.loss.item())

                if batch_idx % self.args.log_interval == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            self.loss.item()
                        )
                    )
                continue

            # covariance pass bookkeeping
            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score=score_2d)

            recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
            recon_loss_source = self.loss_reduction(
                score=score[is_source_list],
                n_loss=n_source,
            )
            if n_target > 0:
                recon_loss_target = self.loss_reduction(
                    score=score[is_target_list],
                    n_loss=n_target,
                )
            else:
                recon_loss_target = 0

            self.loss = recon_loss

            train_loss += float(self.loss)
            train_recon_loss += float(recon_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)

            y_pred.append(self.loss.item())

        if is_calc_cov:
            cov_x_source /= max(num_source - 1, 1)
            if num_target == 0:
                cov_x_target = cov_x_source.clone().detach()
            else:
                cov_x_target /= max(num_target - 1, 1)

            self.model.cov_source.data = cov_x_source
            self.model.cov_target.data = cov_x_target

            inv_cov_source, inv_cov_target = calc_inv_cov(
                model=self.model,
                device=self.device,
            )

            y_pred_mahala = []
            for _, batch in enumerate(tqdm(train_loader)):
                y_pred_mahala = self.calc_valid_mahala_score(
                    data=batch[0],
                    y_pred=y_pred_mahala,
                    inv_cov_source=inv_cov_source,
                    inv_cov_target=inv_cov_target,
                )
            for _, batch in enumerate(self.valid_loader):
                y_pred_mahala = self.calc_valid_mahala_score(
                    data=batch[0],
                    y_pred=y_pred_mahala,
                    inv_cov_source=inv_cov_source,
                    inv_cov_target=inv_cov_target,
                )

            self.fit_anomaly_score_distribution(
                y_pred=y_pred_mahala,
                score_distr_file_path=self.mahala_score_distr_file_path,
            )

        # validation test
        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            for _, batch in enumerate(self.valid_loader):
                data = batch[0].to(self.device).float()

                recon_batch, _ = self.model(data)
                score = self.loss_fn(
                    recon_batch,
                    data
                )
                loss = score.mean()

                val_loss += float(loss)
                y_pred.append(loss.item())

        if not is_calc_cov:
            print(
                '====> Epoch: {} Average loss: {:.4f} Validation loss: {:.4f}'.format(
                    epoch,
                    train_loss / len(train_loader),
                    val_loss / len(self.valid_loader),
                )
            )

            with open(self.log_path, 'a') as log:
                np.savetxt(
                    log,
                    ["{0},{1},{2},{3},{4}".format(
                        train_loss / len(train_loader),
                        val_loss / len(self.valid_loader),
                        train_recon_loss / len(train_loader),
                        train_recon_loss_source / len(train_loader),
                        train_recon_loss_target / len(train_loader),
                    )],
                    fmt="%s",
                )

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

        # save model
        torch.save(self.model.state_dict(), self.model_path)
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss,
            },
            self.checkpoint_path,
        )


# aliases
FRAE = FRAEV4
FRAEV2 = FRAEV4
FRAEV3 = FRAEV4