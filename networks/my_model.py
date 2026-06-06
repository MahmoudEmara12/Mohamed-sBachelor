
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



def augment(x, frames, n_mels):
    """Light SpecAugment-style augmentation."""
    scale = torch.empty(x.size(0), 1, device=x.device).uniform_(0.85, 1.15)
    x = x * scale
    x = x + torch.randn_like(x) * 0.02

    roll = np.random.randint(-2, 3)
    if roll != 0:
        x_3d = x.reshape(x.size(0), frames, n_mels)
        x_3d = torch.roll(x_3d, roll, dims=2)
        x = x_3d.reshape(x.size(0), -1)

    return x



class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x + residual



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden=512, latent_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x):
        h = self.proj(x)
        h = self.blocks(h)
        return self.head(h)



class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden=512, output_dim=640, skip_dim=32):
        super().__init__()
        in_dim = latent_dim + skip_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            ResBlock(hidden),
            ResBlock(hidden),
        )
        self.head = nn.Linear(hidden, output_dim)

    def forward(self, z, skip):
        x = torch.cat([z, skip], dim=-1)
        h = self.proj(x)
        h = self.blocks(h)
        return self.head(h)



class FRAMENet(nn.Module):
    def __init__(self, input_dim, frames, n_mels, cov_dim, latent_dim=64, hidden=512):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels

        skip_dim = 32
        self.skip_proj = nn.Sequential(
            nn.Linear(input_dim, skip_dim),
            nn.GELU(),
        )

        self.encoder = Encoder(
            input_dim=input_dim,
            hidden=hidden,
            latent_dim=latent_dim,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden=hidden,
            output_dim=input_dim,
            skip_dim=skip_dim,
        )

        self.freq_weights = nn.Parameter(torch.linspace(0.8, 1.5, n_mels))
        self.register_buffer("cov_source", torch.eye(cov_dim))
        self.register_buffer("cov_target", torch.eye(cov_dim))

    def forward(self, x):
        skip = self.skip_proj(x)
        z = self.encoder(x)
        recon = self.decoder(z, skip)
        return recon, z



class FRAME(DCASE2023T2AE):
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
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

    def init_model(self):
        latent_dim = getattr(self.args, "latent_dim", 64)
        hidden = getattr(self.args, "hidden", 512)

        print(f"[FRAME] latent_dim={latent_dim} hidden={hidden}")

        self.block_size = self.data.height

        return FRAMENet(
            input_dim=self.data.input_dim,
            frames=self.args.frames,
            n_mels=self.args.n_mels,
            cov_dim=self.block_size,
            latent_dim=latent_dim,
            hidden=hidden,
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

    def _freq_weights(self):
        return F.softmax(self.model.freq_weights.detach(), dim=0)

    def _weighted_flat(self, x):
        x_2d = x.reshape(x.size(0), self.args.frames, self.args.n_mels)
        weights = self._freq_weights().view(1, 1, -1)
        return (x_2d * weights).reshape(x.size(0), -1)

    def _weighted_residual(self, recon_x, x):
        return self._weighted_flat(x) - self._weighted_flat(recon_x)

    def loss_fn(self, recon_x, x):
        """
        Frequency-weighted MSE.
        Must return unreduced tensor with shape (B, D).
        """
        x_2d = x.reshape(x.size(0), self.args.frames, self.args.n_mels)
        recon_2d = recon_x.reshape(recon_x.size(0), self.args.frames, self.args.n_mels)

        weights = F.softmax(self.model.freq_weights, dim=0)
        loss = ((x_2d - recon_2d) ** 2) * weights.unsqueeze(0).unsqueeze(0)

        return loss.reshape(loss.size(0), -1)

    def _cov_from_rows(self, rows, device):
        """
        Build an unbiased covariance from a list of 2D tensors with shape (N_i, D).
        Returns a (D, D) matrix.
        """
        if len(rows) == 0:
            return torch.eye(self.block_size, device=device, dtype=torch.float32)

        mat = torch.cat(rows, dim=0).to(device=device, dtype=torch.float32)
        if mat.size(0) <= 1:
            return torch.eye(self.block_size, device=device, dtype=torch.float32)

        mu = torch.mean(mat.t(), 1)
        diff = torch.sub(mat, mu)
        cov = cov_v(diff, mat.size(0))

        cov = cov + 1e-6 * torch.eye(cov.size(0), device=device, dtype=torch.float32)
        return cov

    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.to(self.device).float()
        recon_data, _ = self.model(data)

        data_w = self._weighted_flat(data)
        recon_w = self._weighted_flat(recon_data)

        loss_source, num = loss_function_mahala(
            recon_x=recon_w,
            x=data_w,
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
            recon_x=recon_w,
            x=data_w,
            block_size=self.block_size,
            cov=inv_cov_target,
            use_precision=True,
            reduction=False,
        )
        loss_target = self.loss_reduction(
            score=self.loss_reduction_1d(loss_target),
            n_loss=num,
        )

        y_pred.append(min(loss_target.item(), loss_source.item()))
        return y_pred

    def train(self, epoch):
        if epoch <= self.epoch:
            return

        train_loss = 0
        train_recon_loss = 0
        train_recon_loss_source = 0
        train_recon_loss_target = 0
        y_pred = []

        train_loader = self.train_loader

        
        if epoch == self.args.epochs + 1:
            print("\n============== CALCULATE COVARIANCE ==============")
            is_calc_cov = True
            self.model.eval()
            torch.set_grad_enabled(False)

            cov_x_source = torch.zeros(
                (self.block_size, self.block_size),
                device=self.device,
                dtype=torch.float32,
            )
            cov_x_target = cov_x_source.clone().detach()
            num_source = 0
            num_target = 0
            epoch = self.args.epochs

            source_rows = []
            target_rows = []
        else:
            self.model.train()
            is_calc_cov = False

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0].to(self.device).float()
            if data.shape[0] <= 1:
                continue

            data_name_list = batch[3]
            machine_id = torch.argmax(batch[2], dim=1).long().to(self.device)

            is_target_list = torch.tensor(
                ["target" in data_name for data_name in data_name_list],
                device=self.device,
                dtype=torch.bool,
            )
            is_source_list = ~is_target_list
            n_source = int(is_source_list.sum().item())
            n_target = int(is_target_list.sum().item())

            if not is_calc_cov:
                self.optimizer.zero_grad()

            recon_batch, z = self.model(data)

            if is_calc_cov:
                loss_2d, cov_diff_source, cov_diff_target = loss_function_mahala(
                    recon_x=recon_batch,
                    x=data,
                    block_size=self.block_size,
                    is_source_list=is_source_list,
                    is_target_list=is_target_list,
                    update_cov=True,
                    reduction=False,
                )

                if cov_diff_source is not None and cov_diff_source.numel() > 0:
                    source_rows.append(cov_diff_source.detach().cpu())
                    num_source += n_source

                if cov_diff_target is not None and cov_diff_target.numel() > 0:
                    target_rows.append(cov_diff_target.detach().cpu())
                    num_target += n_target

                score_2d = loss_2d
            else:
                score_2d = self.loss_fn(recon_batch, data)

            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score=score_2d)

            recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
            recon_loss_source = (
                self.loss_reduction(score=score[is_source_list], n_loss=n_source)
                if n_source > 0
                else torch.tensor(0.0, device=self.device)
            )
            recon_loss_target = (
                self.loss_reduction(score=score[is_target_list], n_loss=n_target)
                if n_target > 0
                else torch.tensor(0.0, device=self.device)
            )

            self.loss = recon_loss

            if not is_calc_cov:
                # extra FRAME regularization only in normal training phase
                x1 = augment(data, self.args.frames, self.args.n_mels)
                x2 = augment(data, self.args.frames, self.args.n_mels)

                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    recon1, z1 = self.model(x1)
                    recon2, z2 = self.model(x2)

                    
                    z1 = torch.nan_to_num(z1, nan=0.0, posinf=1e4, neginf=-1e4)
                    z2 = torch.nan_to_num(z2, nan=0.0, posinf=1e4, neginf=-1e4)

                    loss1 = self.loss_fn(recon1, data).mean()
                    loss2 = self.loss_fn(recon2, data).mean()
                    frame_recon_loss = 0.5 * (loss1 + loss2)

                    consistency_loss = F.mse_loss(z1, z2)

                
                    z1_var = torch.var(z1, dim=0, unbiased=False)
                    z1_var = torch.nan_to_num(z1_var, nan=0.0, posinf=0.0, neginf=0.0)
                    z2_var = torch.var(z2, dim=0, unbiased=False)
                    z2_var = torch.nan_to_num(z2_var, nan=0.0, posinf=0.0, neginf=0.0)

                    std1 = torch.sqrt(z1_var + 1e-4)
                    std2 = torch.sqrt(z2_var + 1e-4)
                    var_loss = F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()

                    self.loss = frame_recon_loss + 0.01 * consistency_loss + 0.005 * var_loss

                self.scaler.scale(self.loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()

            train_loss += float(self.loss)
            train_recon_loss += float(recon_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)

            y_pred.append(self.loss.item())

            if batch_idx % self.args.log_interval == 0 and not is_calc_cov:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        self.loss.item(),
                    )
                )

        if is_calc_cov:
            self.model.cov_source.data = self._cov_from_rows(source_rows, self.device)
            if len(target_rows) == 0:
                self.model.cov_target.data = self.model.cov_source.data.clone().detach()
            else:
                self.model.cov_target.data = self._cov_from_rows(target_rows, self.device)

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

        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            for _, batch in enumerate(self.valid_loader):
                data = batch[0].to(self.device).float()
                recon_batch, _ = self.model(data)
                score = self.loss_fn(recon_batch, data)
                loss = score.mean()
                val_loss += float(loss)
                y_pred.append(loss.item())

        if not is_calc_cov:
            print(
                "====> Epoch: {} Average loss: {:.4f} Validation loss: {:.4f}".format(
                    epoch,
                    train_loss / len(train_loader),
                    val_loss / len(self.valid_loader),
                )
            )

            with open(self.log_path, "a") as log:
                np.savetxt(
                    log,
                    [f"{train_loss / len(train_loader)},{val_loss / len(self.valid_loader)},{train_recon_loss / len(train_loader)},{train_recon_loss_source / len(train_loader)},{train_recon_loss_target / len(train_loader)}"],
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

        torch.save(self.model.state_dict(), self.model_path)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss": self.loss,
            },
            self.checkpoint_path,
        )

