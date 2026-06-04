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
# CONFORMER BLOCK
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
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
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
        return z


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
# CONFORMER NETWORK
# =========================================================
class ConformerNet(nn.Module):

    def __init__(self, input_dim, frames, n_mels, latent_dim=32):
        super().__init__()
        self.frames = frames
        self.n_mels = n_mels

        self.encoder = Encoder(
            frames=frames,
            n_mels=n_mels,
            dim=64,
            latent_dim=latent_dim,
            depth=2,
        )
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=input_dim)

        self.freq_weights = nn.Parameter(torch.linspace(0.8, 1.2, n_mels))

        # Required by the DCASE Mahalanobis utilities
        self.register_buffer("cov_source", torch.eye(n_mels))
        self.register_buffer("cov_target", torch.eye(n_mels))

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# =========================================================
# AUGMENTATION
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
# CONFORMER MODEL
# =========================================================
class Conformer(DCASE2023T2AE):

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

    # -----------------------------------------------------
    # INIT MODEL
    # -----------------------------------------------------
    def init_model(self):
        latent_dim = getattr(self.args, "latent_dim", 32)

        print(f"[Conformer] latent_dim={latent_dim}")

        self.block_size = self.data.height

        return ConformerNet(
            input_dim=self.data.input_dim,
            frames=self.args.frames,
            n_mels=self.args.n_mels,
            latent_dim=latent_dim,
        )

    # -----------------------------------------------------
    # LOG HEADER
    # -----------------------------------------------------
    def get_log_header(self):
        self.column_heading_list = [
            ["loss"],
            ["val_loss"],
            ["recon_loss"],
            ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"

    # -----------------------------------------------------
    # REDUCTIONS
    # -----------------------------------------------------
    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    # -----------------------------------------------------
    # LOSS
    # Frequency-weighted MSE, returned unreduced as (B, D)
    # -----------------------------------------------------
    def loss_fn(self, recon_x, x):
        x_2d = x.view(x.size(0), self.args.frames, self.args.n_mels)
        recon_2d = recon_x.view(recon_x.size(0), self.args.frames, self.args.n_mels)

        weights = F.softmax(self.model.freq_weights, dim=0)
        loss = ((x_2d - recon_2d) ** 2) * weights.unsqueeze(0).unsqueeze(0)

        return loss.view(loss.size(0), -1)

    # -----------------------------------------------------
    # MAHALANOBIS SCORE ON VALID DATA
    # -----------------------------------------------------
    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.to(self.device).float()
        recon_data, _ = self.model(data)

        loss_source, num = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_source,
            use_precision=True,
            reduction=False
        )
        loss_source = self.loss_reduction(
            score=self.loss_reduction_1d(loss_source),
            n_loss=num
        )

        loss_target, num = loss_function_mahala(
            recon_x=recon_data,
            x=data,
            block_size=self.block_size,
            cov=inv_cov_target,
            use_precision=True,
            reduction=False
        )
        loss_target = self.loss_reduction(
            score=self.loss_reduction_1d(loss_target),
            n_loss=num
        )

        y_pred.append(min(loss_target.item(), loss_source.item()))
        return y_pred

    # -----------------------------------------------------
    # TRAIN
    # Baseline protocol + Conformer objective
    # -----------------------------------------------------
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

        if epoch == self.args.epochs + 1:
            print("\n============== CALCULATE COVARIANCE ==============")
            is_calc_cov = True
            self.model.eval()
            torch.set_grad_enabled(False)

            cov_x_source = np.zeros((self.block_size, self.block_size))
            cov_x_source = torch.from_numpy(cov_x_source)
            cov_x_source = cov_x_source.to(self.device).float()
            cov_x_target = cov_x_source.clone().detach()

            num_source = 0
            num_target = 0
            epoch = self.args.epochs
        else:
            self.model.train()
            is_calc_cov = False

        for batch_idx, batch in enumerate(tqdm(train_loader)):
            data = batch[0]
            data = data.to(self.device).float()

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
                recon_batch, _ = self.model(data)

                score_2d, cov_diff_source, cov_diff_target = loss_function_mahala(
                    recon_x=recon_batch,
                    x=data,
                    block_size=self.block_size,
                    update_cov=True,
                    reduction=False,
                    is_source_list=is_source_list,
                    is_target_list=is_target_list
                )

                cov_x_source_batch = cov_v(
                    diff=cov_diff_source,
                    num=1
                )
                cov_x_source += cov_x_source_batch.clone().detach()
                num_source += n_source

                if n_target > 0:
                    cov_x_target_batch = cov_v(
                        diff=cov_diff_target,
                        num=1
                    )
                    cov_x_target += cov_x_target_batch.clone().detach()
                    num_target += n_target

            else:
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    recon_batch, _ = self.model(data)

                    # Baseline-style score used for logging / distribution fitting
                    score_2d = self.loss_fn(recon_batch, data)

                    n_loss = len(score_2d)
                    score = self.loss_reduction_1d(score=score_2d)

                    recon_loss = self.loss_reduction(score=score, n_loss=n_loss)
                    recon_loss_source = self.loss_reduction(
                        score=score[is_source_list],
                        n_loss=n_source
                    )
                    if n_target > 0:
                        recon_loss_target = self.loss_reduction(
                            score=score[is_target_list],
                            n_loss=n_target
                        )
                    else:
                        recon_loss_target = 0

                    # Conformer regularization
                    x1 = augment(data, self.args.frames, self.args.n_mels)
                    x2 = augment(data, self.args.frames, self.args.n_mels)

                    recon1, z1 = self.model(x1)
                    _, z2 = self.model(x2)

                    latent_loss = F.mse_loss(z1, z2)
                    train_obj = recon_loss + 0.05 * latent_loss

                self.loss = recon_loss

                self.scaler.scale(train_obj).backward()
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    5.0
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
            recon_loss_source = self.loss_reduction(score=score[is_source_list], n_loss=n_source)
            if n_target > 0:
                recon_loss_target = self.loss_reduction(score=score[is_target_list], n_loss=n_target)
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
                device=self.device
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
                score_distr_file_path=self.mahala_score_distr_file_path
            )

        # validation test
        val_loss = 0
        with torch.no_grad():
            self.model.eval()
            for _, batch in enumerate(self.valid_loader):
                data = batch[0]
                data = data.to(self.device).float()

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
                    val_loss / len(self.valid_loader)
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
                    fmt="%s"
                )

            csv_to_figdata(
                file_path=self.log_path,
                column_heading_list=self.column_heading_list,
                ylabel="loss",
                fig_count=len(self.column_heading_list),
                cut_first_epoch=True
            )

            self.fit_anomaly_score_distribution(
                y_pred=y_pred,
                score_distr_file_path=self.mse_score_distr_file_path
            )

        # save model
        torch.save(self.model.state_dict(), self.model_path)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
        }, self.checkpoint_path)


# alias
Conformerv1 = Conformer