import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.criterion.mahala import cov_v, loss_function_mahala, calc_inv_cov
from tools.plot_loss_curve import csv_to_figdata


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
        block_size,
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

        # Required by the DCASE Mahalanobis utilities
        self.register_buffer("cov_source", torch.eye(block_size))
        self.register_buffer("cov_target", torch.eye(block_size))

    def forward(self, x, alpha=None):
        z = self.encoder(x)
        recon = self.decoder(z)

        # Inference / baseline-style calls should return only 2 outputs.
        if alpha is None:
            return recon, z

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

    similarity_matrix = similarity_matrix / temperature

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
# FRAE_DANN MODEL WRAPPER
# Keeps the DCASE2023T2AE evaluation pipeline unchanged.
# =========================================================
class FRAE_DANN(DCASE2023T2AE):

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
            lr=self.args.learning_rate,
            weight_decay=1e-4
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(self.args.epochs, 1),
            eta_min=self.args.learning_rate * 0.05
        )

        self.use_amp = bool(torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.best_hmean = 0.0
        self.early_stop_patience = 20

    # -----------------------------------------------------
    # INIT MODEL
    # -----------------------------------------------------
    def init_model(self):
        self.block_size = self.data.height

        return FRAE_DANNNet(
            input_dim=self.data.input_dim,
            block_size=self.block_size,
            n_mels=self.args.n_mels,
            latent_dim=32
        )

    # -----------------------------------------------------
    # TRAIN
    # Same covariance / validation / score-distribution flow
    # as the DCASE baseline, with extra DANN + contrastive loss
    # during the normal training phase only.
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

        alpha = (epoch / max(self.args.epochs, 1)) ** 2

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
                batch_size = data.shape[0]
                x_spec = data.view(
                    batch_size,
                    self.args.frames,
                    self.args.n_mels
                )

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

                domain_labels = torch.tensor(
                    [1 if is_t else 0 for is_t in is_target_list],
                    device=self.device
                ).long()

                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    recon_batch, _, dom_logits = self.model(data, alpha)

                    _, z1, _ = self.model(x1, alpha)
                    _, z2, _ = self.model(x2, alpha)

                    # Baseline-style reconstruction score for logging/distribution fitting
                    score_2d = self.loss_fn(
                        recon_batch,
                        data
                    )

                    n_loss = len(score_2d)
                    score = self.loss_reduction_1d(score=score_2d)

                    recon_loss = self.loss_reduction(
                        score=score,
                        n_loss=n_loss
                    )
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

                    # Extra FRAE+DANN objective
                    w = F.softmax(
                        self.model.freq_weights,
                        dim=0
                    )
                    w = w.unsqueeze(0).unsqueeze(0)

                    recon_2d = recon_batch.view(
                        batch_size,
                        self.args.frames,
                        self.args.n_mels
                    )
                    weighted_recon_loss = (
                        ((x_spec - recon_2d) ** 2) * w
                    ).mean()

                    cont_loss = contrastive_loss(z1, z2)
                    dom_loss = F.cross_entropy(dom_logits, domain_labels)

                    optim_loss = (
                        weighted_recon_loss
                        + 0.3 * cont_loss
                        + 0.1 * dom_loss
                    )

                self.loss = recon_loss

                self.scaler.scale(optim_loss).backward()
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

                # calculate y_pred for fitting anomaly score distribution
                y_pred.append(self.loss.item())

                if batch_idx % self.args.log_interval == 0:
                    print(
                        'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100. * batch_idx / len(train_loader),
                            self.loss.item()
                        )
                    )
                continue

            # Covariance stage bookkeeping
            n_loss = len(score_2d)
            score = self.loss_reduction_1d(score=score_2d)

            recon_loss = self.loss_reduction(
                score=score,
                n_loss=n_loss
            )
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

            self.loss = recon_loss

            train_loss += float(self.loss)
            train_recon_loss += float(recon_loss)
            train_recon_loss_source += float(recon_loss_source)
            train_recon_loss_target += float(recon_loss_target)

            # calculate y_pred for fitting anomaly score distribution
            y_pred.append(self.loss.item())

        if is_calc_cov:
            # save cov_x
            cov_x_source /= num_source - 1
            if num_target == 0:
                cov_x_target = cov_x_source.clone().detach()
            else:
                cov_x_target /= num_target - 1

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
            for batch_idx, batch in enumerate(self.valid_loader):
                data = batch[0]
                data = data.to(self.device).float()

                recon_batch, _ = self.model(data)
                score = self.loss_fn(
                    recon_batch,
                    data
                )
                loss = score.mean()

                val_loss += float(loss)

                # calculate y_pred for fitting anomaly score distribution
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
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': self.loss
            },
            self.checkpoint_path
        )