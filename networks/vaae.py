import sys
import csv
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from sklearn import metrics

from networks.dcase2023t2_ae.dcase2023t2_ae import DCASE2023T2AE
from networks.dcase2023t2_ae.network import AENet
from networks.criterion.mahala import loss_function_mahala, calc_inv_cov

ALPHA = 0.5


class VAAE(DCASE2023T2AE):

    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)
        self.optimizer = optim.Adam(
            [{"params": self.model.parameters()}],
            lr=self.args.learning_rate
        )

    def init_model(self):
        self.block_size = self.data.height
        return AENet(input_dim=self.data.input_dim, block_size=self.block_size)

    def get_log_header(self):
        self.column_heading_list = [
            ["loss"],
            ["val_loss"],
            ["recon_loss"],
            ["recon_loss_source", "recon_loss_target"],
        ]
        return "loss,val_loss,recon_loss,recon_loss_source,recon_loss_target"

    def loss_fn(self, recon_x, x):
        return F.mse_loss(recon_x, x.view(recon_x.shape), reduction="none")

    def loss_reduction_1d(self, score):
        return torch.mean(score, dim=1)

    def loss_reduction(self, score, n_loss):
        return torch.sum(score) / n_loss

    def _variance_aware_mahala_score(self, recon, x, inv_cov):
        per_frame_scores, num = loss_function_mahala(
            recon_x=recon,
            x=x,
            block_size=self.block_size,
            cov=inv_cov,
            use_precision=True,
            reduction=False
        )
        per_frame = self.loss_reduction_1d(per_frame_scores)
        if per_frame.shape[0] < 2:
            return per_frame.mean()
        return per_frame.mean() + ALPHA * per_frame.std()

    def calc_valid_mahala_score(self, data, y_pred, inv_cov_source, inv_cov_target):
        data = data.to(self.device).float()
        recon_data, _ = self.model(data)

        score_source = self._variance_aware_mahala_score(
            recon_data, data, inv_cov_source
        )
        score_target = self._variance_aware_mahala_score(
            recon_data, data, inv_cov_target
        )
        y_pred.append(min(score_source.item(), score_target.item()))
        return y_pred

    def eval(
        self,
        test_loader,
        y_pred,
        anomaly_score_list,
        decision_result_list,
        domain_list,
        y_true,
        decision_threshold,
        mode,
        inv_cov_source,
        inv_cov_target,
    ):
        
        for j, batch in enumerate(test_loader):
            data = batch[0].to(self.device).float()
            y_true.append(batch[1][0].item())
            basename = batch[3][0]

            recon_data, _ = self.model(data)

            if self.args.score == "MAHALA":
                score_source = self._variance_aware_mahala_score(
                    recon_data, data, inv_cov_source
                )
                score_target = self._variance_aware_mahala_score(
                    recon_data, data, inv_cov_target
                )
                score = min(score_source.item(), score_target.item())
            else:
                per_frame = F.mse_loss(
                    recon_data,
                    data.view(recon_data.shape),
                    reduction="none"
                ).mean(dim=1)
                if per_frame.shape[0] < 2:
                    score = per_frame.mean().item()
                else:
                    score = (per_frame.mean() + ALPHA * per_frame.std()).item()

            y_pred.append(score)
            anomaly_score_list.append([basename, score])
            decision_result_list.append(
                [basename, 1 if score > decision_threshold else 0]
            )

            if mode:
                domain_list.append("target" if "target" in basename else "source")

        return y_pred, anomaly_score_list, decision_result_list, domain_list