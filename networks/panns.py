
import os
import sys
import csv
import pickle
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as TAT
from pathlib import Path
from tqdm import tqdm
from sklearn import metrics
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA

from networks.base_model import BaseModel

PANNS_CHECKPOINT = os.environ.get(
    "PANNS_CHECKPOINT",
    "models/pretrained/Cnn14_mAP=0.431.pth",
)

_SR     = 32_000   
_N_FFT  = 1024
_HOP    = 320      
_N_MELS = 64
_FMIN   = 50.0
_FMAX   = 14_000.0

PCA_DIMS = 128   
REG_COV  = 1e-4   


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor, pool=(2, 2)) -> torch.Tensor:
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        return F.avg_pool2d(x, pool) + F.max_pool2d(x, pool)


class Cnn14(nn.Module):

    def __init__(self):
        super().__init__()
        self.bn0         = nn.BatchNorm2d(64)
        self.conv_block1 = _ConvBlock(1,    64)
        self.conv_block2 = _ConvBlock(64,   128)
        self.conv_block3 = _ConvBlock(128,  256)
        self.conv_block4 = _ConvBlock(256,  512)
        self.conv_block5 = _ConvBlock(512,  1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1         = nn.Linear(2048, 2048)
        self.fc_audioset = nn.Linear(2048, 527)
        self._reset_params()

    def _reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn0(x.transpose(1, 3)).transpose(1, 3)

        x = F.dropout(self.conv_block1(x),              p=0.2, training=self.training)
        x = F.dropout(self.conv_block2(x),              p=0.2, training=self.training)
        x = F.dropout(self.conv_block3(x),              p=0.2, training=self.training)
        x = F.dropout(self.conv_block4(x),              p=0.2, training=self.training)
        x = F.dropout(self.conv_block5(x),              p=0.2, training=self.training)
        x = F.dropout(self.conv_block6(x, pool=(1, 1)), p=0.2, training=self.training)
        x = torch.mean(x, dim=3)                            
        x = torch.max(x, dim=2)[0] + torch.mean(x, dim=2)  

        x = F.relu_(self.fc1(F.dropout(x, p=0.5, training=self.training)))
        return F.dropout(x, p=0.5, training=self.training)  


def _load_audio(path: str):
    
    try:
        wav, sr = torchaudio.load(path)
        return wav, sr
    except Exception:
        pass
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr


def _load_panns_weights(model: Cnn14, ckpt_path: str):
    if not os.path.exists(ckpt_path):
        print(
            f"[WARN] CNN14 checkpoint not found at '{ckpt_path}'.\n"
            "       Using random initialisation — AUC will be much worse.\n"
            "       Set the PANNS_CHECKPOINT env var or place the file there."
        )
        return
    print(f"Loading CNN14 pretrained weights from: {ckpt_path}")
    raw = torch.load(ckpt_path, map_location="cpu")
    state = raw.get("model", raw)
    own   = set(model.state_dict().keys())
    match = {k: v for k, v in state.items() if k in own}
    missing = own - set(match.keys())
    if missing:
        print(f"  [WARN] {len(missing)} keys missing in checkpoint "
              f"(first 5: {sorted(missing)[:5]})")
    model.load_state_dict(match, strict=False)
    print(f"  Loaded {len(match)}/{len(own)} parameter tensors.")



class PANNS(BaseModel):

    def __init__(self, args, train, test):
        super().__init__(args=args, train=train, test=test)

        dataset_name = args.dataset[:11].lower()          
        machine_type = args.dataset[11:]                  
        data_type    = "dev_data" if args.dev else "eval_data"
        self._audio_root = Path(os.path.abspath(
            os.path.join(os.getcwd(), args.dataset_directory,
                         dataset_name, data_type, "raw", machine_type)
        ))
        print(f"Audio root: {self._audio_root}")

        stem = (f"{args.model}_{args.dataset}{self.model_name_suffix}"
                f"{self.eval_suffix}_seed{args.seed}")
        self._gaussian_path = self.model_dir / f"gaussian_{stem}.pickle"

        self._mel_tf = TAT.MelSpectrogram(
            sample_rate=_SR, n_fft=_N_FFT, hop_length=_HOP,
            n_mels=_N_MELS, f_min=_FMIN, f_max=_FMAX, power=2.0,
        )
        self._to_db      = TAT.AmplitudeToDB(stype="power", top_db=80.0)
        self._resamplers : dict = {}   
        self._gparams = None


    def init_model(self) -> nn.Module:
        Path("models/pretrained").mkdir(parents=True, exist_ok=True)
        cnn14 = Cnn14()
        _load_panns_weights(cnn14, PANNS_CHECKPOINT)
        return cnn14

    def get_log_header(self) -> str:
        return "epoch,note"


    def train(self, epoch: int):
        if epoch <= self.epoch:
            return                   

        if epoch != 1:
            if epoch == self.args.epochs + 1:
                print("Post-training hook: no-op (work done in epoch 1).")
            return

        print("\n======== CNN14 FEATURE EXTRACTION + GAUSSIAN FIT ========")
        self.model.eval()

        src_files, tgt_files = self._scan_train_files()
        src_embs = self._embed_files(src_files, "source")
        tgt_embs = self._embed_files(tgt_files, "target")

        gparams = self._fit_gaussians(src_embs, tgt_embs)

        with open(self._gaussian_path, "wb") as f:
            pickle.dump(gparams, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Gaussian params saved ->{self._gaussian_path}")

        scores  = [self._score(e, gparams, domain="source") for e in src_embs]
        scores += [self._score(e, gparams, domain="target") for e in tgt_embs]
        self.fit_anomaly_score_distribution(y_pred=scores)
        print(f"Score distribution saved ->{self.score_distr_file_path}")
        torch.save(self.model.state_dict(), self.model_path)
        torch.save({
            "epoch":                1,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": {},
            "loss":                 float(np.mean(scores)),
        }, self.checkpoint_path)
        print(f"Model weights saved ->{self.model_path}")

        with open(self.log_path, "a") as f:
            np.savetxt(f, ["1,feature_extraction_and_gaussian_fit_done"], fmt="%s")

        self.epoch = 1


    def test(self):
        map_loc = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model.load_state_dict(
                torch.load(self.model_path, map_location=map_loc)
            )
            print(f"CNN14 weights loaded from: {self.model_path}")
        except (RuntimeError, FileNotFoundError) as e:
            print(f"[WARN] Could not load {self.model_path}: {e}\n"
                  "       Falling back to PANNS pretrained checkpoint.")
            _load_panns_weights(self.model, PANNS_CHECKPOINT)
        self.model.eval()

        if not self._gaussian_path.exists():
            raise FileNotFoundError(
                f"Gaussian params not found: {self._gaussian_path}\n"
                "You need to run training on Kaggle first, then copy these 3 files:\n"
                f"  {self.model_path}\n"
                f"  {self.score_distr_file_path}\n"
                f"  {self._gaussian_path}"
            )
        with open(self._gaussian_path, "rb") as f:
            gparams = pickle.load(f)
        print(f"Gaussian params loaded from: {self._gaussian_path}")

        if not self.score_distr_file_path.exists():
            raise FileNotFoundError(
                f"Score distribution not found: {self.score_distr_file_path}\n"
                "Copy  score_distr_*.pickle  from the Kaggle training run."
            )
        decision_threshold = self.calc_decision_threshold()
        mode = self.data.mode

        csv_lines             = []
        performance           = []
        performance_over_all  = []

        for idx, test_loader in enumerate(self.test_loader):
            section_name = f"section_{self.data.section_id_list[idx]}"
            out_dir = self.result_dir if self.args.dev else self.eval_data_result_dir

            anm_csv = out_dir / (
                f"anomaly_score_{self.args.dataset}_{section_name}_test"
                f"_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv")
            dec_csv = out_dir / (
                f"decision_result_{self.args.dataset}_{section_name}_test"
                f"_seed{self.args.seed}{self.model_name_suffix}{self.eval_suffix}.csv")

            y_pred, y_true, domain_list = [], [], []
            anm_list, dec_list = [], []

            print(f"\n===== TEST {section_name} =====")
            for batch in test_loader:
                # All rows in the batch belong to the same audio file
                basename = batch[3][0]
                label    = batch[1][0].item()
                domain   = "target" if "target" in basename else "source"

                audio_path = self._audio_root / "test" / basename
                if not audio_path.exists():
                    # Evaluation sets may use 'test_rename'
                    audio_path = self._audio_root / "test_rename" / basename
                if not audio_path.exists():
                    raise FileNotFoundError(
                        f"Audio file not found: {audio_path}\n"
                        f"  audio_root = {self._audio_root}"
                    )

                emb   = self._embed(audio_path)
                score = self._score(emb, gparams, domain=domain)

                y_pred.append(score)
                y_true.append(label)
                domain_list.append(domain)
                anm_list.append([basename, score])
                dec_list.append([basename, int(score > decision_threshold)])

            _save_csv(anm_csv, anm_list)
            _save_csv(dec_csv, dec_list)
            print(f"anomaly score  ->{anm_csv}")
            print(f"decision result ->{dec_csv}")

            if mode:
                self._report_metrics(
                    y_true, y_pred, domain_list, decision_threshold,
                    section_name, csv_lines, performance, performance_over_all,
                )

        if mode:
            amean = np.mean(np.array(performance, dtype=float), axis=0)
            hmean = scipy.stats.hmean(
                np.maximum(np.array(performance, dtype=float),
                           sys.float_info.epsilon), axis=0)
            csv_lines.append(["arithmetic mean"] + list(amean))
            csv_lines.append(["harmonic mean"]   + list(hmean))
            out_dir = self.result_dir if self.args.dev else self.eval_data_result_dir
            rpath = out_dir / (
                f"result_{self.args.dataset}_test_seed{self.args.seed}"
                f"{self.model_name_suffix}{self.eval_suffix}_roc.csv")
            _save_csv(rpath, csv_lines)
            print(f"Results ->{rpath}")


    def _scan_train_files(self):
        train_dir = self._audio_root / "train"
        if not train_dir.exists():
            raise FileNotFoundError(f"Training audio directory not found: {train_dir}")
        sections = [f"section_{sid}" for sid in self.data.section_id_list]
        src, tgt = [], []
        for wav in sorted(train_dir.glob("*.wav")):
            if not any(s in wav.name for s in sections):
                continue
            (tgt if "target" in wav.name else src).append(wav)
        print(f"Training files found: {len(src)} source, {len(tgt)} target")
        return src, tgt

    def _embed_files(self, files: list, desc: str = "") -> np.ndarray:
        if not files:
            return np.empty((0, 2048), dtype=np.float32)
        embs = [self._embed(f) for f in tqdm(files, desc=f"CNN14 [{desc}]")]
        return np.array(embs, dtype=np.float32)

    def _embed(self, audio_path) -> np.ndarray:

        wav, sr = _load_audio(str(audio_path))   
        if wav.shape[0] > 1:                      
            wav = wav.mean(0, keepdim=True)
        if sr != _SR:
            if sr not in self._resamplers:
                self._resamplers[sr] = TAT.Resample(sr, _SR)
            wav = self._resamplers[sr](wav)

        mel = self._to_db(self._mel_tf(wav))    
        x   = mel.transpose(1, 2).unsqueeze(0)  
        x   = x.to(self.device)

        with torch.no_grad():
            emb = self.model(x)[0]             
        return emb.cpu().numpy()

    def _fit_gaussians(self, src_embs: np.ndarray, tgt_embs: np.ndarray) -> dict:
       
        n_src = src_embs.shape[0]
        n_tgt = tgt_embs.shape[0]
        print(f"Fitting Gaussians: {n_src} source, {n_tgt} target embeddings")

        all_embs = (np.vstack([src_embs, tgt_embs]) if n_tgt > 0 else src_embs)
        n_comp   = min(PCA_DIMS, all_embs.shape[0] - 1, all_embs.shape[1])
        pca      = PCA(n_components=n_comp, random_state=42)
        all_pca  = pca.fit_transform(all_embs)
        print(f"  PCA({n_comp}): cumulative explained variance = "
              f"{pca.explained_variance_ratio_.sum():.3f}")

        src_pca = all_pca[:n_src]
        tgt_pca = all_pca[n_src:] if n_tgt > 0 else None

        src_mean = src_pca.mean(0)
        lw = LedoitWolf()
        lw.fit(src_pca)
        cov  = lw.covariance_ + REG_COV * np.eye(n_comp)
        prec = np.linalg.inv(cov).astype(np.float32)
        print(f"  Source: mean_norm={np.linalg.norm(src_mean):.2f}, "
              f"cov_trace={np.trace(lw.covariance_):.2f}")

        tgt_mean = tgt_pca.mean(0).astype(np.float32) if tgt_pca is not None \
                   else src_mean.copy()
        print(f"  Target mean: dist_to_src_mean="
              f"{np.linalg.norm(tgt_mean - src_mean):.3f}")

        return {
            "pca":      pca,
            "src_mean": src_mean.astype(np.float32),
            "prec":     prec,           # shared precision matrix
            "tgt_mean": tgt_mean,
        }

    def _score(self, emb: np.ndarray, gparams: dict, domain: str = "source") -> float:
        e    = gparams["pca"].transform(emb.reshape(1, -1))[0].astype(np.float32)
        p    = gparams["prec"]
        mean = gparams["tgt_mean"] if domain == "target" else gparams["src_mean"]
        d    = e - mean
        return float(d @ p @ d)

    def _report_metrics(self, y_true, y_pred, domain_list, threshold,
                        section_name, csv_lines, performance, performance_over_all):
        idx_sa = [i for i in range(len(y_true))
                  if domain_list[i] == "source" or y_true[i] == 1]
        idx_ta = [i for i in range(len(y_true))
                  if domain_list[i] == "target" or y_true[i] == 1]
        idx_s  = [i for i in range(len(y_true)) if domain_list[i] == "source"]
        idx_t  = [i for i in range(len(y_true)) if domain_list[i] == "target"]

        def sel(idx): return [y_true[i] for i in idx], [y_pred[i] for i in idx]

        yt_sa, yp_sa = sel(idx_sa)
        yt_ta, yp_ta = sel(idx_ta)
        yt_s,  yp_s  = sel(idx_s)
        yt_t,  yp_t  = sel(idx_t)

        auc_s   = metrics.roc_auc_score(yt_sa, yp_sa)
        p_auc   = metrics.roc_auc_score(y_true, y_pred, max_fpr=self.args.max_fpr)
        p_auc_s = metrics.roc_auc_score(yt_s,  yp_s,   max_fpr=self.args.max_fpr)

        dec_s = [1 if v > threshold else 0 for v in yp_s]
        tn_s, fp_s, fn_s, tp_s = metrics.confusion_matrix(yt_s, dec_s).ravel()
        prec_s   = tp_s / max(tp_s + fp_s, sys.float_info.epsilon)
        recall_s = tp_s / max(tp_s + fn_s, sys.float_info.epsilon)
        f1_s     = 2*prec_s*recall_s / max(prec_s+recall_s, sys.float_info.epsilon)

        print(f"AUC (source)      : {auc_s:.4f}")
        print(f"pAUC              : {p_auc:.4f}")
        print(f"pAUC (source)     : {p_auc_s:.4f}")
        print(f"precision (source): {prec_s:.4f}  recall: {recall_s:.4f}  "
              f"F1: {f1_s:.4f}")

        if len(yt_t) > 0:
            auc_t   = metrics.roc_auc_score(yt_ta, yp_ta)
            p_auc_t = metrics.roc_auc_score(yt_t, yp_t, max_fpr=self.args.max_fpr)
            dec_t   = [1 if v > threshold else 0 for v in yp_t]
            tn_t, fp_t, fn_t, tp_t = metrics.confusion_matrix(yt_t, dec_t).ravel()
            prec_t   = tp_t / max(tp_t + fp_t, sys.float_info.epsilon)
            recall_t = tp_t / max(tp_t + fn_t, sys.float_info.epsilon)
            f1_t     = 2*prec_t*recall_t / max(prec_t+recall_t, sys.float_info.epsilon)

            print(f"AUC (target)      : {auc_t:.4f}")
            print(f"pAUC (target)     : {p_auc_t:.4f}")
            print(f"precision (target): {prec_t:.4f}  recall: {recall_t:.4f}  "
                  f"F1: {f1_t:.4f}")

            if not csv_lines:
                csv_lines.append(self.result_column_dict["source_target"])
            csv_lines.append([section_name.split("_", 1)[1],
                               auc_s, auc_t, p_auc, p_auc_s, p_auc_t,
                               prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
            performance.append([auc_s, auc_t, p_auc, p_auc_s, p_auc_t,
                                 prec_s, prec_t, recall_s, recall_t, f1_s, f1_t])
        else:
            if not csv_lines:
                csv_lines.append(self.result_column_dict["single_domain"])
            csv_lines.append([section_name.split("_", 1)[1],
                               auc_s, p_auc, prec_s, recall_s, f1_s])
            performance.append([auc_s, p_auc, prec_s, recall_s, f1_s])

        performance_over_all.append(performance[-1])


def _save_csv(path, data):
    with open(path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(data)