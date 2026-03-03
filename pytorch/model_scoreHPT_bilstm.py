from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extractor import get_feature_extractor_and_bins
from model_HPT import OriginalModelHPT2020, init_bn


def _safe_logit(p: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    p = torch.clamp(p, eps, 1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def _clean_name(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return None
    return text


def _to_str_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if _clean_name(v)]
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"", "none", "null"}:
            return []
        return [tok.strip() for tok in text.split(",") if tok.strip()]
    return [str(value).strip()]


def _fit_roll_shape(
    roll: Optional[torch.Tensor],
    target_t: int,
    target_p: int,
) -> Optional[torch.Tensor]:
    if roll is None:
        return None
    if roll.dim() != 3:
        raise ValueError(f"Expected conditioning roll shape (B,T,P), but got {tuple(roll.shape)}")

    t = roll
    if t.size(1) > target_t:
        t = t[:, :target_t]
    elif t.size(1) < target_t:
        t = F.pad(t, (0, 0, 0, target_t - t.size(1)))

    if t.size(2) > target_p:
        t = t[:, :, :target_p]
    elif t.size(2) < target_p:
        t = F.pad(t, (0, target_p - t.size(2), 0, 0))

    return t


class ScoreBiLSTM_HPT(nn.Module):
    """
    Score-informed BiLSTM + HPT.
    Uses HPT as base acoustic estimator and BiLSTM for score-conditioned correction.
    """

    def __init__(self, cfg):
        super().__init__()
        sample_rate = cfg.feature.sample_rate
        fft_size = cfg.feature.fft_size
        frames_per_second = cfg.feature.frames_per_second
        audio_feature = cfg.feature.audio_feature
        classes_num = cfg.feature.classes_num
        momentum = 0.01

        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(
            audio_feature,
            sample_rate,
            fft_size,
            frames_per_second,
        )
        self.bn0 = nn.BatchNorm2d(self.FRE, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, self.FRE, momentum)
        init_bn(self.bn0)

        bilstm_cfg = getattr(cfg.model, "score_hpt_bilstm", None)

        input2_name = _clean_name(getattr(cfg.model, "input2", None))
        input3_name = _clean_name(getattr(cfg.model, "input3", None))
        default_cond = [k for k in [input2_name, input3_name] if k]
        cond_keys = _to_str_list(getattr(bilstm_cfg, "cond_keys", None) if bilstm_cfg else None)
        if not cond_keys:
            cond_keys = default_cond
        self.cond_keys = cond_keys

        in_feats = _to_str_list(getattr(bilstm_cfg, "in_feats", None) if bilstm_cfg else None)
        if not in_feats:
            in_feats = ["vel_logits"]
        self.in_feats = in_feats

        self.mode = str(getattr(bilstm_cfg, "mode", "residual") if bilstm_cfg else "residual")
        if self.mode not in {"direct", "residual"}:
            raise ValueError(f"Unsupported score_hpt_bilstm.mode: {self.mode}")

        self.alpha = float(getattr(bilstm_cfg, "alpha", 0.2) if bilstm_cfg else 0.2)
        hidden = int(getattr(bilstm_cfg, "hidden", 256) if bilstm_cfg else 256)
        num_layers = int(getattr(bilstm_cfg, "num_layers", 2) if bilstm_cfg else 2)
        dropout = float(getattr(bilstm_cfg, "dropout", 0.1) if bilstm_cfg else 0.1)

        n_ch = len(self.in_feats) + len(self.cond_keys)
        if n_ch <= 0:
            raise ValueError("ScoreBiLSTM_HPT requires at least one feature channel.")

        self.rnn = nn.LSTM(
            input_size=88 * n_ch,
            hidden_size=hidden,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden * 2, 88)

        self.input2_name = input2_name
        self.input3_name = input3_name

    def _build_cond_map(
        self,
        input2: Optional[torch.Tensor],
        input3: Optional[torch.Tensor],
        target_t: int,
        target_p: int,
    ) -> Dict[str, torch.Tensor]:
        cond_map: Dict[str, torch.Tensor] = {}
        if input2 is not None and self.input2_name is not None:
            cond_map[self.input2_name] = _fit_roll_shape(input2, target_t, target_p)
        if input3 is not None and self.input3_name is not None:
            cond_map[self.input3_name] = _fit_roll_shape(input3, target_t, target_p)
        return cond_map

    def _resolve_feat(self, name: str, vel: torch.Tensor, vel_logits: torch.Tensor) -> torch.Tensor:
        if name == "vel":
            return vel
        if name == "vel_logits":
            return vel_logits
        raise KeyError(
            f"Unsupported feature '{name}' in score_hpt_bilstm.in_feats. "
            "Use one of: ['vel', 'vel_logits']"
        )

    def forward(self, input1, input2=None, input3=None):
        x = self.feature_extractor(input1)
        x = x.unsqueeze(3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        base_velocity = self.velocity_model(x)
        base_velocity = torch.clamp(base_velocity, 0.0, 1.0)
        vel_logits = _safe_logit(base_velocity)

        bsz, t_steps, pitches = base_velocity.shape
        cond_map = self._build_cond_map(input2, input3, t_steps, pitches)

        feat_list = [self._resolve_feat(name, base_velocity, vel_logits) for name in self.in_feats]
        for key in self.cond_keys:
            if key not in cond_map:
                raise ValueError(
                    f"Missing conditioning roll '{key}'. "
                    f"Available from inputs: {sorted(cond_map.keys())}. "
                    "Set cfg.model.input2/input3 accordingly."
                )
            feat_list.append(cond_map[key])

        x_cat = torch.cat(feat_list, dim=-1)
        h, _ = self.rnn(x_cat)
        out = self.fc(h)

        if self.mode == "direct":
            vel_corr = torch.sigmoid(out)
            delta_roll = None
            corr_logits = out
        else:
            delta_roll = self.alpha * torch.tanh(out)
            vel_corr = torch.clamp(base_velocity + delta_roll, 0.0, 1.0)
            corr_logits = _safe_logit(vel_corr)

        return {
            "velocity_output": vel_corr,
            "vel_corr": vel_corr,
            "velocity_logits": corr_logits,
            "velocity_base": base_velocity,
            "delta": delta_roll,
        }


ScoreHPT_BiLSTM = ScoreBiLSTM_HPT
