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


def _fit_roll_shape(roll: Optional[torch.Tensor], target_t: int, target_p: int) -> Optional[torch.Tensor]:
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


class _ConformerLikeLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, conv_kernel: int = 7):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)

        self.dwconv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=d_model,
        )
        self.pwconv = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop2 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.drop3 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.drop1(attn_out))

        y = x.transpose(1, 2)
        y = self.dwconv(y)
        y = self.pwconv(y)
        y = y.transpose(1, 2)
        x = self.norm2(x + self.drop2(y))

        y = self.ffn(x)
        x = self.norm3(x + self.drop3(y))
        return x


class NoteEventEditor(nn.Module):
    """
    Note-level velocity correction:
      - tokenize note onsets from onset roll
      - predict note-wise delta
      - write delta back onto onset locations
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
        arch: str = "conformer",
        alpha: float = 0.2,
        max_frames: int = 4096,
        use_cond_feats: Optional[List[str]] = None,
        onset_threshold: float = 0.5,
    ):
        super().__init__()
        if arch not in {"transformer", "conformer"}:
            raise ValueError(f"Unsupported arch: {arch}")

        self.arch = arch
        self.alpha = float(alpha)
        self.onset_threshold = float(onset_threshold)
        self.use_cond_feats = list(use_cond_feats or [])

        self.pitch_emb = nn.Embedding(88, d_model)
        self.time_emb = nn.Embedding(max_frames, d_model)
        self.feat_proj = nn.Linear(1 + len(self.use_cond_feats), d_model)

        if arch == "transformer":
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        else:
            self.encoder = nn.ModuleList(
                [_ConformerLikeLayer(d_model, n_heads, dropout=dropout) for _ in range(n_layers)]
            )

        self.head = nn.Linear(d_model, 1)

    def _resolve_cond_feats(
        self,
        onset: torch.Tensor,
        cond_map: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for feat_name in self.use_cond_feats:
            if feat_name in cond_map and cond_map[feat_name] is not None:
                out[feat_name] = cond_map[feat_name]
                continue
            if feat_name == "exframe" and cond_map.get("frame") is not None:
                out[feat_name] = cond_map["frame"] * (1.0 - onset)
                continue
            if feat_name == "frame" and cond_map.get("exframe") is not None:
                out[feat_name] = torch.clamp(cond_map["exframe"] + onset, 0.0, 1.0)
                continue
            out[feat_name] = torch.zeros_like(onset)
        return out

    def forward(
        self,
        base_velocity: torch.Tensor,
        onset_roll: torch.Tensor,
        cond_map: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        cond_map = dict(cond_map or {})
        vel0 = torch.clamp(base_velocity, 0.0, 1.0)
        bsz, t_steps, pitches = vel0.shape

        onset = _fit_roll_shape(onset_roll, t_steps, pitches)
        onset_mask = onset > self.onset_threshold

        for k, v in list(cond_map.items()):
            cond_map[k] = _fit_roll_shape(v, t_steps, pitches)
        cond_feats = self._resolve_cond_feats(onset, cond_map)

        token_embs: List[torch.Tensor] = []
        token_coords: List[torch.Tensor] = []
        lengths: List[int] = []

        for b in range(bsz):
            coords = onset_mask[b].nonzero(as_tuple=False)  # (N, 2): [t, p]
            n = int(coords.shape[0])
            lengths.append(n)
            token_coords.append(coords)

            if n == 0:
                token_embs.append(torch.zeros((0, self.pitch_emb.embedding_dim), device=vel0.device))
                continue

            t_idx = coords[:, 0]
            p_idx = coords[:, 1].clamp_max(self.pitch_emb.num_embeddings - 1)
            v0 = vel0[b, t_idx, p_idx].float()

            cont_feats = [v0]
            for feat_name in self.use_cond_feats:
                cont_feats.append(cond_feats[feat_name][b, t_idx, p_idx].float())
            cont = torch.stack(cont_feats, dim=-1)

            token_emb = (
                self.pitch_emb(p_idx)
                + self.time_emb(t_idx.clamp_max(self.time_emb.num_embeddings - 1))
                + self.feat_proj(cont)
            )
            token_embs.append(token_emb)

        nmax = max(lengths) if lengths else 0
        vel_corr = vel0.clone()
        delta_roll = torch.zeros_like(vel0)

        if nmax == 0:
            return {
                "vel_corr": vel_corr,
                "delta_roll": delta_roll,
                "delta_token": None,
                "note_count": torch.tensor(lengths, device=vel0.device),
            }

        d_model = self.pitch_emb.embedding_dim
        x = torch.zeros((bsz, nmax, d_model), device=vel0.device)
        key_padding_mask = torch.ones((bsz, nmax), device=vel0.device, dtype=torch.bool)

        for b in range(bsz):
            n = lengths[b]
            if n == 0:
                continue
            x[b, :n] = token_embs[b]
            key_padding_mask[b, :n] = False

        valid_rows = [b for b, n in enumerate(lengths) if n > 0]
        if len(valid_rows) == bsz:
            if self.arch == "transformer":
                h = self.encoder(x, src_key_padding_mask=key_padding_mask)
            else:
                h = x
                for layer in self.encoder:
                    h = layer(h, key_padding_mask=key_padding_mask)
        else:
            h = torch.zeros_like(x)
            idx = torch.tensor(valid_rows, device=vel0.device, dtype=torch.long)
            xv = x[idx]
            kv = key_padding_mask[idx]
            if self.arch == "transformer":
                hv = self.encoder(xv, src_key_padding_mask=kv)
            else:
                hv = xv
                for layer in self.encoder:
                    hv = layer(hv, key_padding_mask=kv)
            h[idx] = hv

        delta_token = self.head(h).squeeze(-1)  # (B, Nmax)

        for b in range(bsz):
            n = lengths[b]
            if n == 0:
                continue
            coords = token_coords[b]
            dv = self.alpha * torch.tanh(delta_token[b, :n])
            t_idx = coords[:, 0]
            p_idx = coords[:, 1]
            vel_corr[b, t_idx, p_idx] = torch.clamp(vel0[b, t_idx, p_idx] + dv, 0.0, 1.0)
            delta_roll[b, t_idx, p_idx] = dv

        return {
            "vel_corr": vel_corr,
            "delta_roll": delta_roll,
            "delta_token": delta_token,
            "note_count": torch.tensor(lengths, device=vel0.device),
        }


class ScoreNoteEditor_HPT(nn.Module):
    """
    HPT base + note-level editor.
    Expected forward signature:
      model(audio, onset_roll, cond_roll_optional)
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

        score_cfg = getattr(cfg.model, "score_hpt", None)
        input3_name = _clean_name(getattr(cfg.model, "input3", None))
        self.input3_name = input3_name if input3_name in {"frame", "exframe"} else None

        use_cond_feats = _to_str_list(getattr(score_cfg, "use_cond_feats", None) if score_cfg else None)
        if not use_cond_feats and self.input3_name is not None:
            use_cond_feats = [self.input3_name]
        use_cond_feats = [k for k in use_cond_feats if k in {"frame", "exframe"}]

        self.note_editor = NoteEventEditor(
            d_model=int(getattr(score_cfg, "d_model", 128) if score_cfg else 128),
            n_layers=int(getattr(score_cfg, "n_layers", 2) if score_cfg else 2),
            n_heads=int(getattr(score_cfg, "n_heads", 4) if score_cfg else 4),
            dropout=float(getattr(score_cfg, "dropout", 0.1) if score_cfg else 0.1),
            arch=str(getattr(score_cfg, "arch", "conformer") if score_cfg else "conformer"),
            alpha=float(getattr(score_cfg, "alpha", 0.2) if score_cfg else 0.2),
            max_frames=int(getattr(score_cfg, "max_frames", 4096) if score_cfg else 4096),
            use_cond_feats=use_cond_feats,
            onset_threshold=float(getattr(score_cfg, "onset_threshold", 0.5) if score_cfg else 0.5),
        )

    def _build_cond_map(
        self,
        onset_roll: torch.Tensor,
        input3: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cond_map: Dict[str, torch.Tensor] = {}
        if input3 is None:
            return cond_map

        input3 = _fit_roll_shape(input3, onset_roll.size(1), onset_roll.size(2))
        if self.input3_name is not None:
            cond_map[self.input3_name] = input3
        else:
            # Backward-safe fallback when input3 role is not explicitly set.
            cond_map["frame"] = input3

        if "frame" in cond_map and "exframe" not in cond_map:
            cond_map["exframe"] = cond_map["frame"] * (1.0 - onset_roll)
        return cond_map

    def forward(self, input1, input2=None, input3=None):
        if input2 is None:
            raise ValueError(
                "ScoreNoteEditor_HPT requires onset conditioning as second input. "
                "Set model.input2='onset' in config."
            )

        x = self.feature_extractor(input1)
        x = x.unsqueeze(3)
        x = self.bn0(x)
        x = x.transpose(1, 3)
        base_velocity = self.velocity_model(x)

        onset_roll = _fit_roll_shape(input2, base_velocity.size(1), base_velocity.size(2))
        cond_map = self._build_cond_map(onset_roll, input3)
        edited = self.note_editor(base_velocity, onset_roll, cond_map)
        vel_corr = edited["vel_corr"]

        return {
            "velocity_output": vel_corr,
            "vel_corr": vel_corr,
            "velocity_logits": _safe_logit(vel_corr),
            "velocity_base": base_velocity,
            "delta": edited["delta_roll"],
            "delta_token": edited["delta_token"],
            "note_count": edited["note_count"],
        }


# Alias for shorter config name if needed.
ScoreHPT = ScoreNoteEditor_HPT
