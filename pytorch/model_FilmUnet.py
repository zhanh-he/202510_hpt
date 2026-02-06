import logging
import sys
import types
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

# Inject a lightweight torchlibrosa shim so the original FiLM code can import it.
_AUDIO_SRC = Path("/media/datadisk/home/22828187/zhanh/202601_midisemi/kim_ismir2024/src")
if not _AUDIO_SRC.exists():
    raise FileNotFoundError(f"Missing audio transforms source at {_AUDIO_SRC}")
sys.path.insert(0, str(_AUDIO_SRC))
import audio_transforms  # type: ignore  # noqa: E402

torchlibrosa_mod = types.ModuleType("torchlibrosa")
torchlibrosa_stft_mod = types.ModuleType("torchlibrosa.stft")
torchlibrosa_stft_mod.Spectrogram = audio_transforms.Spectrogram
torchlibrosa_stft_mod.LogmelFilterBank = audio_transforms.LogmelFilterBank
sys.modules["torchlibrosa"] = torchlibrosa_mod
sys.modules["torchlibrosa.stft"] = torchlibrosa_stft_mod

# Reuse the untouched FiLM codebase.
_KIM_SRC = Path("/media/datadisk/home/22828187/zhanh/backup/Kim_ismir2024_code/src")
if not _KIM_SRC.exists():
    raise FileNotFoundError(f"Expected FiLM U-Net sources at {_KIM_SRC}")
sys.path.insert(0, str(_KIM_SRC))

import config as kim_config  # type: ignore  # noqa: E402
from model import ScoreInformedMidiVelocityEstimator  # type: ignore  # noqa: E402


class FiLMUNetPretrained(nn.Module):
    """Thin wrapper around the original FiLM U-Net with pretrained weights."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._maybe_override_conditioning(cfg)

        self.model = ScoreInformedMidiVelocityEstimator(
            frames_per_second=kim_config.frames_per_second,
            classes_num=kim_config.classes_num,
        )
        checkpoint_path = self._resolve_checkpoint_path(cfg)
        state_dict = self._prepare_state_dict(self._load_state_dict(checkpoint_path))
        self.model.load_state_dict(state_dict, strict=True)

    @staticmethod
    def _maybe_override_conditioning(cfg) -> None:
        """Sync conditioning flags with whatever extra inputs Hydra enables."""
        wants_condition = cfg.model.input2 is not None
        kim_config.condition_check = wants_condition
        if wants_condition and cfg.model.input2:
            kim_config.condition_type = cfg.model.input2
        else:
            kim_config.condition_type = "onset"

    @staticmethod
    def _resolve_checkpoint_path(cfg) -> Path:
        ckpt_path = Path(cfg.model.pretrained_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Pretrained FiLM U-Net checkpoint not found at {ckpt_path}"
            )
        return ckpt_path

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise RuntimeError(f"Unsupported checkpoint format in {checkpoint_path}")
        keys = list(checkpoint.keys())
        if keys and all(k.startswith("module.") for k in keys):
            checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
        keys = list(checkpoint.keys())
        if keys and all(k.startswith("model.") for k in keys):
            checkpoint = {k.replace("model.", "", 1): v for k, v in checkpoint.items()}
        return checkpoint

    def _prepare_state_dict(self, state_dict: dict) -> dict:
        """Preserve deterministic frontend weights while enforcing strict loading."""
        prepared = {
            k: v
            for k, v in state_dict.items()
            if not (
                k.startswith("spectrogram_extractor")
                or k.startswith("logmel_extractor")
            )
        }
        local_state = self.model.state_dict()
        for key, tensor in local_state.items():
            if key.startswith("spectrogram_extractor") or key.startswith("logmel_extractor"):
                prepared[key] = tensor.detach().clone()
        return prepared

    def forward(self, waveform, score: Optional[torch.Tensor] = None):
        return self.model(waveform, score)
