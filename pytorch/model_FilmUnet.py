import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


def _resolve_film_src(cfg) -> Path:
    root = Path(cfg.model.film_repo_root).expanduser().resolve()
    return root / "src"


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_kim_runtime(src_dir: Path) -> Tuple[object, object]:
    src_str = str(src_dir)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)
    importlib.invalidate_caches()

    audio_transforms = importlib.import_module("audio_transforms")
    torchlibrosa_mod = types.ModuleType("torchlibrosa")
    torchlibrosa_stft_mod = types.ModuleType("torchlibrosa.stft")
    torchlibrosa_stft_mod.Spectrogram = audio_transforms.Spectrogram
    torchlibrosa_stft_mod.LogmelFilterBank = audio_transforms.LogmelFilterBank
    sys.modules["torchlibrosa"] = torchlibrosa_mod
    sys.modules["torchlibrosa.stft"] = torchlibrosa_stft_mod

    kim_config = _load_module_from_file("kim_film_config_runtime", src_dir / "config.py")

    prev_config = sys.modules.get("config")
    sys.modules["config"] = kim_config
    try:
        kim_model = _load_module_from_file("kim_film_model_runtime", src_dir / "model.py")
    finally:
        if prev_config is not None:
            sys.modules["config"] = prev_config
        else:
            sys.modules.pop("config", None)

    return kim_config, kim_model


class FiLMUNetPretrained(nn.Module):
    """Thin wrapper around the original FiLM U-Net with pretrained weights."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        kim_src_dir = _resolve_film_src(cfg)
        self.kim_config, kim_model = _load_kim_runtime(kim_src_dir)
        self._maybe_override_conditioning(cfg)

        self.model = kim_model.ScoreInformedMidiVelocityEstimator(
            frames_per_second=self.kim_config.frames_per_second,
            classes_num=self.kim_config.classes_num,
        )
        checkpoint_path = self._resolve_checkpoint_path(cfg)
        state_dict = self._prepare_state_dict(self._load_state_dict(checkpoint_path))
        self.model.load_state_dict(state_dict, strict=True)

    def _maybe_override_conditioning(self, cfg) -> None:
        """Sync conditioning flags with whatever extra inputs Hydra enables."""
        wants_condition = cfg.model.input2 is not None
        self.kim_config.condition_check = wants_condition
        if wants_condition and cfg.model.input2:
            self.kim_config.condition_type = cfg.model.input2
        else:
            self.kim_config.condition_type = "onset"

    @staticmethod
    def _resolve_checkpoint_path(cfg) -> Path:
        return Path(cfg.model.pretrained_checkpoint)

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            elif "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
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
