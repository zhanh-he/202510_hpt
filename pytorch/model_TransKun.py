from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as AF

from utilities import note_events_to_velocity_roll


def _as_optional_str(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


class TransKunPretrained(nn.Module):
    """Adapter that exposes pretrained TransKun as a velocity-roll predictor."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.frames_per_second = int(cfg.feature.frames_per_second)
        self.classes_num = int(cfg.feature.classes_num)
        self.begin_note = int(cfg.feature.begin_note)
        self.velocity_scale = float(cfg.feature.velocity_scale)
        self.input_sample_rate = int(cfg.feature.sample_rate)

        self.segment_seconds = float(cfg.feature.segment_seconds)
        self.segment_frames = int(round(self.frames_per_second * self.segment_seconds)) + 1

        self.transkun_repo_root = self._resolve_repo_root()
        self.transkun_conf = self._resolve_conf_path()

        self.transkun_model = None
        self.transkun_sample_rate = None

    def _resolve_repo_root(self) -> Optional[Path]:
        configured = _as_optional_str(getattr(self.cfg.model, "transkun_repo_root", None))
        if configured:
            path = Path(configured).expanduser().resolve()
            return path if path.exists() else None

        default_root = (Path(__file__).resolve().parents[2] / "Transkun").resolve()
        return default_root if default_root.exists() else None

    def _resolve_conf_path(self) -> Optional[Path]:
        configured = _as_optional_str(getattr(self.cfg.model, "transkun_conf", None))
        if configured:
            path = Path(configured).expanduser().resolve()
            return path

        if self.transkun_repo_root:
            candidate = self.transkun_repo_root / "transkun" / "pretrained" / "2.0.conf"
            if candidate.exists():
                return candidate
        return None

    def _ensure_transkun_importable(self) -> None:
        if self.transkun_repo_root is not None:
            root_str = str(self.transkun_repo_root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)

    def load_external_checkpoint(self, checkpoint_path: str, device: torch.device) -> None:
        self._ensure_transkun_importable()

        if self.transkun_conf is None or not self.transkun_conf.exists():
            raise FileNotFoundError(
                "TransKun config not found. Set `model.transkun_conf=/path/to/2.0.conf`."
            )

        with open(self.transkun_conf, "r", encoding="utf-8") as f:
            conf_payload = json.load(f)
        model_payload = conf_payload["Model"]
        module_name = model_payload.get("module", "transkun.ModelTransformer")
        config_class_name = model_payload.get("configClassName", "Config")

        try:
            model_module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = getattr(exc, "name", "")
            raise ModuleNotFoundError(
                "Failed to import TransKun runtime dependency "
                f"'{missing}'. Install TransKun requirements first."
            ) from exc
        transkun_cls = getattr(model_module, "TransKun")
        config_cls = getattr(model_module, config_class_name)
        transkun_conf = config_cls()
        for key, value in model_payload.get("config", {}).items():
            setattr(transkun_conf, key, value)

        model = transkun_cls(conf=transkun_conf).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "best_state_dict" in checkpoint:
            state_dict = checkpoint["best_state_dict"]
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model.eval()

        self.transkun_model = model
        self.transkun_sample_rate = int(model.fs)

    def _render_roll(self, notes, duration_sec: float, expected_frames: int) -> np.ndarray:
        note_events = [
            {
                "midi_note": int(n.pitch),
                "onset_time": float(n.start),
                "offset_time": float(n.end),
                "velocity": int(np.clip(int(n.velocity), 0, int(self.velocity_scale) - 1)),
            }
            for n in notes
            if int(n.pitch) >= self.begin_note and int(n.pitch) < self.begin_note + self.classes_num
        ]

        frames_num = int(round(duration_sec * self.frames_per_second)) + 1
        roll = note_events_to_velocity_roll(
            note_events=note_events,
            frames_num=frames_num,
            num_keys=self.classes_num,
            frames_per_second=self.frames_per_second,
            begin_note=self.begin_note,
            velocity_scale=self.velocity_scale,
        )

        if roll.shape[0] < expected_frames:
            pad = np.zeros((expected_frames - roll.shape[0], roll.shape[1]), dtype=roll.dtype)
            roll = np.concatenate([roll, pad], axis=0)
        elif roll.shape[0] > expected_frames:
            roll = roll[:expected_frames]
        return roll.astype(np.float32, copy=False)

    def forward(self, waveform: torch.Tensor, *_unused_inputs):
        if self.transkun_model is None:
            raise RuntimeError(
                "TransKun model is not loaded. `load_external_checkpoint` must run before inference."
            )

        batch_size, samples_num = waveform.shape
        duration_sec = float(samples_num) / float(self.input_sample_rate)
        expected_frames = max(1, int(round(duration_sec * self.frames_per_second)) + 1)

        rolls = []
        for idx in range(batch_size):
            audio = waveform[idx].detach().to(dtype=torch.float32, device=waveform.device)
            audio = audio.unsqueeze(0)  # [1, time]

            if self.input_sample_rate != self.transkun_sample_rate:
                audio = AF.resample(
                    audio,
                    orig_freq=self.input_sample_rate,
                    new_freq=self.transkun_sample_rate,
                )
            audio = audio.transpose(0, 1)  # [time, channel]

            notes = self.transkun_model.transcribe(
                audio,
                stepInSecond=None,
                segmentSizeInSecond=None,
                discardSecondHalf=False,
            )
            roll = self._render_roll(notes, duration_sec=duration_sec, expected_frames=expected_frames)
            rolls.append(torch.from_numpy(roll))

        velocity_output = torch.stack(rolls, dim=0).to(device=waveform.device, dtype=torch.float32)
        return {"velocity_output": velocity_output}
