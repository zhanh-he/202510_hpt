import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import librosa
import numpy as np
from hydra import compose, initialize

from inference import VeloTranscription
from utilities import (
    TargetProcessor,
    create_folder,
    get_model_name,
    read_midi,
    write_events_to_midi,
)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run score-informed velocity estimation and rewrite MIDI velocities.",
    )
    parser.add_argument("--audio-path", required=True, help="Input audio file (wav/mp3).")
    parser.add_argument("--midi-path", required=True, help="Score-aligned MIDI file.")
    parser.add_argument("--output-path", required=True, help="Path for the velocity-updated MIDI.")
    parser.add_argument(
        "--velocity-method",
        default="max_frame",
        choices=["max_frame", "onset_only"],
        help="Strategy for picking a predicted velocity per note.",
    )
    parser.add_argument(
        "--midi-format",
        default="maestro",
        choices=["maestro", "hpt", "smd", "maps"],
        help="Layout hint for parsing the MIDI file with mido.",
    )
    parser.add_argument(
        "--config-path",
        default="./",
        help="Hydra config path (relative to this script).",
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Hydra config name.",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        help="Optional Hydra override strings, e.g. exp.ckpt_iteration=100000 model.name=FiLMUNetPretrained",
    )
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help="Override checkpoint path. Defaults to cfg.exp.workspace/checkpoints/... or cfg.model.pretrained_checkpoint.",
    )
    return parser.parse_args(argv)


def _resolve_checkpoint(cfg, explicit_path: Optional[str]) -> Path:
    if explicit_path:
        ckpt = Path(explicit_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint override {ckpt} does not exist.")
        return ckpt

    if cfg.model.name == "FiLMUNetPretrained":
        ckpt = Path(cfg.model.pretrained_checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"Pretrained Kim et al. checkpoint missing at {ckpt}. "
                "Provide --checkpoint-path or update cfg.model.pretrained_checkpoint."
            )
        return ckpt

    if cfg.exp.ckpt_file:
        ckpt = Path(cfg.exp.ckpt_file)
        if ckpt.exists():
            return ckpt
        raise FileNotFoundError(f"cfg.exp.ckpt_file points to {ckpt}, but the file does not exist.")

    if not cfg.exp.ckpt_iteration:
        raise ValueError(
            "cfg.exp.ckpt_iteration is empty. "
            "Set exp.ckpt_iteration or supply --checkpoint-path when launching the script."
        )

    model_name = get_model_name(cfg)
    ckpt = (
        Path(cfg.exp.workspace)
        / "checkpoints"
        / model_name
        / f"{cfg.exp.ckpt_iteration}_iterations.pth"
    )
    if not ckpt.exists():
        raise FileNotFoundError(f"Could not locate checkpoint at {ckpt}")
    return ckpt


def _load_audio(audio_path: Path, sample_rate: int) -> np.ndarray:
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    return audio.astype(np.float32)


def _prepare_aux_rolls(
    cfg,
    midi_events_time: np.ndarray,
    midi_events: Sequence[str],
    duration_sec: float,
) -> Tuple[dict, List[dict], List[dict]]:
    processor = TargetProcessor(segment_seconds=duration_sec, cfg=cfg)
    target_dict, note_events, pedal_events = processor.process(
        start_time=0.0,
        midi_events_time=midi_events_time,
        midi_events=midi_events,
        extend_pedal=True,
    )

    # Prepare auxiliary rolls required by FiLM/HPT variants.
    if "exframe_roll" not in target_dict:
        target_dict["exframe_roll"] = target_dict["frame_roll"] * (1 - target_dict["onset_roll"])

    return target_dict, note_events, pedal_events


def _original_score_events(
    cfg,
    midi_events_time: np.ndarray,
    midi_events: Sequence[str],
    duration_sec: float,
) -> Tuple[List[dict], List[dict]]:
    """Re-run the processor without pedal extension to preserve the exact MIDI timing."""
    processor = TargetProcessor(segment_seconds=duration_sec, cfg=cfg)
    _, note_events, pedal_events = processor.process(
        start_time=0.0,
        midi_events_time=midi_events_time,
        midi_events=midi_events,
        extend_pedal=False,
    )
    return note_events, pedal_events


def _select_condition_roll(target_dict: dict, condition_name: Optional[str]) -> Optional[np.ndarray]:
    if condition_name is None:
        return None
    name = str(condition_name).strip()
    if not name or name.lower() in {"none", "null"}:
        return None
    key = f"{name}_roll"
    if key not in target_dict:
        raise KeyError(f"Condition '{name}' requested, but '{key}' not found in target_dict.")
    return target_dict[key]


def _pick_velocity(
    note_events: List[dict],
    velocity_roll: np.ndarray,
    cfg,
    strategy: str,
) -> None:
    fps = cfg.feature.frames_per_second
    begin_note = cfg.feature.begin_note
    velocity_scale = cfg.feature.velocity_scale
    num_frames, num_keys = velocity_roll.shape

    for event in note_events:
        pitch_idx = event["midi_note"] - begin_note
        if pitch_idx < 0 or pitch_idx >= num_keys:
            continue

        onset_frame = int(round(event["onset_time"] * fps))
        offset_frame = int(round(event["offset_time"] * fps))
        onset_frame = np.clip(onset_frame, 0, max(0, num_frames - 1))
        offset_frame = max(onset_frame + 1, offset_frame)
        offset_frame = min(offset_frame, num_frames)

        note_curve = velocity_roll[onset_frame:offset_frame, pitch_idx]
        if note_curve.size == 0:
            picked = 0.0
        elif strategy == "max_frame":
            picked = float(np.max(note_curve))
        elif strategy == "onset_only":
            picked = float(note_curve[0])
        else:
            raise ValueError(f"Unknown velocity pick strategy: {strategy}")

        scaled_velocity = np.clip(picked * velocity_scale, 0, velocity_scale - 1)
        event["velocity"] = int(round(scaled_velocity))


def _check_duration_alignment(audio_len: float, midi_events_time: np.ndarray) -> None:
    midi_len = float(midi_events_time[-1]) if midi_events_time.size else 0.0
    diff = abs(audio_len - midi_len)
    if diff > 0.05:
        print(
            f"[warn] Audio duration ({audio_len:.2f}s) and MIDI duration ({midi_len:.2f}s) "
            f"differ by {diff:.2f}s. Proceeding with audio duration as reference.",
        )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    audio_path = Path(args.audio_path)
    midi_path = Path(args.midi_path)
    output_path = Path(args.output_path)

    with initialize(config_path=args.config_path, job_name="score_velocity", version_base=None):
        cfg = compose(config_name=args.config_name, overrides=args.overrides)

    checkpoint_path = _resolve_checkpoint(cfg, args.checkpoint_path)
    transcriber = VeloTranscription(checkpoint_path=checkpoint_path, cfg=cfg)

    audio = _load_audio(audio_path, sample_rate=cfg.feature.sample_rate)
    audio_duration = audio.shape[0] / cfg.feature.sample_rate

    midi_dict = read_midi(str(midi_path), dataset=args.midi_format)
    midi_events_time = np.asarray(midi_dict["midi_event_time"], dtype=float)
    midi_events = [
        msg.decode() if isinstance(msg, (bytes, bytearray)) else str(msg)
        for msg in midi_dict["midi_event"]
    ]

    _check_duration_alignment(audio_duration, midi_events_time)

    target_dict, _, _ = _prepare_aux_rolls(cfg, midi_events_time, midi_events, audio_duration)
    input2 = _select_condition_roll(target_dict, cfg.model.input2)
    input3 = _select_condition_roll(target_dict, cfg.model.input3)

    transcribed = transcriber.transcribe(audio, input2=input2, input3=input3)
    velocity_roll = transcribed["output_dict"]["velocity_output"]

    note_events, pedal_events = _original_score_events(cfg, midi_events_time, midi_events, audio_duration)
    _pick_velocity(note_events, velocity_roll, cfg, strategy=args.velocity_method)

    create_folder(str(output_path.parent))
    write_events_to_midi(0.0, note_events, pedal_events, str(output_path))
    print(f"[done] Wrote MIDI with {len(note_events)} notes to {output_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
