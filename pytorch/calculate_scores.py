import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from hydra import compose, initialize
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

from inference import VeloTranscription
from utilities import (
    TargetProcessor,
    create_folder,
    get_model_name,
    int16_to_float32,
    resolve_hdf5_dir,
    traverse_folder,
)


def note_level_l1_per_window(
    output_segment: Dict[str, np.ndarray],
    target_segment: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, int]:
    """Kim et al. note-level L1 error for a single segment."""
    error_rows: List[np.ndarray] = []
    num_notes = 0
    frames = min(
        output_segment["velocity_output"].shape[0],
        target_segment["velocity_roll"].shape[0],
    )

    for nth_frame in range(frames):
        gt_onset_frame = target_segment["onset_roll"][nth_frame]
        if np.count_nonzero(gt_onset_frame) == 0:
            continue

        pred_frame = output_segment["velocity_output"][nth_frame]
        gt_frame = target_segment["velocity_roll"][nth_frame]
        pred_onset = np.multiply(pred_frame, gt_onset_frame) * 128.0
        gt_onset = np.multiply(gt_frame, gt_onset_frame)
        note_error = np.abs(pred_onset - gt_onset)

        num_notes += int(np.count_nonzero(gt_onset_frame))
        error_rows.append(note_error[np.newaxis, :])

    if error_rows:
        segment_error = np.concatenate(error_rows, axis=0)
    else:
        segment_error = np.empty((0, 88), dtype=float)
    return segment_error, num_notes


def classification_error(score: np.ndarray, estimation: np.ndarray) -> Tuple[float, float, float]:
    """Binary frame classification metrics on flattened frame/key pairs."""
    score_bin = score.copy()
    estimation_bin = estimation.copy()

    score_bin[score_bin > 0] = 1
    estimation_bin[estimation_bin > 0.0001] = 1
    estimation_bin[estimation_bin <= 0.0001] = 0

    flat_score = score_bin.flatten()
    flat_est = estimation_bin.flatten()
    f1 = f1_score(flat_score, flat_est, average="macro", zero_division=0)
    precision = precision_score(flat_score, flat_est, average="macro", zero_division=0)
    recall = recall_score(flat_score, flat_est, average="macro", zero_division=0)
    return f1, precision, recall


def num_simultaneous_notes(note_profile: Dict[str, np.ndarray], score: np.ndarray) -> int:
    start, end = note_profile["duration"]
    sim_note_count = 0
    for key in range(score.shape[0]):
        sim_note = score[key][start:end]
        if np.sum(sim_note) > 0:
            sim_note_count += 1
    return sim_note_count


def pedal_check(note_profile: Dict[str, np.ndarray], pedal: np.ndarray) -> bool:
    start, end = note_profile["duration"]
    return bool(np.sum(pedal[start:end]) > 0)


def get_midi_sound_profile(midi_vel_roll: np.ndarray) -> List[Dict[str, np.ndarray]]:
    sound_profile: List[Dict[str, np.ndarray]] = []
    for pitch, key in enumerate(midi_vel_roll):
        iszero = np.concatenate(([0], np.equal(key, 0).astype(np.int8), [0]))
        absdiff = np.abs(np.diff(iszero))
        ranges = np.where(absdiff == 1)[0]
        if ranges.size <= 2:
            continue
        temp = np.delete(ranges, [0, -1])
        sound_durations = temp.reshape(-1, 2)
        for duration in sound_durations:
            vel = midi_vel_roll[pitch, duration[0]]
            sound_profile.append({"pitch": pitch, "velocity": vel, "duration": duration})
    return sound_profile


def gt_to_note_list(
    output_dict_list: Sequence[Dict[str, np.ndarray]],
    target_list: Sequence[Dict[str, np.ndarray]],
    ) -> Tuple[float, float, List[Dict[str, np.ndarray]], float, float, float]:
    """Kim et al. per-note error profile."""
    score_rows: List[np.ndarray] = []
    pedal_list: List[np.ndarray] = []
    estimation_rows: List[np.ndarray] = []

    for target_segment, output_segment in zip(target_list, output_dict_list):
        frames = target_segment["velocity_roll"].shape[0]
        for nth_frame in range(frames):
            gt_velframe = target_segment["velocity_roll"][nth_frame]
            gt_pedal = target_segment["pedal_frame_roll"][nth_frame]
            output_vel_frame = output_segment["velocity_output"][nth_frame]

            score_rows.append(gt_velframe[np.newaxis, :])
            pedal_list.append(np.asarray(gt_pedal).reshape(1))
            estimation_rows.append(output_vel_frame[np.newaxis, :])

    if not score_rows:
        return (0.0, 0.0, [], 0.0, 0.0, 0.0)

    score = np.concatenate(score_rows, axis=0)
    estimation = np.concatenate(estimation_rows, axis=0)
    pedal = np.concatenate(pedal_list, axis=0)

    f1, precision, recall = classification_error(score.copy(), estimation.copy())

    score = np.transpose(score)
    estimation = np.transpose(estimation)

    score_sound_profile = get_midi_sound_profile(score)
    error_profile: List[Dict[str, np.ndarray]] = []
    accum_error: List[float] = []

    for note_profile in score_sound_profile:
        start, end = note_profile["duration"]
        vel_est = estimation[note_profile["pitch"]][start:end].copy()
        vel_est[vel_est <= 0.0001] = 0

        classification_check = bool(np.sum(vel_est) > 0)
        max_estimation = float(np.max(vel_est) * 128.0) if vel_est.size else 0.0
        notelevel_error = abs(max_estimation - float(note_profile["velocity"]))
        sim_note_count = num_simultaneous_notes(note_profile, score)
        pedal_onoff = pedal_check(note_profile, pedal)

        error_profile.append(
            {
                "pitch": note_profile["pitch"],
                "duration": (int(start), int(end)),
                "note_error": notelevel_error,
                "ground_truth": float(note_profile["velocity"]),
                "estimation": max_estimation,
                "pedal_check": pedal_onoff,
                "simultaneous_notes": sim_note_count,
                "classification_check": classification_check,
            } # type: ignore
        )
        accum_error.append(notelevel_error)

    frame_max_error = float(np.mean(accum_error)) if accum_error else 0.0
    std_max_error = float(np.std(accum_error)) if accum_error else 0.0

    return frame_max_error, std_max_error, error_profile, f1, precision, recall


def eval_from_list(
    output_dict_list: Sequence[Dict[str, np.ndarray]],
    target_dict_list: Sequence[Dict[str, np.ndarray]],
) -> Tuple[float, float]:
    """Kim et al. onset-only evaluation."""
    score_error_rows: List[np.ndarray] = []
    num_note = 0
    for output_dict_segmentseconds, target_dict_segmentseconds in zip(
        output_dict_list, target_dict_list
    ):
        segment_error, num_onset = note_level_l1_per_window(
            output_dict_segmentseconds, target_dict_segmentseconds
        )
        if segment_error.size:
            score_error_rows.append(segment_error)
        num_note += num_onset

    if num_note == 0:
        return 0.0, 0.0

    score_error = np.concatenate(score_error_rows, axis=0) if score_error_rows else np.empty((0, 88))
    mean_error = float(np.sum(score_error) / num_note)
    non_zero = score_error[score_error != 0]
    std_error = float(non_zero.std()) if non_zero.size else 0.0
    return mean_error, std_error


class KimStyleEvaluator:
    """Run HPT inference and compute Kim et al. evaluation metrics."""

    CSV_FIELDS = [
        "test_h5",
        "frame_max_error",
        "frame_max_std",
        "f1_score",
        "precision",
        "recall",
        "onset_masked_error",
        "onset_masked_std",
    ]

    def __init__(self, cfg):
        if cfg.model.type != "velo":
            raise ValueError("Kim-style evaluation is defined for velocity models only.")
        if not cfg.exp.ckpt_iteration:
            raise ValueError("cfg.exp.ckpt_iteration must be set for evaluation.")

        self.cfg = cfg
        self.model_name = get_model_name(cfg)
        self.ckpt_iteration = str(cfg.exp.ckpt_iteration)
        checkpoint_path = Path(cfg.exp.workspace) / "checkpoints" / self.model_name / f"{self.ckpt_iteration}_iterations.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.transcriptor = VeloTranscription(str(checkpoint_path), cfg)

        hdf5_dir = resolve_hdf5_dir(cfg.exp.workspace, cfg.dataset.test_set, cfg.feature.sample_rate)
        _, self.hdf5_paths = traverse_folder(hdf5_dir)

        self.results_dir = (
            Path(cfg.exp.workspace)
            / "kim_eval"
            / cfg.dataset.test_set
            / self.model_name
            / f"{self.ckpt_iteration}_iterations"
        )
        self.error_dict_dir = self.results_dir / "error_dict"
        create_folder(str(self.error_dict_dir))

    def _prepare_inputs(self, target_dict: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        target_dict["exframe_roll"] = target_dict["frame_roll"] * (1 - target_dict["onset_roll"])

        input2 = target_dict.get(f"{self.cfg.model.input2}_roll") if self.cfg.model.input2 else None
        input3 = target_dict.get(f"{self.cfg.model.input3}_roll") if self.cfg.model.input3 else None
        return input2, input3

    def _process_file(self, hdf5_path: str) -> Optional[Dict[str, np.ndarray]]:
        with h5py.File(hdf5_path, "r") as hf:
            if hf.attrs["split"].decode() != "test":
                return None
            audio = int16_to_float32(hf["waveform"][:])
            midi_events = [e.decode() for e in hf["midi_event"][:]]
            midi_events_time = hf["midi_event_time"][:]

        segment_seconds = len(audio) / self.cfg.feature.sample_rate
        target_processor = TargetProcessor(segment_seconds=segment_seconds, cfg=self.cfg)
        target_dict, _, _ = target_processor.process(
            start_time=0, midi_events_time=midi_events_time, midi_events=midi_events, extend_pedal=True
        )

        input2, input3 = self._prepare_inputs(target_dict)
        transcribed = self.transcriptor.transcribe(audio, input2, input3, midi_path=None)
        output_dict = transcribed["output_dict"]

        align_len = min(output_dict["velocity_output"].shape[0], target_dict["velocity_roll"].shape[0])

        output_entry = {
            "velocity_output": output_dict["velocity_output"][:align_len],
        }
        target_entry = {
            "velocity_roll": target_dict["velocity_roll"][:align_len],
            "frame_roll": target_dict["frame_roll"][:align_len],
            "onset_roll": target_dict["onset_roll"][:align_len],
            "pedal_frame_roll": target_dict["pedal_frame_roll"][:align_len],
        }

        output_dict_list = [output_entry]
        target_dict_list = [target_entry]

        (
            frame_max_error,
            frame_max_std,
            error_profile,
            f1,
            precision,
            recall,
        ) = gt_to_note_list(output_dict_list, target_dict_list)

        onset_masked_error, onset_masked_std = eval_from_list(output_dict_list, target_dict_list)

        return {
            "audio_name": Path(hdf5_path).name,
            "frame_max_error": frame_max_error,
            "frame_max_std": frame_max_std,
            "f1_score": f1,
            "precision": precision,
            "recall": recall,
            "onset_masked_error": onset_masked_error,
            "onset_masked_std": onset_masked_std,
            "error_profile": np.array(error_profile, dtype=object),
        } # type: ignore

    def run(self) -> Dict[str, List[float]]:
        csv_path = self.results_dir / f"{self.model_name}_{self.cfg.dataset.test_set}_kim.csv"
        create_folder(str(self.results_dir))

        aggregated: Dict[str, List[float]] = {field: [] for field in self.CSV_FIELDS if field != "test_h5"}
        processed = 0

        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.CSV_FIELDS)

            progress = tqdm(sorted(self.hdf5_paths), desc="Kim Eval", unit="file", ncols=80)
            for hdf5_path in progress:
                metrics = self._process_file(hdf5_path)
                if not metrics:
                    continue

                error_profile = metrics.pop("error_profile")
                audio_name = metrics["audio_name"]
                error_dict_path = self.error_dict_dir / f"{Path(audio_name).stem}_aligned.npy"
                np.save(error_dict_path, error_profile, allow_pickle=True)

                row = [audio_name] + [metrics[field] for field in self.CSV_FIELDS[1:]]
                writer.writerow(row)

                for field in aggregated.keys():
                    aggregated[field].append(float(metrics[field]))

                processed += 1
                avg_frame_err = np.mean(aggregated["frame_max_error"])
                progress.set_postfix({"frame_err": f"{avg_frame_err:.2f}"}, refresh=False)

        if not processed:
            return {}

        return aggregated


def main() -> None:
    initialize(config_path="./", job_name="kim_eval", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])

    print("=" * 80)
    print("Evaluation Mode : Kim et al.")
    print(f"Model Name      : {get_model_name(cfg)}")
    print(f"Test Set        : {cfg.dataset.test_set}")
    print("=" * 80)

    evaluator = KimStyleEvaluator(cfg)
    stats_dict = evaluator.run()

    if not stats_dict:
        print("No test files processed.")
        return

    print("\n===== Kim-style Average Metrics =====")
    for key, values in stats_dict.items():
        mean_value = float(np.mean(values))
        print(f"{key}: {mean_value:.4f}")


if __name__ == "__main__":
    main()
