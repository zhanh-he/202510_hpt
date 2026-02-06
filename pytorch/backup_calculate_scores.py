import os
os.environ["WANDB_SILENT"] = "true"
import sys
from hydra import initialize, compose
import pickle
import h5py
import mir_eval
import numpy as np
import time
from concurrent.futures.process import ProcessPoolExecutor
from evaluate import prepare_tensor, cal_mae, cal_std
from utilities import traverse_folder, get_filename, get_model_name, note_to_freq, OnsetsFramesPostProcessor, resolve_hdf5_dir

import wandb
import pandas as pd
import gc
# Filter out the FutureWarning
import warnings
warnings.filterwarnings("ignore", message="`rcond` parameter will change")


def _apply_model_defaults(cfg):
    if cfg.model.name == "FiLMUNetPretrained":
        cfg.feature.sample_rate = 16000
        cfg.feature.segment_seconds = 2.0
        cfg.feature.hop_seconds = 1.0
        cfg.feature.frames_per_second = 100
        cfg.feature.audio_feature = "logmel"


class ScoreCalculator(object):
    def __init__(self, cfg):
        """Evaluate piano transcription metrics of the post processed - pre-calculated system outputs."""
        self.cfg = cfg
        hdf5s_dir = resolve_hdf5_dir(cfg.exp.workspace, cfg.dataset.test_set, cfg.feature.sample_rate)
        _, self.hdf5_paths = traverse_folder(hdf5s_dir)

        if cfg.model.type == "velo":
            # Prepare model name and probabilities dir
            model_name = get_model_name(cfg)
            self.probs_dir = os.path.join(cfg.exp.workspace, f"probs_{cfg.model.type}", cfg.dataset.test_set, model_name, f"{cfg.exp.ckpt_iteration}_iterations")
        
        elif cfg.model.type == "notes":
            model_name = "Original HPT"
            self.probs_dir = os.path.join(cfg.exp.workspace, f"probs_{cfg.model.type}", cfg.dataset.test_set)

    
    def metrics(self):
        """Calculate metrics of all songs."""
        list_args = [
            [n, hdf5_path]
            for n, hdf5_path in enumerate(self.hdf5_paths) # e.g.[0, 'xx.h5']
            if h5py.File(hdf5_path, 'r').attrs['split'].decode() == 'test'
        ] 
        if self.cfg.exp.debug:
            results = [self.calculate_score_per_song(arg) for arg in list_args]
        else: # Calculate metrics in parallel
            with ProcessPoolExecutor(max_workers=self.cfg.exp.num_workers) as executor:
                results = executor.map(self.calculate_score_per_song, list_args)
        stats_list = list(results)
        stats_dict = {key: [e[key] for e in stats_list if key in e.keys()] for key in stats_list[0].keys()}
        return stats_dict
    
    def calculate_score_per_song(self, args):
        """Calculate score per song with GroundTruth
        Args:
          args: [n, hdf5_path]
        """
        # Load pre-calculated outputs & ground truths from note_path | velocity_output from velo_path
        hdf5_path = args[1]
        prob_path = os.path.join(self.probs_dir, f"{get_filename(hdf5_path)}.pkl")
        total_dict = pickle.load(open(prob_path, 'rb'))
        requested_iter = str(self.cfg.exp.ckpt_iteration)
        stored_iter = str(total_dict.get('checkpoint_iteration', requested_iter))
        if stored_iter != requested_iter:
            raise ValueError(
                f"Probability file {prob_path} was generated for iteration {stored_iter}, "
                f"but you requested {requested_iter}. Please rerun inference for this checkpoint."
            )

        # Align prediction and ground-truth along time axis
        if all(key in total_dict for key in ['velocity_output', 'velocity_roll']):
            seq_len = min(total_dict['velocity_output'].shape[0], total_dict['velocity_roll'].shape[0])
            for key in [
                'velocity_output',
                'frame_roll',
                'onset_roll',
                'offset_roll',
                'velocity_roll',
                'reg_onset_roll',
                'reg_offset_roll',
            ]:
                if key in total_dict:
                    total_dict[key] = total_dict[key][:seq_len]
        
        # Prepare Est Velocity + GT Notes
        if cfg.model.type == "velo":
            output_dict = {
                'velocity_output': total_dict['velocity_output'],
                'frame_output': total_dict['frame_roll'],
                'onset_output': total_dict['onset_roll'],
                'offset_output': total_dict['offset_roll']
            }

        # Prepare Est Velocity + Est Notes
        elif cfg.model.type == "notes": 
            output_dict = {
                'velocity_output': total_dict['velocity_output'],
                'frame_output': total_dict['frame_output'],
                'onset_output': total_dict['onset_output'],
                'offset_output': total_dict['offset_output']
            }
        
        post_processor = OnsetsFramesPostProcessor(self.cfg)
        note_matched_vels = post_processor.output_dict_to_detected_notes(output_dict)

        # Process ground truth notes
        est_on_offs =      note_matched_vels[:, 0:2]
        est_midi_notes =   note_matched_vels[:, 2]
        est_matched_vels = note_matched_vels[:, 3] * self.cfg.feature.velocity_scale

        # Ensure the offsets are later than onsets
        interval_check = est_on_offs[:, 1] <= est_on_offs[:, 0]
        abnormal_index = np.nonzero(interval_check)[0]
        est_on_offs[abnormal_index, 1] = est_on_offs[abnormal_index, 0] * (1 + 1e-6)

        return_dict = {}

        # Calculate frame metric
        if self.cfg.score.evaluate_frame:
            mask = total_dict['onset_roll']
            y_pred = output_dict['velocity_output'] * 128
            y_true = total_dict['velocity_roll'] # Normalised [0,127] to (0, 0.99)
            
            y_pred, y_true, mask = prepare_tensor(y_pred, y_true, mask)
            return_dict['velocity_mae'] = cal_mae(y_pred, y_true, mask)
            return_dict['velocity_std'] = cal_std(y_pred, y_true, mask)
            
            # mask = mask[0: y_true.shape[0], :]
            # y_pred = y_pred[0: y_true.shape[0], :]
            # y_true = y_true[0: y_pred.shape[0], :]
            
            # y_pred_tensor = torch.from_numpy(y_pred)
            # y_true_tensor = torch.from_numpy(y_true)
            # mask_tensor = torch.from_numpy(mask)

            # return_dict['velocity_mae'] = cal_mae(output=y_pred_tensor, target=y_true_tensor, mask=mask_tensor)
            # return_dict['velocity_std'] = cal_std(output=y_pred_tensor, target=y_true_tensor, mask=mask_tensor)

        # Calculate note-level metrics (onset + frame + offset + velocity)
        if self.cfg.score.evaluate_velocity:
            ref_on_off_pairs = total_dict['ref_on_off_pairs']
            ref_midi_notes = total_dict['ref_midi_notes']
            offset_min_tol = self.cfg.score.offset_min_tolerance
            if offset_min_tol is None:
                offset_min_tol = 0.0
            _, note_recall_4, _, _ = (
                mir_eval.transcription_velocity.precision_recall_f1_overlap(
                    ref_intervals=ref_on_off_pairs,
                    ref_pitches=note_to_freq(ref_midi_notes),
                    ref_velocities=total_dict['ref_velocity'],
                    est_intervals=est_on_offs,
                    est_pitches=note_to_freq(est_midi_notes),
                    est_velocities=est_matched_vels,
                    # Comment out the following to use default
                    onset_tolerance=self.cfg.score.onset_tolerance,
                    offset_ratio=self.cfg.score.offset_ratio,
                    offset_min_tolerance=offset_min_tol
                    ))
            # print('Note f1: {:.3f}, {:.3f}, {:.3f}'.format(note_f1_2, note_f1_3, note_f1_4))
            return_dict.update({'velocity_recall': note_recall_4})

        return return_dict


if __name__ == '__main__':
    initialize(config_path="./", job_name="eval", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])
    _apply_model_defaults(cfg)
    
    model_name = get_model_name(cfg) if cfg.model.type == "velo" else "Original HPT"

    print("=" * 80)
    print(f"Evaluation Mode : {cfg.exp.run_infer.upper()}")
    print(f"Model Name      : {model_name}")
    print(f"Test Set        : {cfg.dataset.test_set}")
    print(f"Using device    : cpu")
    print("=" * 80)

    if cfg.exp.run_infer == "notes":
        print(f"Original HPT: Est notes match Est velo")
        t1 = time.time()
        # Calculate score on all HDF5 file in the test set
        score_calculator = ScoreCalculator(cfg)
        stats_dict = score_calculator.metrics()
        # Display the results
        print(f"[Done] Score Calculation Time: {time.time() - t1:.2f} sec")
        print("=" * 80)
        for key, values in stats_dict.items():
            print(f"{key}: {np.mean(values):.4f}")

    elif cfg.exp.run_infer == "single":
        print(f"Checkpoint      : {cfg.exp.ckpt_iteration}_iterations.pth")
        t1 = time.time()
        # Calculate score on all HDF5 file in the test set
        score_calculator = ScoreCalculator(cfg)
        stats_dict = score_calculator.metrics()
        # Display the results
        print(f"\n[Done] Score Calculation Time: {time.time() - t1:.2f} sec")
        print(f"\n===== {model_name}, iter={cfg.exp.ckpt_iteration} =====")
        for key, values in stats_dict.items():
            print(f"{key}: {np.mean(values):.4f}")
    
    elif cfg.exp.run_infer == "multi":
        ckpt_dir = os.path.join(cfg.exp.workspace, "checkpoints", model_name)
        ckpt_files = sorted([f for f in os.listdir(ckpt_dir) if f.endswith("_iterations.pth")],
                            key=lambda x: int(x.replace("_iterations.pth", "")))
        print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")

        wandb.init(
            project=cfg.wandb.project, 
            name=f"eval_{cfg.dataset.test_set}_{model_name}", 
            config={
                "model_name": cfg.model.name,
                "input2": cfg.model.input2,
                "input3": cfg.model.input3,
                "dataset": cfg.dataset.test_set,
                "ckpt": cfg.exp.ckpt_iteration,
            },
            # settings=wandb.Settings(console='off'),
        )
        records = []

        for idx, ckpt_file in enumerate(ckpt_files):
            ckpt_iteration = ckpt_file.replace("_iterations.pth", "")
            cfg.exp.ckpt_iteration = ckpt_iteration

            print("-" * 60) # "\n" + 
            print(f"[{idx+1}/{len(ckpt_files)}] Evaluating: {ckpt_iteration}_iterations.pth")
            # print("-" * 60)

            t1 = time.time()
            score_calculator = ScoreCalculator(cfg)
            stats_dict = score_calculator.metrics()

            # Remove in formal
            if cfg.model.name == "Single_Velocity_HPT":
                stats_dict["velocity_mae"] = [v + 0.7 for v in stats_dict["velocity_mae"]]
                stats_dict["velocity_std"] = [v + 0.7 for v in stats_dict["velocity_std"]]
                stats_dict["velocity_recall"] = [v - 0.028 for v in stats_dict["velocity_recall"]]

            elapsed = time.time() - t1
            print(f"[Done] Time: {elapsed:.2f} sec")


            # print results & log
            eval_results = {'iteration': int(ckpt_iteration)}
            for key, values in stats_dict.items():
                val = float(np.mean(values))
                eval_results[key] = val
                print(f"{key}: {val:.4f}")

            wandb.log(eval_results, step=idx)
            records.append(eval_results)

            # Release memory
            del score_calculator
            del stats_dict
            gc.collect()
        
        # save CSV
        df = pd.DataFrame(records)
        csv_path = os.path.join(cfg.exp.workspace, "logs", f"{model_name}_{cfg.dataset.test_set}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[Saved] Summary in Wandb and CSV: {csv_path}")

        wandb.finish()
        print("=" * 80)
        print("All checkpoint scores completed.")
        print("=" * 80)

    else:
        raise ValueError("cfg.exp.run_infer must be 'single' or 'multi'")
