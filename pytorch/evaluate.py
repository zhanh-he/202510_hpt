import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
from pytorch_utils import forward_dataloader
from calculate_scores import gt_to_note_list, eval_from_list

def _segments_from_output(output_dict):
    """Convert batched output/target rolls to per-segment dicts used by Kim metrics."""
    velocity = output_dict.get('velocity_output')
    if velocity is None:
        return [], []
    vel_roll = output_dict.get('velocity_roll')
    frame_roll = output_dict.get('frame_roll')
    onset_roll = output_dict.get('onset_roll')
    pedal_roll = output_dict.get('pedal_frame_roll')

    segments = []
    targets = []
    segs = velocity.shape[0]
    for idx in range(segs):
        pred = velocity[idx]
        gt_vel = vel_roll[idx]
        frames = min(pred.shape[0], gt_vel.shape[0])
        seg_pred = {'velocity_output': pred[:frames]}
        segments.append(seg_pred)

        pedal = pedal_roll[idx] if pedal_roll is not None else np.zeros(frames)
        if pedal.ndim > 1:
            pedal = np.squeeze(pedal, axis=-1)
        target_entry = {
            'velocity_roll': gt_vel[:frames],
            'frame_roll': frame_roll[idx][:frames],
            'onset_roll': onset_roll[idx][:frames],
            'pedal_frame_roll': pedal[:frames],
        }
        targets.append(target_entry)
    return segments, targets


def _kim_metrics_from_segments(segments, targets):
    """Run the same Kim-style metrics used in calculate_scores."""
    if not segments or not targets:
        return {}

    (
        frame_max_err,
        frame_max_std,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = gt_to_note_list(segments, targets)
    onset_masked_error, onset_masked_std = eval_from_list(segments, targets)

    stats = {
        'frame_max_error': round(frame_max_err, 4),
        'frame_max_std': round(frame_max_std, 4),
        'onset_masked_error': round(onset_masked_error, 4),
        'onset_masked_std': round(onset_masked_std, 4),
    }
    # Backward-compatible aliases for legacy logging keys.
    stats['velocity_mae'] = stats['frame_max_error']
    stats['velocity_std'] = stats['frame_max_std']
    return stats


class SegmentEvaluator(object):
    
    def __init__(self, model, cfg):
        """Evaluate segment-wise metrics.
        Args: model: object
              batch_size: int
        """
        self.model = model
        self.batch_size = cfg.exp.batch_size
        self.input2 = cfg.model.input2
        self.input3 = cfg.model.input3

    def evaluate(self, dataloader):
        """Evaluate over a few mini-batches.
        Args: dataloader: object, used to generate mini-batches for evaluation.
        Returns: statistics: dict, e.g. {
            'frame_f1': 0.800, 
            (if exist) 'onset_f1': 0.500, 
            (if exist) 'offset_f1': 0.300, 
            ...}
        """
        statistics = {}
        output_dict = forward_dataloader(self.model, dataloader, self.batch_size, self.input2, self.input3)
        if 'velocity_output' in output_dict:
            segments, targets = _segments_from_output(output_dict)
            statistics.update(_kim_metrics_from_segments(segments, targets))
        return statistics
