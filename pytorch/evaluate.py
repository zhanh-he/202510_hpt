import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import torch
from pytorch_utils import forward_dataloader

def prepare_tensor(y_pred, y_true, mask):
    # Force predicted data len same as groundtruth
    y_pred = y_pred[0: y_true.shape[0], :] 
    y_true = y_true[0: y_pred.shape[0], :]
    mask = mask[0: y_true.shape[0], :] 
    y_pred_tensor = torch.from_numpy(y_pred)
    y_true_tensor = torch.from_numpy(y_true)
    mask_tensor = torch.from_numpy(mask)
    return y_pred_tensor, y_true_tensor, mask_tensor

def cal_mae(y_pred, y_true, mask):
    """Mean absolute error (MAE) with mask"""
    abs_diff = torch.abs(y_pred - y_true)
    abs_diff = abs_diff * mask
    return torch.sum(abs_diff) / torch.sum(mask)

def cal_std(y_pred, y_true, mask):
    """Standard Deviation of Absolute Error (std_ae) with mask"""
    abs_diff = torch.abs(y_pred - y_true)
    abs_diff = abs_diff * mask
    mean_abs_diff = torch.sum(abs_diff) / torch.sum(mask)
    squared_diff = (abs_diff - mean_abs_diff) ** 2
    squared_diff = squared_diff * mask
    variance = torch.sum(squared_diff) / torch.sum(mask)
    return torch.sqrt(variance)

def cal_recall(y_pred, y_true, mask, threshold=12.7):
    y_pred, y_true = y_pred[mask], y_true[mask]
    diff = torch.abs(y_pred - y_true)
    tp = torch.sum(diff < threshold).item() # True positives
    fn = torch.sum(y_true != 0).item() - tp # False negatives
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

# def cal_recall(y_pred, y_true, mask, threshold=12.7):
#     y_pred, y_true = y_pred[mask], y_true[mask]
#     diff = np.abs(y_pred - y_true)
#     tp = np.sum(diff < threshold)    # True positives
#     fn = np.sum(y_true != 0) - tp    # False negatives
#     return tp / (tp + fn) if (tp + fn) > 0 else 0.0


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
            """Mask indicates only evaluate where onset exists"""

            # If onset_mask is None, do not count this short segment.
            mask = output_dict['onset_roll'] != 0        
            if mask is None:
                return statistics
            y_pred = output_dict['velocity_output'] * 128
            y_true = output_dict['velocity_roll']

            # Convert to tensor
            y_pred, y_true, mask = prepare_tensor(y_pred, y_true, mask)
            mae = cal_mae(y_pred, y_true, mask)
            std = cal_std(y_pred, y_true, mask)
            recall = cal_recall(y_pred, y_true, mask, threshold=12.7)
            
            # # Calculate recall using the provided function.
            # est_velo, gt_velo = est_velo[mask], gt_velo[mask]
            # recall = cal_recall(y_pred, y_true, mask, threshold=12.7)
            # mae = np.mean(np.abs(est_velo - gt_velo))
            # std = np.std(est_velo - gt_velo)

            # Round the metrics to 4 decimal places, velocity_mse, 
            statistics['velocity_mae'] = round(mae.item(), 4) #np.around(mae, decimals=4)
            statistics['velocity_std'] = round(std.item(), 4) #np.around(std, decimals=4)
            statistics['velocity_recall'] = round(recall, 4) #np.around(recall, decimals=4)

        return statistics