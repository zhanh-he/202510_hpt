import torch


def _align_time_dim(*tensors):
    """Trim all tensors along time dimension to the minimum shared length."""
    if not tensors:
        return tensors
    min_steps = min(tensor.size(1) for tensor in tensors)
    if all(tensor.size(1) == min_steps for tensor in tensors):
        return tensors
    return tuple(tensor[:, :min_steps] for tensor in tensors)


def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE."""
    output, target, mask = _align_time_dim(output, target, mask)
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)


def mse(output, target, mask):
    """Mean squared error (MSE) with mask"""
    output, target, mask = _align_time_dim(output, target, mask)
    return torch.sum(((output - target) ** 2) * mask) / torch.sum(mask)

# def mae(output, target, mask):
#     """Mean absolute error (MAE) with mask"""
#     abs_diff = torch.abs(output - target)
#     abs_diff = abs_diff * mask
#     mean_abs_diff = torch.sum(abs_diff) / torch.sum(mask)
#     return mean_abs_diff

# def std(output, target, mask):
#     """Standard Deviation of Absolute Error (std_ae) with mask"""
#     abs_diff = torch.abs(output - target)
#     abs_diff = abs_diff * mask
#     mean_abs_diff = torch.sum(abs_diff) / torch.sum(mask)

#     squared_diff = (abs_diff - mean_abs_diff) ** 2
#     squared_diff = squared_diff * mask
#     variance = torch.sum(squared_diff) / torch.sum(mask)
#     return torch.sqrt(variance)

# def std_ae(output, target, mask):
#     """Standard Deviation of Absolute Error (std_ae) with mask"""
#     abs_diff = torch.abs(output - target)
#     masked_diff = abs_diff * mask
#     # Calculate mean absolute difference
#     mean_abs_diff = torch.mean(masked_diff)
#     # Calculate squared differences from mean
#     squared_diff = (masked_diff - mean_abs_diff) ** 2
#     # Mask out non-relevant entries
#     squared_diff_masked = squared_diff * mask
#     # Calculate variance (std_ae is square root of variance)
#     variance = torch.sum(squared_diff_masked) / torch.sum(mask)
#     return torch.sqrt(variance)
    

############ Velocity loss ############
def velocity_bce(model, output_dict, target_dict):                                
    """velocity regression losses only, used bce in HPT"""                                                 ## frame_roll
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    return velocity_loss

def velocity_mse(model, output_dict, target_dict):
    """velocity regression losses only, used mse in ONF"""
    velocity_loss = mse(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    return velocity_loss

# def velocity_mae(model, output_dict, target_dict):
#     """Test the performance"""
#     velocity_loss = mae(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
#     return velocity_loss

# def velocity_std(model, output_dict, target_dict):
#     """Test the performance"""
#     velocity_loss = std_ae(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
#     return velocity_loss

def count_inversions(model, output_dict, target_dict):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mask = target_dict['frame_roll']
    tensor = output_dict['velocity_output']
    mask, tensor = _align_time_dim(mask, tensor)
    
    batch_size, rows, cols = tensor.shape
    inversions = torch.zeros((batch_size, rows), dtype=torch.float).to(device)
    
    total_num = torch.zeros((batch_size, rows), dtype=torch.float).to(device)
    
    for i in range(batch_size):
        for j in range(rows):
            
            total_num[i, j] += 1e-5 # 保证除法不为 0
            
            seq_start = -1  # 序列开始的索引
            for k in range(cols):
                # 检测连续序列的开始
                if mask[i, j, k] == 1 and seq_start == -1:
                    seq_start = k
                # 检测连续序列的结束
                if (mask[i, j, k] == 0 or k == cols - 1) and seq_start != -1:
                    seq_end = k if mask[i, j, k] == 0 else k + 1
                    # 对当前连续序列计算连续的顺序数
                    for m in range(seq_start, seq_end):
                        for n in range(m + 1, seq_end):
                            if tensor[i, j, m] < tensor[i, j, n]:
                                inversions[i, j] += 1
                    seq_len = seq_end - seq_start
                    seq_inversion_num = seq_len * (seq_len - 1) / 2
                    total_num[i, j] += seq_inversion_num
                    
                    seq_start = -1  # 重置序列开始的索引，为检测下一个序列做准备
   
    
    inversion_loss = (inversions / total_num)

    inversion_loss_mask = inversion_loss.unsqueeze(-1).repeat(1, 1, cols).to(device)
    
    inversion_loss = (mask * inversion_loss_mask * tensor).mean()
    
    return inversion_loss


def combined_loss(model, output_dict, target_dict): 
    velocity_bce_loss = velocity_bce(model, output_dict, target_dict)
    count_inversions_loss = count_inversions(model, output_dict, target_dict)
    
    return velocity_bce_loss + count_inversions_loss
                                             

def get_loss_func(loss_type):
    if loss_type == 'velocity_bce':
        return velocity_bce
    elif loss_type == 'velocity_mse':
        return velocity_mse
    # elif loss_type == 'velocity_mae':
    #     return velocity_mae
    # elif loss_type == 'velocity_std':
    #     return velocity_std
    elif loss_type == 'combine_bce':
        return combined_loss
    else:
        raise Exception('Incorrect loss_type!')
