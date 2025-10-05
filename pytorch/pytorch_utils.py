import numpy as np
import torch

from utilities import pad_truncate_sequence


def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def append_to_dict(dict, key, value):
    
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

 
def forward_dataloader(model, dataloader, batch_size, input2, input3, return_target=True):
    """Forward data generated from dataloader to model.
    Args: model: object
          dataloader: object, used to generate mini-batches for evaluation.
          batch_size: int
          input2: onset, frame, exframe. default is None
          input3: onset, frame, exframe. default is None
          return_target: bool
    Keys in batch_data_dict
    [waveform  onset_roll  offset_roll  reg_onset_roll  reg_offset_roll  frame_roll  
     velocity_roll  mask_roll  reg_pedal_onset_roll  pedal_onset_roll  pedal_offset_roll
     reg_pedal_offset_roll  pedal_frame_roll]
    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        'frame_roll': (segments_num, frames_num, classes_num),
        'onset_roll': (segments_num, frames_num, classes_num),...}
    """
    output_dict = {}
    device = next(model.parameters()).device
    for n, batch_data_dict in enumerate(dataloader):
        batch_input1 = move_data_to_device(batch_data_dict['waveform'], device)
        if input2 is not None:
            batch_input2 = move_data_to_device(batch_data_dict[f'{input2}_roll'], device)
        if input3 is not None:
            batch_input3 = move_data_to_device(batch_data_dict[f'{input3}_roll'], device)

        with torch.no_grad():
            model.eval()
            if input2 is not None:
                if input3 is not None:
                    batch_output_dict = model(batch_input1, batch_input2, batch_input3)
                else:
                    batch_output_dict = model(batch_input1, batch_input2)
            else:
                batch_output_dict = model(batch_input1)

        for key in batch_output_dict.keys():
            if '_list' not in key:
                append_to_dict(output_dict, key, 
                    batch_output_dict[key].data.cpu().numpy())

        if return_target:
            for target_type in batch_data_dict.keys():
                if 'roll' in target_type or 'reg_distance' in target_type or \
                    'reg_tail' in target_type:
                    append_to_dict(output_dict, target_type, 
                        batch_data_dict[target_type])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    
    return output_dict


def forward(model, x, batch_size):
    """Forward data to model in mini-batch. 
    Args: 
      model: object
      x: (N, segment_samples)
      batch_size: int
    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        ...}
    """
    
    output_dict = {}
    device = next(model.parameters()).device
    
    pointer = 0
    while True:
        if pointer >= len(x):
            break
        batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
        pointer += batch_size
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)
            
        for key in batch_output_dict.keys():
            # if '_list' not in key:
            append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict


def forward_velo(model, x1, x2, x3, batch_size):
    """Forward data to model in mini-batch.
    Args:
      model: object
      x1: audio_segments (N, segment_samples)
      x2: onset_roll(N, frames_num, classes_num)
      x3: frame_roll or exframe_roll (onset-excluded frame) (N, frames_num, classes_num)
      batch_size: int
    Returns:
      output_dict: dict, {'velocity_output': (segments_num, frames_num, classes_num)}
    """
    output_dict = {}
    device = next(model.parameters()).device
    pointer = 0

    while pointer < len(x1):
        # Process batches
        batch_audio = move_data_to_device(x1[pointer: pointer + batch_size], device)
        batch_input2 = move_data_to_device(x2[pointer: pointer + batch_size], device) if x2 is not None else None
        batch_input3 = move_data_to_device(x3[pointer: pointer + batch_size], device) if x3 is not None else None
        pointer += batch_size

        with torch.no_grad():
            model.eval()
            inputs = [batch_audio]
            if x2 is not None:
                inputs.append(batch_input2)
            if x3 is not None:
                inputs.append(batch_input3)
            batch_output_dict = model(*inputs)

        append_to_dict(output_dict, 'velocity_output', batch_output_dict['velocity_output'].data.cpu().numpy())

    output_dict['velocity_output'] = np.concatenate(output_dict['velocity_output'], axis=0)
    return output_dict

    #     for key in batch_output_dict.keys():
    #         append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())
    #
    # for key in output_dict.keys():
    #     output_dict[key] = np.concatenate(output_dict[key], axis=0)
    # return output_dict