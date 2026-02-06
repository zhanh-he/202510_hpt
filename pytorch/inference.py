import inspect
import inspect
import os
import time

import h5py
import numpy as np
import pickle
import torch
from hydra import compose, initialize
from tqdm import tqdm

from utilities import (
    TargetProcessor,
    RegressionPostProcessor,
    OnsetsFramesPostProcessor,
    create_folder,
    get_filename,
    get_model_name,
    int16_to_float32,
    resolve_hdf5_dir,
    traverse_folder,
    write_events_to_midi,
)
from pytorch_utils import forward, forward_velo
from model_DynEst import DynestAudioCNN
from model_FilmUnet import FiLMUNetPretrained
from model_HPT import Dual_Velocity_HPT, Single_Velocity_HPT, Triple_Velocity_HPT

class TranscriptionBase:
    def __init__(self, checkpoint_path, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda') if cfg.exp.cuda and torch.cuda.is_available() else torch.device('cpu')
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds) # 16000*10
        self.segment_frames = int(round(cfg.feature.frames_per_second * cfg.feature.segment_seconds)) + 1 # 100*10 + 1 
        
        # Initialize and load model
        model_cls = eval(cfg.model.name)
        sig = inspect.signature(model_cls.__init__)
        params = [
            p for p in list(sig.parameters.values())[1:]
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]
        names = {p.name for p in params}
        if {'frames_per_second', 'classes_num'}.issubset(names):
            self.model = model_cls(
                frames_per_second=cfg.feature.frames_per_second,
                classes_num=cfg.feature.classes_num
            )
        else:
            self.model = model_cls(cfg)
        if os.path.getsize(checkpoint_path) == 0:
            raise ValueError(f"Checkpoint file for Inference is empty: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint_path}")

        def _strip_prefix(state, prefix):
            keys = list(state.keys())
            if keys and all(k.startswith(prefix) for k in keys):
                return {k[len(prefix):]: v for k, v in state.items()}
            return state

        state_dict = _strip_prefix(state_dict, "module.")

        target_model = self.model
        if isinstance(self.model, FiLMUNetPretrained):
            inner_state = state_dict
            if any(k.startswith("model.") for k in inner_state.keys()):
                inner_state = {k.replace("model.", "", 1): v for k, v in inner_state.items()}
            inner_state = self.model._prepare_state_dict(inner_state)
            target_model = self.model.model
            state_dict = inner_state

        target_model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)

    def enframe(self, x, is_audio=True):
        """Enframe long sequence to short segments.
        Args:
          x: (1, audio_samples) or (frames, 88)
            (x.shape[0], x.shape[1], frames_num of data segment) = (20020 88 1001)
            (x.shape[0], x.shape[1], samples_num of audio segment) = (1 3200000 160000)
          segment_length: int, length of each segment
          is_audio: bool, True if x is audio samples, False if x is frames
        Returns:
          batch: (N, segment_samples) or (N, frames_num, 88)
        """
        segment_length = self.segment_samples if is_audio else self.segment_frames
        assert x.shape[1 if is_audio else 0] % segment_length == 0
        
        batch = [
            x[:, pointer:pointer + segment_length] if is_audio else x[pointer:pointer + segment_length, :]
            for pointer in range(0, (x.shape[1] if is_audio else x.shape[0]) - segment_length + 1, segment_length // 2)
        ]
        return np.concatenate(batch, axis=0) if is_audio else np.stack(batch, axis=0)

    def deframe(self, x):
        """Deframe predicted segments to original sequence.
        Args:
          x: (N, segment_frames, classes_num)
        Returns:
          y: (audio_frames, classes_num)
        """
        if x.shape[0] == 1:
            return x[0]
        
        x = x[:, :-1, :]
        segment_samples = x.shape[1]
        quarter = max(1, segment_samples // 4)
        three_quarters = segment_samples - quarter
        if three_quarters <= quarter:
            return np.concatenate(x, axis=0)
        """Remove an extra frame in the end of each segment caused by the
        'center=True' argument when calculating spectrogram."""
        y = [
            x[0, :three_quarters],
            *[x[i, quarter:three_quarters] for i in range(1, x.shape[0] - 1)],
            x[-1, quarter:]
        ]
        return np.concatenate(y, axis=0)

class VeloTranscription(TranscriptionBase):
    def transcribe(self, audio, input2=None, input3=None, midi_path=None):
        """Transcribe audio into velocity predictions.
        Args:
          audio: (audio_samples,) - 1st input
          input2: extra input - 2nd
          input3: extra input - 3rd
          midi_path: str, path to write out the transcribed MIDI.
        Returns:
          transcribed_dict, dict: {'velocity_output': (N, segment_frames, classes_num), ...}
        """
        # Pad audio to be evenly divided by segment_samples
        audio = audio[None, :]                                                          # (1, audio_samples)
        audio_len = audio.shape[1] # audio (samples)                                           # if audio length is 306_0000
        audio_segments_num = int(np.ceil(audio_len / self.segment_samples))               # ceil 306_0000/(16000x10) = ceil(19.125) = 20 segments
        pad_audio_len = audio_segments_num * self.segment_samples - audio_len         # padding = 20 * 16_0000 - 306_0000
        pad_audio = np.pad(audio, ((0, 0), (0, pad_audio_len)), mode='constant')  
        # pad_audio = np.concatenate((audio, np.zeros((1, pad_audio_samples))), axis=1)   # pad audio to desired length
        audio_segments = self.enframe(pad_audio, is_audio=True)                         # enframe audio to segments

        def process_extra_input(aux_input, audio_segments_num):
            """Process and pad an auxiliary input to match the audio segments."""
            if aux_input is None:
                return None
            aux_len = aux_input.shape[0] # input (frames)
            aux_segments_num = int(np.ceil(aux_len / self.segment_frames))

            # We expect audio & extra iput segments same amount, if not, pad input until being same
            if aux_segments_num != audio_segments_num:
                aux_segments_num = audio_segments_num
            
            pad_aux_len = aux_segments_num * self.segment_frames - aux_len
            pad_aux = np.pad(aux_input, ((0, pad_aux_len), (0, 0)), mode='constant')
            aux_segments = self.enframe(pad_aux, is_audio=False)
            return aux_segments
        
        input2_segments = process_extra_input(input2, audio_segments_num)
        input3_segments = process_extra_input(input3, audio_segments_num)

        # Exact velocity from the output only
        output_dict = forward_velo(self.model, audio_segments, input2_segments, input3_segments, batch_size=1)
        output_dict['velocity_output'] = self.deframe(output_dict['velocity_output'])[0:audio_len]
        """
        output_dict: {
        'reg_onset_output': (segment_frames, classes_num),   X
        'reg_offset_output': (segment_frames, classes_num),  X
        'velocity_output': (N, segment_frames, classes_num),
        ...}
        """
        # return output_dict
        return {
            'output_dict': output_dict,
        }

class PianoTranscription(TranscriptionBase):
    """
    Transcribe audio into note events predictions.
    Args:
        audio: (audio_samples,)
        midi_path: str, path to write out the transcribed MIDI.
    Returns:
        transcribed_dict, dict: {'output_dict':, ..., 'est_note_events': ..., 'est_pedal_events': ...}
    """
    def transcribe(self, audio, midi_path):
        # Pad audio to be evenly divided by segment_samples
        audio = audio[None, :] # (1, audio_samples)
        audio_len = audio.shape[1]
        segments_num = int(np.ceil(audio_len / self.segment_samples))
        pad_len = segments_num * self.segment_samples - audio_len
        pad_audio = np.pad(audio, ((0, 0), (0, pad_len)), mode='constant')
        # pad_audio = np.concatenate((audio, np.zeros((1, pad_len))), axis=1)
        segments = self.enframe(pad_audio, is_audio=True) # (N, segment_samples)
        
        # Exact all outputs
        output_dict = forward(self.model, segments, batch_size=1) # {'reg_onset_output': (N, segment_frames, classes_num), ...}
        for key in output_dict.keys():
            output_dict[key] = self.deframe(output_dict[key])[0:audio_len]
        """
        output_dict: {
        'reg_onset_output': (segment_frames, classes_num),  Y
        'reg_offset_output': (segment_frames, classes_num), Y
        'frame_output': (segment_frames, classes_num),      Y
        'velocity_output': (segment_frames, classes_num),   Y
        'reg_pedal_onset_output': (segment_frames, 1),      Y
        'reg_pedal_offset_output': (segment_frames, 1),     Y
        'pedal_frame_output': (segment_frames, 1)           Y
        ...}
        """
        # Post processor
        if self.post_processor_type == 'regression':
            post_processor = RegressionPostProcessor(self.cfg)
        elif self.post_processor_type == 'onsets_frames':
            """Google's onsets and frames post processing algorithm. Only used 
            for comparison."""
            post_processor = OnsetsFramesPostProcessor(self.cfg)
        
        est_note_events, est_pedal_events = post_processor.output_dict_to_midi_events(output_dict)
        
        if midi_path:
            write_events_to_midi(0, est_note_events, est_pedal_events, midi_path)
            print('Write out to {}'.format(midi_path))

        return {
            'output_dict': output_dict,
            'est_note_events': est_note_events,
            'est_pedal_events': est_pedal_events
        }


def infer(cfg):
    """Perform inference for notes or velocity, depending on cfg.model.type."""
    # Prepare model name and checkpoint path
    model_name = get_model_name(cfg)
    checkpoint_path = os.path.join(cfg.exp.workspace, "checkpoints", model_name, f"{cfg.exp.ckpt_iteration}_iterations.pth")

    # Load data paths
    hdf5s_dir = resolve_hdf5_dir(cfg.exp.workspace, cfg.dataset.test_set, cfg.feature.sample_rate)
    _, hdf5_paths = traverse_folder(hdf5s_dir)

    # Saving Probabilities Folder
    probs_dir = os.path.join(cfg.exp.workspace, f"probs_{cfg.model.type}", cfg.dataset.test_set, model_name, f"{cfg.exp.ckpt_iteration}_iterations")
    create_folder(probs_dir)

    # Initialize the appropriate transcriptor
    transcriptor = {"notes": PianoTranscription, "velo": VeloTranscription}[cfg.model.type](checkpoint_path=checkpoint_path, cfg=cfg)

    # Perform inference on each HDF5 file in the test set
    progress_bar = tqdm(hdf5_paths, desc=f"Proc {cfg.exp.ckpt_iteration} Ckpt", unit="file", ncols=80)
    for hdf5_path in progress_bar:
        with h5py.File(hdf5_path, 'r') as hf:
            if hf.attrs['split'].decode() == 'test':

                # Load audio and MIDI events
                audio = int16_to_float32(hf['waveform'][:])
                midi_events = [e.decode() for e in hf['midi_event'][:]]
                midi_events_time = hf['midi_event_time'][:]

                # Process ground truths
                segment_seconds = len(audio) / cfg.feature.sample_rate
                target_processor = TargetProcessor(segment_seconds=segment_seconds, cfg=cfg)
                target_dict, note_events, pedal_events = target_processor.process(start_time=0, midi_events_time=midi_events_time, midi_events=midi_events, extend_pedal=True)

                # Extract reference data from note events
                ref_on_off_pairs = np.array([[event['onset_time'], event['offset_time']] for event in note_events])
                ref_midi_notes = np.array([event['midi_note'] for event in note_events])
                ref_velocity = np.array([event['velocity'] for event in note_events])

                # Transcribe audio
                if cfg.model.type == "velo":

                    # Generate exframe_roll
                    target_dict['exframe_roll'] = target_dict['frame_roll'] * (1 - target_dict['onset_roll'])
                    
                    # Additional inputs for velocity transcription
                    input2 = target_dict.get(f"{cfg.model.input2}_roll") if cfg.model.input2 else None
                    input3 = target_dict.get(f"{cfg.model.input3}_roll") if cfg.model.input3 else None
                    
                    transcribed_dict = transcriptor.transcribe(audio, input2, input3, midi_path=None)
                
                elif cfg.model.type == "notes":

                    # Notes transcription
                    transcribed_dict = transcriptor.transcribe(audio, midi_path=None)
                
                output_dict = transcribed_dict['output_dict']

                # Prepare total_dict = output & target (groundtruth)
                total_dict = {key: output_dict[key] for key in output_dict.keys()}

                total_dict.update({
                    'frame_roll': target_dict['frame_roll'],
                    'onset_roll': target_dict['onset_roll'],
                    'offset_roll': target_dict['offset_roll'],
                    'velocity_roll': target_dict['velocity_roll'],
                    'reg_onset_roll': target_dict['reg_onset_roll'],
                    'reg_offset_roll': target_dict['reg_offset_roll'],
                    'ref_on_off_pairs': ref_on_off_pairs,
                    'ref_midi_notes': ref_midi_notes,
                    'ref_velocity': ref_velocity,
                    'checkpoint_iteration': cfg.exp.ckpt_iteration,
                })

                # Save the combined data to Probabilities Folder
                prob_path = os.path.join(probs_dir, f"{get_filename(hdf5_path)}.pkl")
                create_folder(os.path.dirname(prob_path))
                pickle.dump(total_dict, open(prob_path, 'wb'))


if __name__ == '__main__':
    initialize(config_path="./", job_name="infer", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])

    print("=" * 80)
    print(f"Inference Mode : {cfg.exp.run_infer.upper()}")
    print(f"Model Name     : {get_model_name(cfg)}")
    print(f"Test Set       : {cfg.dataset.test_set}")
    print(f"Using Device   : {torch.device('cuda') if cfg.exp.cuda and torch.cuda.is_available() else 'cpu'}")
    print("=" * 80)

    if cfg.exp.run_infer == "single":
        print(f"Checkpoint     : {cfg.exp.ckpt_iteration}_iterations.pth")
        t1 = time.time()
        infer(cfg)
        print(f"\n[Done] Inference time: {time.time() - t1:.2f} sec")


    elif cfg.exp.run_infer == "multi":
        model_name = get_model_name(cfg)
        ckpt_dir = os.path.join(cfg.exp.workspace, "checkpoints", model_name)
        def _iteration_id(filename):
            stem = filename.replace("_iterations.pth", "")
            return int(stem) if stem.isdigit() else None

        ckpt_files = sorted(
            [
                f for f in os.listdir(ckpt_dir)
                if f.endswith("_iterations.pth")
            ],
            key=lambda x: _iteration_id(x) or 0,
        )
        print(f"Found {len(ckpt_files)} checkpoints in {ckpt_dir}")

        total_start = time.time()
        for idx, ckpt_file in enumerate(ckpt_files):
            ckpt_iteration = ckpt_file.replace("_iterations.pth", "")
            cfg.exp.ckpt_iteration = ckpt_iteration
            
            tqdm.write("-" * 60)
            tqdm.write(f"[{idx+1}/{len(ckpt_files)}] {ckpt_iteration}_iterations.pth")

            t1 = time.time()
            infer(cfg)
            tqdm.write(f"[Done] Time: {time.time() - t1:.2f} sec")

        print("\n" + "=" * 80)
        print(f"All checkpoint inference completed in {time.time() - total_start:.2f} sec")
        print("=" * 80)

    else:
        raise ValueError("cfg.exp.run_infer must be 'single' or 'multi'")
