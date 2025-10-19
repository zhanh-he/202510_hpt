import os
import numpy as np
import h5py
import csv
import librosa
import sox
import logging

from utilities import (create_folder, int16_to_float32, traverse_folder, 
    pad_truncate_sequence, TargetProcessor, write_events_to_midi, 
    plot_waveform_midi_targets)

class Maestro_Dataset(object):
    def __init__(self, cfg):
        """
        This class takes the meta of an audio segment as input and returns
        the waveform and targets of the audio segment. This class is used by 
        DataLoader.

        Args:
          cfg: OmegaConf configuration object.
        """
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"maestro_sr{int(cfg.feature.sample_rate)}")
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        # Used for processing MIDI events to target | GroundTruth
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)
    def __getitem__(self, meta):
        """
        Prepare input and target of a segment for training.
        Args:
          meta: dict, e.g. {
            'year': '2004', 
            'hdf5_name': 'MIDI-Unprocessed_SMF_12_01_2004_01-05_ORIG_MID--AUDIO_12_R1_2004_10_Track10_wav.h5, 
            'start_time': 65.0}
        Returns:
          data_dict: {
            'waveform': (samples_num,)
            'onset_roll': (frames_num, classes_num), 
            'offset_roll': (frames_num, classes_num), 
            'reg_onset_roll': (frames_num, classes_num), 
            'reg_offset_roll': (frames_num, classes_num), 
            'frame_roll': (frames_num, classes_num),
            ‘frame_exonset_roll':(frames_num, classes_num),
            'velocity_roll': (frames_num, classes_num), 
            'mask_roll':  (frames_num, classes_num), 
            'pedal_onset_roll': (frames_num,), 
            'pedal_offset_roll': (frames_num,), 
            'reg_pedal_onset_roll': (frames_num,), 
            'reg_pedal_offset_roll': (frames_num,), 
            'pedal_frame_roll': (frames_num,)}
        """
        [year, hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, year, hdf5_name)
        data_dict = {}

        # Random pitch shift for augmentation
        note_shift = self.random_state.randint(
            low=-self.cfg.feature.max_note_shift,
            high=self.cfg.feature.max_note_shift + 1
        )

        # Load HDF5 file
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.cfg.feature.sample_rate)
            end_sample = start_sample + self.segment_samples

            # Handle boundary cases
            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            # Load and process waveform
            waveform = int16_to_float32(hf['waveform'][start_sample:end_sample])

            if self.cfg.feature.augmentor:
                # Apply waveform augment
                waveform = self.cfg.feature.augmentor.augment(waveform)

            if note_shift != 0:
                # Apply pitch shift
                waveform = librosa.effects.pitch_shift(waveform, self.cfg.feature.sample_rate, 
                    note_shift, bins_per_octave=12)

            data_dict['waveform'] = waveform

            # Process MIDI events
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            target_dict, note_events, pedal_events = self.target_processor.process(
                start_time, midi_events_time, midi_events, extend_pedal=True, note_shift=note_shift
            )

        # Combine input and target
        data_dict.update(target_dict)
        # Add onset-excluded frame roll
        data_dict['exframe_roll'] = target_dict['frame_roll'] * (1 - target_dict['onset_roll'])

        # Debugging
        if self.cfg.exp.debug:
            plot_waveform_midi_targets(data_dict, start_time, note_events)
            exit()

        return data_dict


class MAPS_Dataset(object):
    def __init__(self, cfg):
        """
        Dataset class for the MAPS dataset.
        """
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"maps_sr{int(cfg.feature.sample_rate)}")
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        # Used for processing MIDI events to target | GroundTruth
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)

    def __getitem__(self, meta):
        """
        Prepare input and target for a segment.
        Args:
          meta: dict, e.g., {'hdf5_name': 'Bach_BWV849-01_001_20090916-SMD.h5', 
                             'start_time': 65.0}
        Returns:
          data_dict: dictionary containing waveform and target data.
        """
        [hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)
        data_dict = {}

        # Random pitch shift for augmentation
        note_shift = self.random_state.randint(
            low=-self.cfg.feature.max_note_shift,
            high=self.cfg.feature.max_note_shift + 1
        )

        # Load HDF5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.cfg.feature.sample_rate)
            end_sample = start_sample + self.segment_samples

            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample:end_sample])

            if self.cfg.feature.augmentor:
                # Apply waveform augment
                waveform = self.cfg.feature.augmentor.augment(waveform)

            if note_shift != 0:
                # Apply pitch augment
                waveform = librosa.effects.pitch_shift(waveform, self.cfg.feature.sample_rate, note_shift, bins_per_octave=12)

            data_dict['waveform'] = waveform

            # Process MIDI events
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            target_dict, note_events, pedal_events = self.target_processor.process(
                start_time, midi_events_time, midi_events, extend_pedal=True, note_shift=note_shift)
       
        # Combine input and target
        data_dict.update(target_dict)
        # Add onset-excluded frame roll
        data_dict['exframe_roll'] = target_dict['frame_roll'] * (1 - target_dict['onset_roll'])

        # Debugging
        if self.cfg.exp.debug:
            plot_waveform_midi_targets(data_dict, start_time, note_events)
            exit()

        return data_dict


class SMD_Dataset(object):
    def __init__(self, cfg):
        """
        Dataset class for the SMD dataset.
        """
        self.cfg = cfg
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"smd_sr{int(cfg.feature.sample_rate)}")
        self.segment_samples = int(cfg.feature.sample_rate * cfg.feature.segment_seconds)
        # Used for processing MIDI events to target | GroundTruth
        self.target_processor = TargetProcessor(cfg.feature.segment_seconds, cfg)

    def __getitem__(self, meta):
        """
        Prepare input and target for a segment.
        Args:
          meta: dict, e.g., {'hdf5_name': 'Bach_BWV849-01_001_20090916-SMD.h5', 
                             'start_time': 65.0}
        Returns:
          data_dict: dictionary containing waveform and target data.
        """
        [hdf5_name, start_time] = meta
        hdf5_path = os.path.join(self.hdf5s_dir, hdf5_name)
        data_dict = {}

        # Random pitch shift for augmentation
        note_shift = self.random_state.randint(
            low=-self.cfg.feature.max_note_shift,
            high=self.cfg.feature.max_note_shift + 1
        )

        # Load HDF5
        with h5py.File(hdf5_path, 'r') as hf:
            start_sample = int(start_time * self.cfg.feature.sample_rate)
            end_sample = start_sample + self.segment_samples

            if end_sample >= hf['waveform'].shape[0]:
                start_sample -= self.segment_samples
                end_sample -= self.segment_samples

            waveform = int16_to_float32(hf['waveform'][start_sample:end_sample])

            if self.cfg.feature.augmentor:
                # Apply waveform augment
                waveform = self.cfg.feature.augmentor.augment(waveform)

            if note_shift != 0:
                # Apply pitch augment
                waveform = librosa.effects.pitch_shift(waveform, self.cfg.feature.sample_rate, note_shift, bins_per_octave=12)

            data_dict['waveform'] = waveform

            # Process MIDI events
            midi_events = [e.decode() for e in hf['midi_event'][:]]
            midi_events_time = hf['midi_event_time'][:]
            target_dict, note_events, pedal_events = self.target_processor.process(
                start_time, midi_events_time, midi_events, extend_pedal=True, note_shift=note_shift)
       
        # Combine input and target
        data_dict.update(target_dict)
        # Add onset-excluded frame roll
        data_dict['exframe_roll'] = target_dict['frame_roll'] * (1 - target_dict['onset_roll'])

        # Debugging
        if self.cfg.exp.debug:
            plot_waveform_midi_targets(data_dict, start_time, note_events)
            exit()

        return data_dict


class Augmentor(object):
    def __init__(self, cfg):
        """Data augmentor."""
        
        self.sample_rate = cfg.feature.sample_rate
        self.random_state = np.random.RandomState(cfg.exp.random_seed)

    def augment(self, x):
        clip_samples = len(x)

        logger = logging.getLogger('sox')
        logger.propagate = False

        tfm = sox.Transformer()
        tfm.set_globals(verbosity=0)

        tfm.pitch(self.random_state.uniform(-0.1, 0.1, 1)[0])
        tfm.contrast(self.random_state.uniform(0, 100, 1)[0])

        tfm.equalizer(frequency=self.loguniform(32, 4096, 1)[0], 
            width_q=self.random_state.uniform(1, 2, 1)[0], 
            gain_db=self.random_state.uniform(-30, 10, 1)[0])

        tfm.equalizer(frequency=self.loguniform(32, 4096, 1)[0], 
            width_q=self.random_state.uniform(1, 2, 1)[0], 
            gain_db=self.random_state.uniform(-30, 10, 1)[0])
        
        tfm.reverb(reverberance=self.random_state.uniform(0, 70, 1)[0])

        aug_x = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate)
        aug_x = pad_truncate_sequence(aug_x, clip_samples)
        
        return aug_x

    def loguniform(self, low, high, size):
        return np.exp(self.random_state.uniform(np.log(low), np.log(high), size))


class Sampler(object):
    def __init__(self, cfg, split, is_eval=None):
        """
        Sampler is used to sample segments for training or evaluation.
        Args:
          cfg: OmegaConf configuration containing dataset and experiment details.
          split: 'train' | 'validation' | 'test'.
          random_seed: int, random seed for reproducibility.
        """
        assert split in ['train', 'validation', 'test']
        self.is_eval = is_eval
        # Point test/eval to the same workspace root used by packing
        sr_tag = f"sr{int(cfg.feature.sample_rate)}"
        if split == "test":
            # Evaluate against a specific dataset name passed via is_eval (e.g., "maestro"|"smd"|"maps")
            self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"{is_eval}_{sr_tag}")
        else:
            # Train/validation use configured train_set, suffixed by sample rate
            self.hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', f"{cfg.dataset.train_set}_{sr_tag}")
        self.segment_seconds = cfg.feature.segment_seconds
        self.hop_seconds = cfg.feature.hop_seconds
        self.sample_rate = cfg.feature.sample_rate
        self.random_state = np.random.RandomState(cfg.exp.random_seed)
        self.batch_size = cfg.exp.batch_size
        self.dataset_type = is_eval if split == "test" else cfg.dataset.train_set
        # self.dataset_type = cfg.dataset.test_set if split == "test" else cfg.dataset.train_set
        self.mini_data = cfg.exp.mini_data
        
        
        (hdf5_names, hdf5_paths) = traverse_folder(self.hdf5s_dir)
        self.segment_list = []

        n = 0
        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, 'r') as hf:
                if hf.attrs['split'].decode() == split:
                    audio_name = hdf5_path.split('/')[-1]
                    start_time = 0

                    # Maestro-specific handling
                    if self.dataset_type == "maestro":
                        year = hf.attrs['year'].decode()
                        file_id = [year, audio_name]
                    elif self.dataset_type == "smd":
                        file_id = [audio_name]
                    elif self.dataset_type == "maps":
                        file_id = [audio_name]

                    while start_time + self.segment_seconds < hf.attrs['duration']:
                        self.segment_list.append(file_id + [start_time])
                        start_time += self.hop_seconds
                    n += 1

                    if self.mini_data and n == 10:
                        break

        """self.segment_list looks like:
        [['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 1.0], 
         ['2004', 'MIDI-Unprocessed_SMF_22_R1_2004_01-04_ORIG_MID--AUDIO_22_R1_2004_17_Track17_wav.h5', 2.0]
         ...]"""
        # Log segment count
        log_prefix = "eval " if self.is_eval else ""
        logging.info(f"{log_prefix}{split} segments: {len(self.segment_list)}")

        self.pointer = 0
        self.segment_indexes = np.arange(len(self.segment_list))
        self.random_state.shuffle(self.segment_indexes)

    def __iter__(self):
        while True:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[self.pointer]
                self.pointer += 1

                if self.pointer >= len(self.segment_indexes):
                    self.pointer = 0
                    self.random_state.shuffle(self.segment_indexes)

                batch_segment_list.append(self.segment_list[index])
                i += 1

            yield batch_segment_list

    def __len__(self):
        return int(np.ceil(len(self.segment_list) / self.batch_size))
        
    def state_dict(self):
        state = {
            'pointer': self.pointer, 
            'segment_indexes': self.segment_indexes}
        return state
            
    def load_state_dict(self, state):
        self.pointer = state['pointer']
        self.segment_indexes = state['segment_indexes']


class EvalSampler(Sampler):
    def __init__(self, cfg, split, is_eval):
        """
        Sampler for Evaluation.

        Args:
          cfg: OmegaConf configuration containing dataset and experiment details.
          split: 'train' | 'validation' | 'test'.
          random_seed: int, random seed for reproducibility.
        """
        super().__init__(cfg, split, is_eval)
        self.max_evaluate_iteration = 40 # Limit validation iterations

    def __iter__(self):
        pointer = 0
        iteration = 0

        while iteration < self.max_evaluate_iteration:
            batch_segment_list = []
            i = 0
            while i < self.batch_size:
                index = self.segment_indexes[pointer]
                pointer += 1
                batch_segment_list.append(self.segment_list[index])
                i += 1

            iteration += 1
            yield batch_segment_list


def collate_fn(list_data_dict):
    """Collate input and target of segments to a mini-batch.

    Args:
      list_data_dict: e.g. [
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        {'waveform': (segment_samples,), 'frame_roll': (segment_frames, classes_num), ...}, 
        ...]

    Returns:
      np_data_dict: e.g. {
        'waveform': (batch_size, segment_samples)
        'frame_roll': (batch_size, segment_frames, classes_num), 
        ...}
    """
    np_data_dict = {}
    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])
    
    return np_data_dict
