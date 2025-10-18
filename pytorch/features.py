import numpy as np
import argparse
import csv
import os
import time
import logging
import h5py
import librosa
from hydra import initialize, compose
from utilities import (create_folder, float32_to_int16, create_logging, 
    get_filename, read_metadata, read_midi)


def _sr_tag(cfg):
    return f"sr{int(cfg.feature.sample_rate)}"

def pack_maestro_dataset_to_hdf5(cfg):
    """
    Load & resample MAESTRO audio files, then write to HDF5 files.

    Args:
        cfg: OmegaConf configuration object.
    """
    # Assign dataset and workspace paths
    dataset_dir = cfg.dataset.maestro_dir
    
    # Paths for metadata, output, and logs
    csv_path = os.path.join(dataset_dir, 'maestro-v3.0.0.csv')
    dataset_name = f"maestro_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', dataset_name)
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode='w')
    logging.info(f"Packing MAESTRO dataset: {dataset_dir}")

    # Read metadata
    meta_dict = read_metadata(csv_path)
    audios_num = len(meta_dict['canonical_composer'])
    logging.info(f"Total audios number: {audios_num}")

    feature_time = time.time()
    # Process each audio file
    for n in range(audios_num):
        logging.info(f"{n}: {meta_dict['midi_filename'][n]}")

        # Read MIDI
        midi_path = os.path.join(dataset_dir, meta_dict['midi_filename'][n])
        midi_dict = read_midi(midi_path, "maestro")

        # Load audio
        audio_path = os.path.join(dataset_dir, meta_dict['audio_filename'][n])
        audio, _ = librosa.core.load(audio_path, sr=cfg.feature.sample_rate, mono=True)

        # Define HDF5 output path
        packed_hdf5_path = os.path.join(
            waveform_hdf5s_dir,
            f"{os.path.splitext(meta_dict['audio_filename'][n])[0]}.h5"
        )
        create_folder(os.path.dirname(packed_hdf5_path))

        # Write data to HDF5
        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('canonical_composer', data=meta_dict['canonical_composer'][n].encode(), dtype='S100')
            hf.attrs.create('canonical_title', data=meta_dict['canonical_title'][n].encode(), dtype='S100')
            hf.attrs.create('split', data=meta_dict['split'][n].encode(), dtype='S20')
            hf.attrs.create('year', data=meta_dict['year'][n].encode(), dtype='S10')
            hf.attrs.create('midi_filename', data=meta_dict['midi_filename'][n].encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=meta_dict['audio_filename'][n].encode(), dtype='S100')
            hf.attrs.create('duration', data=meta_dict['duration'][n], dtype=np.float32)

            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)

    # Logging summary
    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Time: {time.time() - feature_time:.3f} s")


def pack_maps_dataset_to_hdf5(cfg):
    """
    Load & resample MAPS audio files, then write to HDF5 files. MAPS is a piano dataset 
    only used for evaluating our piano transcription system (optional). Ref:

    [1] Emiya, Valentin. "MAPS Database A piano database for multipitch
    estimation and automatic transcription of music." 2016

    Args:
        cfg: OmegaConf configuration object.
    """
    # Assign dataset and workspace paths
    dataset_dir = cfg.dataset.maps_dir

    # Parameters and paths
    pianos = ['ENSTDkCl', 'ENSTDkAm']
    dataset_name = f"maps_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', dataset_name)
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode='w')
    logging.info(f"Packing MAPS dataset: {dataset_dir}")

    feature_time = time.time()
    count = 0

    # Load & resample each audio file to a HDF5 file
    for piano in pianos:
        sub_dir = os.path.join(dataset_dir, piano, 'MUS')
        audio_names = [
            os.path.splitext(name)[0]
            for name in os.listdir(sub_dir)
            if os.path.splitext(name)[-1] == '.mid'
        ]

        for audio_name in audio_names:
            print(f"{count}: {audio_name}")
            audio_path = f"{os.path.join(sub_dir, audio_name)}.wav"
            midi_path = f"{os.path.join(sub_dir, audio_name)}.mid"

            # Load audio and MIDI
            audio, _ = librosa.core.load(audio_path, sr=cfg.feature.sample_rate, mono=True)
            midi_dict = read_midi(midi_path, "maps")
            duration = librosa.get_duration(y=audio, sr=cfg.feature.sample_rate)

            # Define HDF5 output path
            packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f"{audio_name}.h5")
            create_folder(os.path.dirname(packed_hdf5_path))

            # Write data to HDF5
            with h5py.File(packed_hdf5_path, 'w') as hf:
                hf.attrs.create('split', data='test'.encode(), dtype='S20')
                hf.attrs.create('duration', data=np.float32(duration))
                hf.attrs.create('midi_filename', data=f"{audio_name}.mid".encode(), dtype='S100')
                hf.attrs.create('audio_filename', data=f"{audio_name}.wav".encode(), dtype='S100')
                hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
                hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
                hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
            count += 1
            # logging.info(f"Processed {audio_name}")
    
    # Logging summary
    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Total files processed: {count}")
    logging.info(f"Time: {time.time() - feature_time:.3f} s")


def pack_smd_dataset_to_hdf5(cfg):
    """
    Pack the SMD dataset into HDF5 files.
    Args:
      dataset_dir: str, directory of dataset
    """
    # Arguments & parameters
    dataset_dir = cfg.dataset.smd_dir

    # Paths
    dataset_name = f"smd_{_sr_tag(cfg)}"
    waveform_hdf5s_dir = os.path.join(cfg.exp.workspace, 'hdf5s', dataset_name)
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', f"{get_filename(__file__)}_{dataset_name}")
    create_logging(logs_dir, filemode='w')
    logging.info(f"Packing SMD dataset: {dataset_dir}")

    feature_time = time.time()
    count = 0

    # Find audio-MIDI pairs
    audio_midi_pairs = [(os.path.splitext(name)[0], os.path.splitext(name)[-1].lower()) for name in os.listdir(dataset_dir)
        if os.path.splitext(name)[-1].lower() in ['.mid', '.mp3']]
    audio_midi_pairs = {name: ext for name, ext in audio_midi_pairs}
    
     # Process each audio-MIDI pair
    for audio_name, ext in audio_midi_pairs.items():
        print(f"{count}: {audio_name}")
        audio_path = os.path.join(dataset_dir, f"{audio_name}.mp3")
        midi_path = os.path.join(dataset_dir, f"{audio_name}.mid")

        # Load audio and MIDI
        audio, _ = librosa.core.load(audio_path, sr=cfg.feature.sample_rate, mono=True)
        midi_dict = read_midi(midi_path, "smd")
        duration = librosa.get_duration(y=audio, sr=cfg.feature.sample_rate)

        # Define HDF5 output path
        packed_hdf5_path = os.path.join(waveform_hdf5s_dir, f"{audio_name}.h5")
        create_folder(os.path.dirname(packed_hdf5_path))

        # Write to HDF5
        with h5py.File(packed_hdf5_path, 'w') as hf:
            hf.attrs.create('split', data='test'.encode(), dtype='S20')
            hf.attrs.create('duration', data=np.float32(duration))
            hf.attrs.create('midi_filename', data=f"{audio_name}.mid".encode(), dtype='S100')
            hf.attrs.create('audio_filename', data=f"{audio_name}.mp3".encode(), dtype='S100')
            hf.create_dataset(name='midi_event', data=[e.encode() for e in midi_dict['midi_event']], dtype='S100')
            hf.create_dataset(name='midi_event_time', data=midi_dict['midi_event_time'], dtype=np.float32)
            hf.create_dataset(name='waveform', data=float32_to_int16(audio), dtype=np.int16)
        count += 1
        # logging.info(f"Processed {audio_name}")

    # Logging summary
    logging.info(f"Write HDF5 to {waveform_hdf5s_dir}")
    logging.info(f"Total files processed: {count}")
    logging.info(f"Time taken: {time.time() - feature_time:.3f} s")


if __name__ == '__main__':
    # Initialize Hydra
    initialize(config_path="./", job_name="features", version_base=None)
    cfg = compose(config_name="config")

    # Argument parser setup
    parser = argparse.ArgumentParser(description='Dataset packing utilities')
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Select a mode of operation')
    subparsers.add_parser('pack_maestro_dataset_to_hdf5', help='Pack Maestro dataset to HDF5')
    subparsers.add_parser('pack_maps_dataset_to_hdf5', help='Pack MAPS dataset to HDF5')
    subparsers.add_parser('pack_smd_dataset_to_hdf5', help='Pack SMD dataset to HDF5')

    # Parse arguments
    args = parser.parse_args()

    # Call the appropriate function based on the mode
    mode_to_function = {
        'pack_maestro_dataset_to_hdf5': pack_maestro_dataset_to_hdf5,
        'pack_maps_dataset_to_hdf5': pack_maps_dataset_to_hdf5,
        'pack_smd_dataset_to_hdf5': pack_smd_dataset_to_hdf5,
    }

    # Execute the selected mode
    if args.mode in mode_to_function:
        mode_to_function[args.mode](cfg)
    else:
        raise ValueError(f"Invalid mode '{args.mode}'. Use --help for available modes.")
