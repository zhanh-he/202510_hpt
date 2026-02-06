import os
import sys
import time
import logging
import torch
import torch.utils.data

# from torch_optimizer import Ranger
from torch.optim import Adam

from pytorch_utils import move_data_to_device
from data_generator import Maestro_Dataset, SMD_Dataset, MAPS_Dataset, Augmentor, Sampler, EvalSampler, collate_fn
from utilities import create_folder, create_logging, RegressionPostProcessor

# from model_HPT import Regress_onset_offset_frame_velocity_CRNN, Regress_pedal_CRNN
from model_HPT import Single_Velocity_HPT, Dual_Velocity_HPT, Triple_Velocity_HPT
from model_FilmUnet import FiLMUNetPretrained
from model_DynEst import DynestAudioCNN

from losses import get_loss_func
from evaluate import SegmentEvaluator

from hydra import initialize, compose
from omegaconf import OmegaConf
import wandb

# https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer

# Initialize WandB
def init_wandb(cfg, wandb_run_id):
    """
    Initialize WandB for experiment tracking.
    """
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        id=wandb_run_id,
        resume="must" if wandb_run_id else "allow",
        config=OmegaConf.to_container(cfg, resolve=True)
    )

def _write_training_stats(cfg, checkpoints_dir, model_name):
    """Persist a short config snapshot next to checkpoints."""
    stats_path = os.path.join(checkpoints_dir, "training_stats.txt")
    file_name = getattr(cfg.wandb, "name", None) or model_name
    condition_inputs = [getattr(cfg.model, "input2", None), getattr(cfg.model, "input3", None)]
    condition_check = any(condition_inputs)
    condition_type = next((c for c in condition_inputs if c), "none")
    condition_net = getattr(cfg.model, "condition_net", "N/A")

    lines = [
        f"file name           :{file_name}",
        f"dev_env             :{getattr(cfg.exp, 'dev_env', 'local')}",
        f"condition_check     :{condition_check}",
        f"condition_net       :{condition_net}",
        f"loss_type           :{cfg.exp.loss_type}",
        f"condition_type      :{condition_type}",
        f"batch_size          :{cfg.exp.batch_size}",
        f"hop_seconds         :{cfg.feature.hop_seconds}",
        f"segment_seconds     :{cfg.feature.segment_seconds}",
        f"frames_per_second   :{cfg.feature.frames_per_second}",
        f"feature type        :{cfg.feature.audio_feature}",
    ]

    with open(stats_path, "w") as f:
        f.write("\n".join(lines))

def train(cfg):
    """
    Train a piano transcription system.
    """
    # Arugments & parameters
    device = torch.device('cuda') if cfg.exp.cuda and torch.cuda.is_available() else torch.device('cpu')
    model = eval(cfg.model.name)(cfg)
    model.kim_loss_alpha = getattr(cfg.exp, "kim_loss_alpha", 0.5)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg.exp.learning_rate)
    
    # Remove in formal
    # if cfg.model.name == 'Single_Velocity_HPT':
    #     optimizer = Ranger(model.parameters(), lr=cfg.exp.learning_rate)
    # else:
    #     optimizer = Adam(model.parameters(), lr=cfg.exp.learning_rate)

    # if cfg.exp.optim == "ranger":
    #     optimizer = Ranger(model.parameters(), lr=cfg.exp.learning_rate)
    # elif cfg.exp.optim == "adam":
    #     optimizer = Adam(model.parameters(), lr=cfg.exp.learning_rate)

    # Paths for results
    extras_inputs = '+'.join(filter(None, [cfg.model.input2, cfg.model.input3]))
    model_name = f"{cfg.model.name}" + (f"+{extras_inputs}" if extras_inputs else "")
    checkpoints_dir = os.path.join(cfg.exp.workspace, 'checkpoints', model_name)
    logs_dir = os.path.join(cfg.exp.workspace, 'logs', model_name)

    # Create logging and checkpoint dir
    create_folder(checkpoints_dir)
    create_folder(logs_dir)
    _write_training_stats(cfg, checkpoints_dir, model_name)
    create_logging(logs_dir, filemode='w')
    logging.info(cfg)
    logging.info(f"Using {device}.")

    # Resume training if applicable
    start_iteration = 0
    wandb_run_id = None
    resume_sampler_state = None
    if cfg.exp.resume_iteration > 0:
        state_path = os.path.join(checkpoints_dir, f'{cfg.exp.resume_iteration}_iterations.pth')
        resume_meta_path = os.path.join(checkpoints_dir, f'{cfg.exp.resume_iteration}_resume.pth')
        if os.path.exists(state_path):
            logging.info(f"Loading checkpoint {state_path} from iteration {cfg.exp.resume_iteration}")
            model_state = torch.load(state_path)
            if isinstance(model_state, dict) and 'model' in model_state and 'optimizer' in model_state:
                # Legacy combined checkpoint
                checkpoint = model_state
                model.load_state_dict(checkpoint['model'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_iteration = checkpoint['iteration']
                wandb_run_id = checkpoint.get('wandb_run_id')
                resume_sampler_state = checkpoint.get('sampler')
            else:
                if not os.path.exists(resume_meta_path):
                    raise FileNotFoundError(f"Missing resume metadata: {resume_meta_path}")
                checkpoint = torch.load(resume_meta_path)
                model.load_state_dict(model_state)
                optimizer.load_state_dict(checkpoint['optimizer'])
                start_iteration = checkpoint['iteration']
                wandb_run_id = checkpoint.get('wandb_run_id')
                resume_sampler_state = checkpoint.get('sampler')
        else:
            logging.warning(f"Checkpoint {state_path} not found. Starting from scratch.")

    # Initialize WandB after potentially loading run_id
    init_wandb(cfg, wandb_run_id)

    # Remove in formal
    if cfg.model.name == 'Single_Velocity_HPT':
        cfg.exp.random_seed = 12

    # Dynamically initialize train | test datasets
    dataset_classes = {
        "maestro": Maestro_Dataset,
        "smd": SMD_Dataset,
        "maps": MAPS_Dataset,
    }
    train_dataset = dataset_classes[cfg.dataset.train_set](cfg)
    # test_dataset = dataset_classes[cfg.dataset.test_set](cfg)
    
    train_sampler = get_sampler(cfg, purpose='train', split='train', is_eval=None)
    if resume_sampler_state:
        train_sampler.load_state_dict(resume_sampler_state)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler, 
        collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True
        )
    eval_train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=get_sampler(cfg, purpose='eval', split='train', is_eval=None), 
        collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True
        )
    # eval_valid_loader = torch.utils.data.DataLoader(
    #     dataset=train_dataset,
    #     batch_sampler=get_sampler(cfg, purpose='eval', split='validation', is_eval=None),
    #     collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True
    #     )
    # eval_test_loader = torch.utils.data.DataLoader(
    #     dataset=test_dataset,
    #     batch_sampler=get_sampler(cfg, purpose='eval', split='test'),
    #     collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True
    #     )
    eval_maestro_loader = torch.utils.data.DataLoader(
        dataset=dataset_classes["maestro"](cfg),
        batch_sampler=get_sampler(cfg, purpose='eval', split='test', is_eval="maestro"),
        collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True
        )
    eval_smd_loader = torch.utils.data.DataLoader(
        dataset=dataset_classes["smd"](cfg),
        batch_sampler=get_sampler(cfg, purpose='eval', split='test', is_eval="smd"),
        collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True
        )
    eval_maps_loader = torch.utils.data.DataLoader(
        dataset=dataset_classes["maps"](cfg),
        batch_sampler=get_sampler(cfg, purpose='eval', split='test', is_eval="maps"),
        collate_fn=collate_fn, num_workers=cfg.exp.num_workers, pin_memory=True
        )
    evaluator = SegmentEvaluator(model, cfg)

    # Check the number of available GPUs, use specific GPU
    gpu_count = torch.cuda.device_count()
    logging.info(f'Number of GPUs available: {gpu_count}')
    if gpu_count > 1:
        torch.cuda.set_device(0)
    model.to(device)

    iteration = start_iteration
    train_bgn_time = time.time()
    train_loss = 0.0
    
    early_phase = int(cfg.exp.total_iteration * 0.05)    # 5% of total iterations, e.g. 10k of 200k
    early_step = int(early_phase * 0.1)                  # 10% of the early phase, e.g. 1k of 10k

    for batch_data_dict in train_loader:

        # Learning rate decay each 10000 iterations
        if cfg.exp.decay:
            if iteration % cfg.exp.reduce_iteration == 0 and iteration != cfg.exp.resume_iteration:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.9
        
        # Forward & Backward
        model.train()
        batch_output_dict, loss = forward_pass(cfg, model, batch_data_dict, device)
        print(iteration, loss)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # if iteration % cfg.exp.eval_iteration == 0: # and iteration != cfg.exp.resume_iteration:
        if ((iteration < early_phase and iteration % early_step == 0) or 
            (iteration >= early_phase and iteration % cfg.exp.eval_iteration == 0)):

            # Evaluate the model
            logging.info('------------------------------------')
            logging.info(f"Iteration: {iteration}/{cfg.exp.total_iteration}")
            train_fin_time = time.time()
            train_loss /= cfg.exp.eval_iteration
            train_statistics = evaluator.evaluate(eval_train_loader)
            # valid_statistics = evaluator.evaluate(eval_valid_loader)
            valid_maestro_statistics = evaluator.evaluate(eval_maestro_loader)
            valid_smd_statistics = evaluator.evaluate(eval_smd_loader)
            valid_maps_statistics = evaluator.evaluate(eval_maps_loader)
            logging.info(f"    Train Loss: {train_loss:.4f}")
            logging.info(f"    Train Stat: {train_statistics}")
            # logging.info(f"    Valid Stat: {valid_statistics}")
            logging.info(f"    Valid Maestro Stat: {valid_maestro_statistics}")
            logging.info(f"    Valid SMD Stat: {valid_smd_statistics}")
            logging.info(f"    Valid MAPS Stat: {valid_maps_statistics}")
            wandb.log({
                "iteration": iteration,
                "train_loss": train_loss,
                "train_stat": train_statistics,
                # "valid_stat": valid_statistics,
                "valid_maestro_stat": valid_maestro_statistics,
                "valid_smd_stat": valid_smd_statistics,
                "valid_maps_stat": valid_maps_statistics,
            })

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))
            
            train_loss = 0.0
            train_bgn_time = time.time()

            # Save model
            checkpoint_path = os.path.join(checkpoints_dir, f'{iteration}_iterations.pth')
            resume_meta_path = os.path.join(checkpoints_dir, f'{iteration}_resume.pth')
            torch.save(model.state_dict(), checkpoint_path)
            resume_payload = {
                'iteration': iteration,
                'sampler': train_sampler.state_dict(),
                'optimizer': optimizer.state_dict(),
                'wandb_run_id': wandb.run.id,
            }
            torch.save(resume_payload, resume_meta_path)
            logging.info(f'Model saved to {checkpoint_path} (state) and {resume_meta_path} (resume)')

        # Stop learning when reaching end
        if iteration == cfg.exp.total_iteration:
            break

        # Increment Iteration Counter
        optimizer.zero_grad()
        iteration += 1
    
    # End WandB Logger
    wandb.finish()


def forward_pass(cfg, model, batch_data_dict, device):
    """
    Return model's output and computed loss for the batch.
    """
    # Move data to device
    for key in batch_data_dict.keys():
        batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)

    # Prepare inputs
    inputs = [batch_data_dict['waveform']]
    if cfg.model.input2:
        inputs.append(batch_data_dict[f"{cfg.model.input2}_roll"])
    if cfg.model.input3:
        inputs.append(batch_data_dict[f"{cfg.model.input3}_roll"])

    # Forward pass & Compute loss
    batch_output_dict = model(*inputs)
    loss = get_loss_func(cfg.exp.loss_type)(model, batch_output_dict, batch_data_dict)

    return batch_output_dict, loss


def get_sampler(cfg, purpose, split, is_eval=None):
    """
    Returns the appropriate sampler based on the purpose (train or eval) and split.
    Remove "is_eval" to use one test set.
    """
    sampler_mapping = {
        'train': Sampler,
        'eval': EvalSampler,
    }

    # Pass `is_eval` when instantiating the sampler
    return sampler_mapping[purpose](cfg, split=split, is_eval=is_eval)



if __name__ == '__main__':
    initialize(config_path="./", job_name="train", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])
    train(cfg)
