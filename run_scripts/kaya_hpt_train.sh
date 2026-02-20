#!/bin/bash
#SBATCH --job-name=hpt36h
#SBATCH --output=hpt_progress_%A_%a.log
#SBATCH --error=hpt_error_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-11
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au


# Load any required modules (if needed) - module load cuda/11.8 gcc/9.4.0
module load Anaconda3/2024.06 cuda/11.8 gcc/11.5.0
module list
source activate bark_env #hpt_mamba

# Print some useful information, Note: SLURM_JOBID is a unique number for every job.
echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "SLURM ID: $SLURM_ARRAY_ID $SLURM_ARRAY_TASK_ID"

#  These are generic variables
FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
EXECUTABLE=$HOME/202510_hpt
SCRATCH=$MYSCRATCH/202510_hpt/$FOLDER_NAME
RESULTS=$MYGROUP/202510_hpt_results/$FOLDER_NAME 

# Creates a unique directory in the SCRATCH directory for this job to run in.
mkdir -p $SCRATCH $RESULTS
echo SCRATCH is $SCRATCH
echo RESULTS dir is $RESULTS

# Copy input files to $SCRATCH, then change directory to $SCRATCH
echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r $EXECUTABLE $SCRATCH
cd $SCRATCH/202510_hpt

# link the dataset to real data folder
# echo "MYSCRATCH path: $MYSCRATCH/workspaces"
# ln -s $MYSCRATCH/workspaces/hdf5s $SCRATCH/202510_hpt/workspaces/hdf5s
DATA_SRC=$MYSCRATCH/202510_hpt_data/workspaces/hdf5s
DATA_VIEW=$SCRATCH/202510_hpt/workspaces/hdf5s
ln -s $DATA_SRC $DATA_VIEW

#############################################
# Run your script with passed arguments

declare -a EXPERIMENTS=(
"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Single_Velocity_HPT'"
"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Single_Velocity_HPT' loss.loss_type='kim_bce_l1' wandb.comment='kimloss'"

"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Dual_Velocity_HPT' model.input2='frame'"
"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Dual_Velocity_HPT' model.input2='frame' loss.loss_type='kim_bce_l1' wandb.comment='kimloss'"

"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='DynestAudioCNN'"
"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='DynestAudioCNN' loss.loss_type='kim_bce_l1' wandb.comment='kimloss'"

"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='HPPNet_SP'"
"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='HPPNet_SP' loss.loss_type='kim_bce_l1' wandb.comment='kimloss'"

"python pytorch/train_iter.py model.name='HPPNet_SP' feature.audio_feature='cqt' feature.sample_rate=16000 feature.frames_per_second=50 feature.hop_seconds=1.0 feature.segment_seconds=20 exp.batch_size=3"
"python pytorch/train_iter.py model.name='HPPNet_SP' feature.audio_feature='cqt' feature.sample_rate=16000 feature.frames_per_second=50 feature.hop_seconds=1.0 feature.segment_seconds=20 exp.batch_size=3 loss.loss_type='kim_bce_l1' wandb.comment='kimloss'"

"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Dual_Velocity_HPT' model.input2='onset'"
"python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Dual_Velocity_HPT' model.input2='onset' loss.loss_type='kim_bce_l1' wandb.comment='kimloss'"

# "python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Triple_Velocity_HPT' model.input2='onset' model.input3='exframe'"
# "python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Triple_Velocity_HPT' model.input2='onset' model.input3='exframe' loss.loss_type='kim_bce_l1' wandb.comment='kimloss'"
# "python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Dual_Velocity_HPT' model.input2='frame'"
# "python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Dual_Velocity_HPT' model.input2='exframe'"
# "python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Triple_Velocity_HPT' model.input2='onset' model.input3='frame'"
# "python pytorch/train_iter.py feature.audio_feature='logmel' model.name='Triple_Velocity_HPT' model.input2='frame' model.input3='exframe'"
)

CMD="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"
eval $CMD

#############################################
#    $OUTPUT file to the unique results dir
# note this can be a copy or move
mv ./workspaces/checkpoints/ ${RESULTS}/
cd $HOME
rm -r $SCRATCH # clean up the scratch space
source deactivate # Deactivate the conda environment - source or conda deactivate
echo hpt36h $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
