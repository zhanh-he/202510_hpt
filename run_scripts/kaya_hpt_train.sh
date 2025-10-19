#!/bin/bash
#SBATCH --job-name=hpt36h
#SBATCH --output=hpt_progress_%A_%a.log
#SBATCH --error=hpt_error_%A_%a.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:p100:1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --mem=16G
#SBATCH --array=0-13
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zhanh.he@research.uwa.edu.au

# Load any required modules (if needed) - module load cuda/11.8 gcc/9.4.0
module load Anaconda3/2024.06 cuda/11.8 gcc/11.5.0
# p100 small: 8cpus, mem16G (max4)
# p100 large: 12cpus, mem24G (max3)
# v100: 12-16cpus, mem16G-32G

# leave in, it lists the environment loaded by the modules - https://wandb.ai/authorize
module list

# Activate the conda environment, may need "conda init"
# source ~/miniconda3/etc/profile.d/conda.sh
source activate hpt_mamba

# Print some useful information, Note: SLURM_JOBID is a unique number for every job.
echo "Running on host: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"
echo "SLURM ID: $SLURM_ARRAY_ID $SLURM_ARRAY_TASK_ID"
# echo "SLURM job ID: $SLURM_JOBID"

#  These are generic variables
FOLDER_NAME=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
EXECUTABLE=$HOME/202510_hpt
SCRATCH=$MYSCRATCH/202510_hpt/$FOLDER_NAME
RESULTS=$MYGROUP/202510_hpt_results/$FOLDER_NAME 

###############################################
# Creates a unique directory in the SCRATCH directory for this job to run in.
if [ ! -d $SCRATCH ]; then
    mkdir -p $SCRATCH
fi
echo SCRATCH is $SCRATCH

###############################################
# Creates a unique directory in your GROUP directory for the results of this job
if [ ! -d $RESULTS ]; then
    mkdir -p $RESULTS
fi
echo the results directory is $RESULTS

#############################################
# Copy input files to $SCRATCH, then change directory to $SCRATCH
echo "Copy path $EXECUTABLE to $SCRATCH"
cp -r $EXECUTABLE $SCRATCH
cd $SCRATCH/202510_hpt

#############################################
# link the dataset to real data folder
# echo "MYSCRATCH path: $MYSCRATCH/workspaces"
ln -s $MYSCRATCH/workspaces/hdf5s $SCRATCH/202510_hpt/workspaces/hdf5s

#############################################
# Run your script with passed arguments

declare -a EXPERIMENTS=(
"python pytorch/main_iter.py feature.audio_feature="logmel" model.name='Single_Velocity_HPT'"
"python pytorch/main_iter.py feature.audio_feature="logmel" model.name='Dual_Velocity_HPT' model.input2='onset'"
"python pytorch/main_iter.py feature.audio_feature="logmel" model.name='Dual_Velocity_HPT' model.input2='frame'"
"python pytorch/main_iter.py feature.audio_feature="logmel" model.name='Dual_Velocity_HPT' model.input2='exframe'"
"python pytorch/main_iter.py feature.audio_feature="logmel" model.name='Triple_Velocity_HPT' model.input2='onset' model.input3='frame'"
"python pytorch/main_iter.py feature.audio_feature="logmel" model.name='Triple_Velocity_HPT' model.input2='onset' model.input3='exframe'"
"python pytorch/main_iter.py feature.audio_feature="logmel" model.name='Triple_Velocity_HPT' model.input2='frame' model.input3='exframe'"
"python pytorch/main_iter.py feature.audio_feature="sone" model.name='Single_Velocity_HPT'"
"python pytorch/main_iter.py feature.audio_feature="sone" model.name='Dual_Velocity_HPT' model.input2='onset'"
"python pytorch/main_iter.py feature.audio_feature="sone" model.name='Dual_Velocity_HPT' model.input2='frame'"
"python pytorch/main_iter.py feature.audio_feature="sone" model.name='Dual_Velocity_HPT' model.input2='exframe'"
"python pytorch/main_iter.py feature.audio_feature="sone" model.name='Triple_Velocity_HPT' model.input2='onset' model.input3='frame'"
"python pytorch/main_iter.py feature.audio_feature="sone" model.name='Triple_Velocity_HPT' model.input2='onset' model.input3='exframe'"
"python pytorch/main_iter.py feature.audio_feature="sone" model.name='Triple_Velocity_HPT' model.input2='frame' model.input3='exframe'"
)

CMD="${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}"
echo "Running: $CMD"
eval $CMD

#############################################
#    $OUTPUT file to the unique results dir
# note this can be a copy or move
mv ./workspaces/checkpoints/ ${RESULTS}/

cd $HOME

###########################
# Clean up $SCRATCH

rm -r $SCRATCH

# Deactivate the conda environment - source or conda deactivate
source deactivate

echo hpt36h $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID finished at `date`
# echo hpt36h $SLURM_JOBID finished at  `date`