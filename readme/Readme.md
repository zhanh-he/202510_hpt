# Introduction
This project is estimating the MIDI Velocity by the audio & MIDI score. Current code wasn't cleanup but okay to run, change setting of `pytorch/config.yaml` should be careful. You can find our paper results with its training params on the **[Wandb Report](public_available_later_for_anonymous)**. We will cleanup and submit github repo if paper is accepted.


## 1. Download the midi dataset
Please download the MAESTRO v3.0.0 and Saarland Music Data, place it in the `/dataset` folder.
- MAESTRO v3: https://magenta.tensorflow.org/datasets/maestro
- Saarland Music Data (version 2): https://zenodo.org/records/13753319
- MAPS dataset: its old public link is broken. You can request it from most music research group or us (sharing via dropbox, etc.).


## 2. Create environment & Setup wandb logger
Tested on Ubuntu 20.04 & Cuda 12.0, and Ubuntu 22.04 & Cuda 12.2.
- Using Conda environment file:
```bash
conda env create -f hpt_env.yaml
conda activate hpt_env
```
Wandb is used for experiment tracking. To switch to TensorBoard, modify `pytorch/train_iter.py`. Otherwise, login your wandb account:
```bash
wandb login # --relogin
```


## 3. Proprocess the data
Now run `Data_Proprocessing.ipynb` to convert MAESTRO/SMD/MAPS dataset midi files into hdf5s, for fast loading during the training and testing step.


## 4. Train the model
Please check the hyperparameters in `pytorch/config.yaml`. The training can be start with the `Train.ipynb`.


## 5. Evaluate the model (Tables 1 & 2)
The evaluation can be start with `Test.ipynb`. Results will be saved to [wandb](public_available_later_for_anonymous), you can check ours as example & verify purpose.


## 6. Interface with trained model (Figure 1)
You can interface with the trained model and visualise its results with `Data_Visual.ipynb`. This is supporting the Figure 1 in our paper.