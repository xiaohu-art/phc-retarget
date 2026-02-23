Credit to [PHC](https://github.com/ZhengyiLuo/PHC) and [simple-retarget](https://github.com/btx0424/simple-retarget)

## Installation
Here we use `uv` to manage the environment, please make sure you have installed `uv` first.

```bash
./install.sh
```

Modifications:
- set OMP_NUM_THREADS to 1 for multiprocessing
- simplified the config for retargeting process

## Data preparation

Download the smpl model first:
```bash
uv run gdown --folder https://drive.google.com/drive/folders/1eSfJma_5VuNqaw_IRE8xVn6UgAmvjoeT\?usp\=drive_link -O ./data/smpl
```

Under the `data/AMASS` folder, I have downloaded one motion file for instance:
```bash
|-- data
|   |-- AMASS
|       |-- SFU                                 # sub-dataset name
|           |-- 0005                            # subject id
|               |-- 0005_Walking001_poses.npz   # motion file name
|               |-- ...
```
Of course, you can download the other motion files you like from the [AMASS](https://amass.is.tue.mpg.de/) website.

## Fitting motion data from AMASS

First we need to fit the smpl shape to the humanoid robot:
export XDG_SESSION_TYPE=x11
```bash
uv run scripts/1-fit_smpl_shape.py
```
It will save the fitted smpl `beta` parameters in the `g1_29dof_shape.npz` file.

Then we need to fit the motion data to the humanoid robot:
```bash
uv run scripts/3-fit_smpl_motion_mp.py +motion_path=./data/AMASS +target_fps=30 +output_name=<motion_file>
```
It will save the fitted motion data in the `data/g1_29dof/<output_name>.pkl` file.

Finally, we can visualize the fitted motion data:
```bash
uv run scripts/3-vis_q_mj.py +motion_file=./data/g1_29dof/<output_name>.pkl
```