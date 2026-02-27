import torch
from smplx import SMPL
import os

DATA_PATH = "data"
body_model = SMPL(model_path=os.path.join(DATA_PATH, "smpl"), gender="neutral")
betas = torch.zeros([1, 10])
result = body_model(betas=betas)
joints = result.joints[0] # [24, 3]
pelvis = joints[0]
l_hip = joints[1]
r_hip = joints[2]
l_shoulder = joints[16]
r_shoulder = joints[17]

print("SMPL Pelvis:", pelvis.tolist())
print("SMPL L_Hip relative to Pelvis:", (l_hip - pelvis).tolist())
print("SMPL R_Hip relative to Pelvis:", (r_hip - pelvis).tolist())
print("SMPL L_Shoulder relative to Pelvis:", (l_shoulder - pelvis).tolist())
