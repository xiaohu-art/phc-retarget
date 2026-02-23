import os
import torch
import yaml
import numpy as np
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as sRot

def build_chain(cfg) -> pk.Chain:
    path = cfg['asset']['assetFileName']
    is_urdf = path.endswith(".urdf")
    tree = ET.parse(path)
    root = tree.getroot()
    cwd = os.getcwd()
    os.chdir(os.path.dirname(path))
    chain = pk.build_chain_from_urdf(ET.tostring(root, encoding='utf8'))
    os.chdir(cwd)
    return chain

with open("cfg/fourier_gr3_fitting.yaml", "r") as f:
    cfg = yaml.safe_load(f)

chain = build_chain(cfg)
th = torch.zeros([1, chain.n_joints])
robot_body_pose = chain.forward_kinematics(th)

robot_body_names = [m[0] for m in cfg['joint_matches']]
robot_keypoints = []
for name in robot_body_names:
    if name in robot_body_pose:
        robot_keypoints.append(robot_body_pose[name].get_matrix()[:, :3, 3])
    else:
        print(f"Warning: {name} not found in robot_body_pose")
        robot_keypoints.append(torch.zeros([1, 3]))
robot_keypoints = torch.stack(robot_keypoints, dim=1)
root_translation = robot_body_pose[chain.get_link_names()[0]].get_matrix()[:, :3, 3]
robot_keypoints = robot_keypoints - root_translation.unsqueeze(1)

print("Robot Keypoints (centered at root):")
for i, name in enumerate(robot_body_names):
    print(f"{name}: {robot_keypoints[0, i].tolist()}")

# Calculate distances
l_hip = robot_keypoints[0, robot_body_names.index("left_thigh_pitch_link")]
r_hip = robot_keypoints[0, robot_body_names.index("right_thigh_pitch_link")]
hip_width = torch.norm(l_hip - r_hip).item()
print(f"Hip width: {hip_width:.4f}m")

l_knee = robot_keypoints[0, robot_body_names.index("left_shank_pitch_link")]
thigh_length = torch.norm(l_hip - l_knee).item()
print(f"Thigh length: {thigh_length:.4f}m")

l_ankle = robot_keypoints[0, robot_body_names.index("left_foot_roll_link")]
shank_length = torch.norm(l_knee - l_ankle).item()
print(f"Shank length: {shank_length:.4f}m")
