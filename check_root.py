import torch
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET
import os

path = "assets/robot/gr3_6dof/gr3_6dof/urdf/gr3_fourier_hand_6dof.urdf"
tree = ET.parse(path)
root = tree.getroot()
chain = pk.build_chain_from_urdf(ET.tostring(root, encoding='utf8'))

print("Chain root:", chain.get_link_names()[0])
print("Joint names:", chain.get_joint_parent_child_names())

th = torch.zeros([1, chain.n_joints])
pos = chain.forward_kinematics(th)
print("Base link pose:", pos['base_link'].get_matrix())
print("Right hip pose:", pos['right_thigh_pitch_link'].get_matrix())
print("Left hip pose:", pos['left_thigh_pitch_link'].get_matrix())
