import os
import numpy as np
import mujoco
import hydra
from hydra.utils import get_original_cwd
import time
import threading
import joblib
import viser
import viser.transforms as tf
from viser.extras import ViserUrdf
import yourdfpy

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")

class MotionVis:
    def __init__(self, motions, server, viser_urdf, robot_frame, index_map):
        self.motions = motions
        self.i = 0
        self.t = 0
        self.motion = self.motions[self.i]
        self.server = server
        self.viser_urdf = viser_urdf
        self.robot_frame = robot_frame
        self.index_map = index_map

        # adding GUI controls
        with self.server.gui.add_folder("Motion Control"):
            self.gui_motion_name = self.server.gui.add_text("Current", initial_value="Motion 0")
            prev_btn = self.server.gui.add_button("Previous (Up)")
            next_btn = self.server.gui.add_button("Next (Down)")

            @prev_btn.on_click
            def _(_) : self.change_motion(-1)
            @next_btn.on_click
            def _(_) : self.change_motion(1)

    def change_motion(self, delta):
        self.i = (self.i + delta) % len(self.motions)
        self.motion = self.motions[self.i]
        self.t = 0
        self.gui_motion_name.value = f"Motion {self.i}"

    def run(self, model, data: mujoco.MjData):
        n_joint = self.motion["joint_pos"].shape[1]

        has_free_joint = (model.nq - n_joint) == 7  # 7 DOF for free joint
        while True:

            joint_angles = self.motion["joint_pos"][self.t]
            root_pos = self.motion["root_pos_w"][self.t]
            q = self.motion["root_quat_w"][self.t]  # [x, y, z, w]

            if has_free_joint:
                data.qpos[7:] = joint_angles
                data.qpos[:3] = root_pos
                data.qpos[3:7] = q[[3, 0, 1, 2]]          
              
            mujoco.mj_forward(model, data)
            
            # update robot frame
            self.robot_frame.position = root_pos
            self.robot_frame.wxyz = (q[3], q[0], q[1], q[2]) 
            
            self.viser_urdf.update_cfg(joint_angles[self.index_map])

            self.t = (self.t + 1) % self.motion["joint_pos"].shape[0]
            time.sleep(1 / self.motion["fps"])

@hydra.main(version_base=None, config_path="../cfg", config_name="fourier_gr3_fitting")
# @hydra.main(version_base=None, config_path="../cfg", config_name="unitree_g1_fitting")
def main(cfg):

    asset_path = cfg.asset.assetFileName
    if not os.path.isabs(asset_path):
        asset_path = os.path.join(get_original_cwd(), asset_path)
    
    print(f"Loading URDF from: {asset_path}")
    urdf_model = yourdfpy.URDF.load(asset_path)
    

    model = mujoco.MjModel.from_xml_path(asset_path)
    data = mujoco.MjData(model)

    # launch viser server
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=5.0, height=5.0)

    robot_frame = server.scene.add_frame("/robot")
    viser_urdf = ViserUrdf(server, urdf_model, root_node_name="/robot")

    # build pk chain for index mapping
    import pytorch_kinematics as pk
    with open(asset_path, "rb") as f:
        chain = pk.build_chain_from_urdf(f.read())
    pk_joint_names = chain.get_joint_parameter_names()
    urdf_joint_names = list(viser_urdf.get_actuated_joint_limits().keys())
    index_map = [pk_joint_names.index(name) for name in urdf_joint_names]

    # load motions and start visualization
    motions = joblib.load(cfg.motion_file)
    motions = list(motions.values())

    # Check for joints that are always zero
    all_joint_pos = np.concatenate([m["joint_pos"] for m in motions], axis=0)
    is_always_zero = np.all(np.abs(all_joint_pos) < 1e-6, axis=0)
    zero_indices = np.where(is_always_zero)[0]
    nonzero_indices = np.where(~is_always_zero)[0]

    
    
    print("\n" + "="*50)
    print(f"检测到 {len(zero_indices)}/{len(pk_joint_names)} 个关节在所有序列中均为 0 (无运动):")
    for idx in zero_indices:
        print(f"编号: {idx:2d} | 关节名: {pk_joint_names[idx]}")
    print("="*50 + "\n")

    print("-" * 20)
    print(f"检测到 {len(nonzero_indices)}/{len(pk_joint_names)} 个关节具有有效运动:")
    for idx in nonzero_indices:
        print(f"编号: {idx:2d} | 关节名: {pk_joint_names[idx]}")
    print("="*50 + "\n")

    motion_vis = MotionVis(motions, server, viser_urdf, robot_frame, index_map)
    
    print(f"Viser server started. Please open your browser to view.")
    motion_vis.run(model, data)

if __name__ == "__main__":
    main()