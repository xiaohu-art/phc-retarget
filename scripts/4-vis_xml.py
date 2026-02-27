import os
import numpy as np
import mujoco
import mujoco.viewer
import hydra
import time
import glob
import threading
from scipy.spatial.transform import Rotation as sRot, Slerp
import joblib

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")


class MotionVis:
    def __init__(self, motions):
        self.motions = motions
        self.i = 0
        self.t = 0
        self.motion = self.motions[self.i]

    def key_callback(self, keycode):
        # print(keycode)
        if keycode == 265: # up arrow, prev motion
            self.i = (self.i - 1) % len(self.motions)
            self.motion = self.motions[self.i]
            self.t = 0
        elif keycode == 264: # down arrow, next motion
            self.i = (self.i + 1) % len(self.motions)
            self.motion = self.motions[self.i]
            self.t = 0

    def run(self, model, data: mujoco.MjData, viewer: mujoco.viewer.Handle):
        while viewer.is_running():
            with viewer.lock():
                data.qpos[7:] = self.motion["joint_pos"][self.t]
                data.qpos[:3] = self.motion["root_pos_w"][self.t]
                data.qpos[3:7] = self.motion["root_quat_w"][self.t][[3, 0, 1, 2]]
                self.t = (self.t + 1) % self.motion["joint_pos"].shape[0]
                mujoco.mj_forward(model, data)
            time.sleep(1 / self.motion["fps"])


@hydra.main(version_base=None, config_path="../cfg", config_name="unitree_g1_fitting")
def main(cfg):
    mjcf_path = cfg.asset.assetFileName

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    data = mujoco.MjData(model)

    motions = joblib.load(cfg.motion_file)
    motions = list(motions.values())

    motion_vis = MotionVis(motions)
    
    viewer = mujoco.viewer.launch_passive(model, data, key_callback=motion_vis.key_callback)
    
    def render_async():
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.02)
    threading.Thread(target=render_async).start()
    
    motion_vis.run(model, data, viewer)


if __name__ == "__main__":
    main()