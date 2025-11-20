import os
import torch
import torch.nn as nn
import hydra
import numpy as np
import pytorch_kinematics as pk
import xml.etree.ElementTree as ET

from smplx import SMPL
from scipy.spatial.transform import Rotation as sRot

from math_utils import quat_mul, quat_from_matrix, quat_error_magnitude

DATA_PATH = os.path.join(os.path.dirname(__file__), "../data")

SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]


def parse_joint_limits_from_mjcf(root):
    limits = {}
    for j in root.findall(".//joint"):
        jtype = j.get("type", "hinge")
        if jtype not in ("hinge", "slide"):
            continue
        name = j.get("name")
        rng = j.get("range")
        if name is None or rng is None:
            continue
        low, high = map(float, rng.split())
        limits[name] = (low, high)
    return limits


def build_chain(cfg) -> pk.Chain:
    mjcf_path = cfg.asset.assetFileName

    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    # remove the free joint of the base link
    root_name = cfg.get("root_name", "pelvis")
    root_body = root.find(f".//body[@name='{root_name}']")
    root_joint = root.find(".//joint[@type='free']")
    root_body.remove(root_joint)
    root_body.set("pos", "0 0 0")

    for extend_config in cfg.extend_config:
        parent = root.find(f".//body[@name='{extend_config.parent_name}']")
        if parent is None:
            raise ValueError(f"Parent body {extend_config.parent_name} not found in MJCF")
        
        pos = extend_config.pos
        rot = extend_config.rot
        # create and insert a dummy body with a fixed joint
        body = ET.Element("body", name=extend_config.joint_name)
        body.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
        body.set("quat", f"{rot[0]} {rot[1]} {rot[2]} {rot[3]}")
        inertial = ET.Element("inertial", pos="0 0 0", quat="0 0 0 1", mass="0.1", diaginertia="0.1 0.1 0.1")
        body.append(inertial)
        parent.append(body)

    cwd = os.getcwd()
    os.chdir(os.path.dirname(mjcf_path))
    chain = pk.build_chain_from_mjcf(ET.tostring(root, method="xml"), body=root_name)
    os.chdir(cwd)

    joint_limits = parse_joint_limits_from_mjcf(root)
    return chain, joint_limits


@hydra.main(version_base=None, config_path="../cfg", config_name="unitree_g1_fitting")
def main(cfg):
    chain, joint_limits = build_chain(cfg)
    chain.forward_kinematics(torch.zeros(1, chain.n_joints))

    motion_path = os.path.join(DATA_PATH, "AMASS/SFU/0005/0005_Walking001_poses.npz")

    with open(motion_path, "rb") as f:
        motion = dict(np.load(f))
    

    fps = int(motion["mocap_framerate"].item()) \
        if "mocap_framerate" in motion \
        else int(motion["mocap_frame_rate"].item())
    T = motion["poses"].shape[0]
    motion["poses"] = motion["poses"][:, :66].reshape(T, 22, 3)
    body_pose = torch.as_tensor(motion["poses"][:, 1:], dtype=torch.float32)
    hand_pose = torch.zeros(T, 2, 3)
    data = {
        "body_pose": torch.cat([body_pose, hand_pose], dim=1),
        "global_orient": torch.as_tensor(motion["poses"][:, 0], dtype=torch.float32),
        "trans": torch.as_tensor(motion["trans"], dtype=torch.float32),
    }

    body_model = SMPL(model_path=os.path.join(DATA_PATH, "smpl"), gender="neutral")

    path = os.path.join(os.path.dirname(__file__), f"{cfg.humanoid_type}_shape.npz")
    fitted_shape = torch.from_numpy(np.load(path)["betas"])

    with torch.no_grad():
        result = body_model.forward(
            fitted_shape,
            body_pose=data["body_pose"].reshape(T, 69),
            global_orient=data["global_orient"],
            transl=data["trans"],
            return_full_pose=True
        )
    
    # which joints to match
    robot_body_names = []
    smpl_joint_idx = []
    for robot_body_name, smpl_joint_name in cfg.joint_matches:
        robot_body_names.append(robot_body_name)
        smpl_joint_idx.append(SMPL_BONE_ORDER_NAMES.index(smpl_joint_name))

    # since the betas are changed and so are the SMPL body morphology,
    # we need to make some corrections to avoid ground pentration
    ground_offset = result.vertices[:, :, 2].min()
    smpl_keypoints_w = result.joints[:, smpl_joint_idx] - ground_offset

    # Acquire SMPL body pose in world frame
    full_pose_aa = result.full_pose.reshape(T, 24, 3).numpy()
    full_pose_quat = sRot.from_rotvec(full_pose_aa.reshape(T * 24, 3)).as_quat()
    full_pose_quat = full_pose_quat.reshape(T, 24, 4)[:, :, [3, 0, 1, 2]]       # [w, x, y, z]

    smpl_local_quat = torch.as_tensor(full_pose_quat, dtype=torch.float32)
    parents = body_model.parents

    smpl_global_quat = torch.zeros_like(smpl_local_quat)
    smpl_global_quat[:, 0] = smpl_local_quat[:, 0]
    for i in range(1, 24):
        parent_idx = parents[i]
        smpl_global_quat[:, i] = quat_mul(smpl_global_quat[:, parent_idx], smpl_local_quat[:, i])

    smpl_global_quat_modified = smpl_global_quat.clone()
    for joint_name, quat_offset in cfg.quat_offset.items():
        joint_idx = SMPL_BONE_ORDER_NAMES.index(joint_name)
        quat_offset = torch.as_tensor(eval(quat_offset), dtype=torch.float32)
        smpl_global_quat_modified[:, joint_idx] = quat_mul(
                                        smpl_global_quat[:, joint_idx], 
                                        quat_offset.expand_as(smpl_global_quat[:, joint_idx])
                                    )
    smpl_keypoints_quat_w = smpl_global_quat_modified[:, smpl_joint_idx]

    # Extract joint limits
    joint_names = chain.get_joint_parameter_names()
    assert len(joint_names) == len(joint_limits), f"Number of joints in chain ({len(joint_names)}) does not match number of joint limits ({len(joint_limits)})"

    low_list, high_list = [], []
    for joint_name, (joint_limit_key, joint_limit_val) in zip(joint_names, joint_limits.items()):
        assert joint_name == joint_limit_key, f"Joint name {joint_name} does not match joint name in joint limits ({joint_limit_key})"
        low_list.append(joint_limit_val[0])
        high_list.append(joint_limit_val[1])
    low = torch.as_tensor(low_list, dtype=torch.float32).unsqueeze(0)   # [1, J]
    high = torch.as_tensor(high_list, dtype=torch.float32).unsqueeze(0) # [1, J]

    robot_rot = sRot.from_rotvec(data["global_orient"]) * sRot.from_euler("xyz", [np.pi/2, 0., np.pi/2]).inv()
    robot_rotmat = torch.as_tensor(robot_rot.as_matrix(), dtype=torch.float32)

    robot_th = torch.nn.Parameter(torch.zeros(T, chain.n_joints))        
    robot_trans = torch.nn.Parameter(data["trans"].clone() - ground_offset)
    opt = torch.optim.Adam([robot_th, robot_trans], lr=0.02)

    indices = chain.get_all_frame_indices()
    
    def mat_rotate(rotmat, v):
        return (rotmat @ v.unsqueeze(-1)).squeeze(-1)
    
    print("=" * 100)
    print(f"{'iter':>6} | {'total_loss':>12} | {'kp_pos_err':>12} | {'kp_quat_err':>12} | {'qpos_reg':>12} | {'qvel_reg':>12} | {'qlimit_reg':>12}")
    print("=" * 100)
        
    for i in range(500):
        fk_output = chain.forward_kinematics(robot_th, indices) # in robot's root frame
        robot_keypoints_b = torch.stack([
            fk_output[name].get_matrix()[:, :3, 3]
            for name in robot_body_names
        ], dim=1)        
        # convert to world frame
        robot_keypoints_w = robot_trans.unsqueeze(1) + mat_rotate(robot_rotmat.unsqueeze(1), robot_keypoints_b)

        robot_orient_mats_b = torch.stack([
            fk_output[name].get_matrix()[:, :3, :3]
            for name in robot_body_names
        ], dim=1) # Shape: [T, N, 3, 3]
        robot_orient_mats_w = torch.matmul(robot_rotmat.unsqueeze(1), robot_orient_mats_b)

        robot_orient_quat_w = quat_from_matrix(robot_orient_mats_w)

        omega = torch.gradient(robot_th, spacing=1/fps, dim=0)[0]    # [T, J]
        
        violate_low  = torch.relu(low  - robot_th)
        violate_high = torch.relu(robot_th - high)
        L_limit = (violate_low**2 + violate_high**2).mean()

        keypoints_pos_error = nn.functional.mse_loss(robot_keypoints_w, smpl_keypoints_w)
        keypoints_quat_error = 1e-2 *quat_error_magnitude(robot_orient_quat_w, smpl_keypoints_quat_w).square().mean()
        joint_pos_reg = 1e-2 * torch.mean(torch.square(robot_th))
        joint_vel_reg = 1e-3 * torch.mean(torch.square(omega))
        joint_limit_reg = 1e2 * L_limit
        loss = keypoints_pos_error + keypoints_quat_error + joint_pos_reg + joint_vel_reg + joint_limit_reg
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print(f"{i:6d} | {loss.item():12.6f} | "
                  f"{keypoints_pos_error.item():12.6f} | "
                  f"{keypoints_quat_error.item():12.6f} | "
                  f"{joint_pos_reg.item():12.6f} | "
                  f"{joint_vel_reg.item():12.6f} | "
                  f"{joint_limit_reg.item():12.6f}")
    
    with torch.no_grad():
        fk_output = chain.forward_kinematics(robot_th, indices)
        robot_keypoints_b = torch.stack([
            fk_output[name].get_matrix()[:, :3, 3]
            for name in robot_body_names
        ], dim=1)
        robot_keypoints_w = robot_trans.unsqueeze(1) + mat_rotate(robot_rotmat.unsqueeze(1), robot_keypoints_b)
    
    motion_name = motion_path.split("/")[-1].split(".")[0]
    save_path = f"{motion_name}.pt"
    data = {
        "joint_pos": robot_th.data.numpy(),
        "keypoint_pos_w": robot_keypoints_w.data.numpy(),
        "root_pos_w": robot_trans.data.numpy(),
        "root_quat_w": robot_rot.as_quat(),
    }
    print(f"Saving to {save_path}")
    np.savez_compressed(save_path, **data)

    def init_mesh(vertices, color=[0.3, 0.3, 0.3]):
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(body_model.faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(color)
        return mesh
    
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh = init_mesh(result.vertices[0])
    vis.add_geometry(mesh)

    plane = o3d.geometry.TriangleMesh.create_box(4., 4., 0.01)
    plane.translate([-2., -2., -0.005])
    plane.compute_vertex_normals()
    vis.add_geometry(plane)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    frame.compute_vertex_normals()
    vis.add_geometry(frame)

    robot_pcd = o3d.geometry.PointCloud()
    robot_pcd.points = o3d.utility.Vector3dVector(robot_keypoints_w[0])
    robot_pcd.colors = o3d.utility.Vector3dVector(torch.tensor([0, 0, 1]).expand_as(robot_keypoints_w[0]))
    vis.add_geometry(robot_pcd)

    smpl_pcd = o3d.geometry.PointCloud()
    smpl_pcd.points = o3d.utility.Vector3dVector(smpl_keypoints_w[0])
    smpl_pcd.colors = o3d.utility.Vector3dVector(torch.tensor([1, 0, 0]).expand_as(smpl_keypoints_w[0]))
    vis.add_geometry(smpl_pcd)

    opt = vis.get_render_option()
    # Set the point size
    point_size = 10.0  # You can adjust this value according to your needs
    opt.point_size = point_size

    for t in range(T):
        mesh.vertices = o3d.utility.Vector3dVector(result.vertices[t]-ground_offset)
        # mesh.compute_vertex_normals()
        # vis.update_geometry(mesh)

        robot_pcd.points = o3d.utility.Vector3dVector(robot_keypoints_w[t])
        vis.update_geometry(robot_pcd)

        smpl_pcd.points = o3d.utility.Vector3dVector(smpl_keypoints_w[t])
        vis.update_geometry(smpl_pcd)

        vis.poll_events()
        vis.update_renderer()
    


if __name__ == "__main__":
    main()