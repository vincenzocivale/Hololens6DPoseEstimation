# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from FoundationPose.estimater import *
from FoundationPose.datareader import *
import argparse


if __name__=='__main__':
  import os
import argparse
import logging
import numpy as np
import cv2
import trimesh
import imageio
import open3d as o3d
import dr # Assuming you have a module 'dr' for RasterizeCudaContext
import Hl2ssStreamer 

# Parse arguments
parser = argparse.ArgumentParser()
code_dir = os.path.dirname(os.path.realpath(__file__))
parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
parser.add_argument('--est_refine_iter', type=int, default=5)
parser.add_argument('--track_refine_iter', type=int, default=2)
parser.add_argument('--debug', type=int, default=1)
parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
args = parser.parse_args()

# Setup logging, seed, and environment
set_logging_format()
set_seed(0)

# Load the mesh
mesh = trimesh.load(args.mesh_file)
to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)

# Clear and setup debug directory
debug = args.debug
debug_dir = args.debug_dir
os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

# Initialize pose estimation components
scorer = ScorePredictor()
refiner = PoseRefinePredictor()
glctx = dr.RasterizeCudaContext()
est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
logging.info("Estimator initialization done")

# Initialize Live Reader for real-time frame acquisition
host = '172.18.165.68'  
calibration_path = '../calibration' 
reader = Hl2ssStreamer(host=host, calibration_path=calibration_path)

# Main loop for processing frames in real-time
frame_count = 0  # Counter for frames
try:
    while True:
        logging.info(f'Processing frame {frame_count}')

        # Get current color and depth frames from the live reader
        color, depth = reader.get_current_frame()
        if color is None or depth is None:
            logging.warning('Frame data not available, skipping frame')
            continue

        if frame_count == 0:
            # For the first frame, get the object mask (this could be modified to your actual mask generation process)
            mask = np.ones(color.shape[:2], dtype=bool)  # Dummy mask, replace with actual mask if needed
            pose = est.register(K=reader.pv_intrinsics, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

            if debug >= 3:
                m = mesh.copy()
                m.apply_transform(pose)
                m.export(f'{debug_dir}/model_tf.obj')

                xyz_map = depth2xyzmap(depth, reader.pv_intrinsics)
                valid = depth >= 0.001
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
        else:
            # Track the object in subsequent frames
            pose = est.track_one(rgb=color, depth=depth, K=reader.pv_intrinsics, iteration=args.track_refine_iter)

        # Save the pose for debugging
        os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{debug_dir}/ob_in_cam/{frame_count}.txt', pose.reshape(4, 4))

        if debug >= 1:
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.pv_intrinsics, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.pv_intrinsics, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('Pose Estimation', vis[..., ::-1])
            cv2.waitKey(1)

        if debug >= 2:
            os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{debug_dir}/track_vis/{frame_count}.png', vis)

        frame_count += 1

except KeyboardInterrupt:
    logging.info("Process interrupted by user")

finally:
    # Cleanup resources
    reader.cleanup()
    cv2.destroyAllWindows()


