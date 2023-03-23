import os
import sys
from os.path import join as pjoin
from matplotlib.colors import rgb_to_hsv
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import math
import open3d as o3d


def draw_bbox3d_on_image(image, bboxes_3d, meta):
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    for category_id, bbox, inst_name in bboxes_3d:
        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        color = tuple(int(x) for x in cmap[category_id])
        for line in lines:
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 1
            linetype = 4
            cv2.line(image, start, end, color, thickness, linetype)
    return image


def draw_bbox3d_with_pose_color_on_image(image, bboxes_3d, meta):
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [
        cmap[0], cmap[2], cmap[4], cmap[6], cmap[8], cmap[10], cmap[12], cmap[16], cmap[14], cmap[14], cmap[14],
        cmap[14]
    ]

    for category_id, bbox, inst_name in bboxes_3d:
        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        for i, line in enumerate(lines):
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 1
            linetype = 4
            color = tuple(int(x) for x in colors[i])
            cv2.line(image, start, end, color, thickness, linetype)
    return image


# TODO to-finish
def draw_bbox3d_with_pose_frame_on_image(image, bboxes_3d, meta):
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)
    frame_axis_length = 50

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    for category_id, bbox, inst_name in bboxes_3d:
        bbox_center = np.mean(bbox, axis=0)
        x = np.array([1, 0, 0])

        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        color = tuple(int(x) for x in cmap[category_id])
        for line in lines:
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 1
            linetype = 4
            cv2.line(image, start, end, color, thickness, linetype)
    return image


def visu_depth_map(depth_map):
    # 最近的点置为0 最远的点置为255 后applyColorMap
    object_mask = (abs(depth_map) >= 1e-6)
    empty_mask = (abs(depth_map) < 1e-6)
    new_map = depth_map - depth_map[object_mask].min()
    new_map = new_map / new_map.max()
    new_map = np.clip(new_map * 255, 0, 255).astype('uint8')
    colored_depth_map = cv2.applyColorMap(new_map, cv2.COLORMAP_JET)
    colored_depth_map[empty_mask] = np.array([0, 0, 0])
    return colored_depth_map


def visu_2D_seg_map(seg_map):
    H, W = seg_map.shape
    seg_image = np.zeros((H, W, 3)).astype("uint8")

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")

    for y in range(0, H):
        for x in range(0, W):
            if seg_map[y, x] == -1:
                continue
            if seg_map[y, x] == 0:
                seg_image[y, x] = cmap[14]
            else:
                seg_image[y, x] = cmap[int(seg_map[y, x])]

    return seg_image


def visu_2D_bbox(rgb_image, bbox_list):
    image = np.copy(rgb_image)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")

    lines = [[0, 1], [1, 2], [2, 3], [0, 3]]

    for category_id, _, bbox_2d_coord, _, _ in bbox_list:
        y_min = bbox_2d_coord[0, 0]
        x_min = bbox_2d_coord[0, 1]
        y_max = bbox_2d_coord[1, 0]
        x_max = bbox_2d_coord[1, 1]
        corners = [(y_min, x_min), (y_max, x_min), (y_max, x_max), (y_min, x_max)]
        color = tuple(int(x) for x in cmap[category_id])
        for line in lines:
            x_start = corners[line[0]][1]
            y_start = corners[line[0]][0]
            x_end = corners[line[1]][1]
            y_end = corners[line[1]][0]
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 1
            linetype = 4
            cv2.line(image, start, end, color, thickness, linetype)

    return image


def visu_3D_bbox_semantic(rgb_image, bbox_list, meta):
    image = np.copy(rgb_image)

    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    for category_id, _, _, bbox, _ in bbox_list:
        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        color = tuple(int(x) for x in cmap[category_id])
        for line in lines:
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 1
            linetype = 4
            cv2.line(image, start, end, color, thickness, linetype)
    return image


def visu_3D_bbox_pose_in_color(rgb_image, bbox_list, meta):
    image = np.copy(rgb_image)

    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]
    colors = [
        cmap[0], cmap[2], cmap[4], cmap[6], cmap[8], cmap[10], cmap[12], cmap[16], cmap[14], cmap[14], cmap[14],
        cmap[14]
    ]

    for _, _, _, bbox, _ in bbox_list:
        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        for i, line in enumerate(lines):
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 1
            linetype = 4
            color = tuple(int(x) for x in colors[i])
            cv2.line(image, start, end, color, thickness, linetype)
    return image


# TODO to-finish
def visu_3D_bbox_pose_in_frame(rgb_image, bbox_list, meta):
    image = np.copy(rgb_image)

    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(meta['camera_intrinsic']).reshape(3, 3)
    frame_axis_length = 50

    cmap = plt.cm.get_cmap('tab20', 20)
    cmap = cmap.colors[:, 0:3]
    cmap = (cmap * 255).clip(0, 255).astype("uint8")
    lines = [[0, 1], [1, 2], [2, 3], [0, 3], [4, 5], [5, 6], [6, 7], [4, 7], [0, 4], [1, 5], [2, 6], [3, 7]]

    for category_id, _, _, bbox, _ in bbox_list:
        bbox_center = np.mean(bbox, axis=0)
        x = np.array([1, 0, 0])

        bbox_camera = (bbox - Rtilt_trl) @ Rtilt_rot
        color = tuple(int(x) for x in cmap[category_id])
        for line in lines:
            x_start = int(bbox_camera[line[0], 0] * K[0][0] / bbox_camera[line[0], 2] + K[0][2])
            y_start = int(bbox_camera[line[0], 1] * K[1][1] / bbox_camera[line[0], 2] + K[1][2])
            x_end = int(bbox_camera[line[1], 0] * K[0][0] / bbox_camera[line[1], 2] + K[0][2])
            y_end = int(bbox_camera[line[1], 1] * K[1][1] / bbox_camera[line[1], 2] + K[1][2])
            start = (x_start, y_start)
            end = (x_end, y_end)
            thickness = 1
            linetype = 4
            cv2.line(image, start, end, color, thickness, linetype)
    return image


def visu_NPCS_map_in_3D(npcs_map, ins_seg_map, anno_mapping, meta):
    width = meta['image_width']
    height = meta['image_height']

    NOCS_RTS_dict = {}
    for final_category_id, part_ins_cnt, bbox_3d_coord, inst_name_list in anno_mapping:
        bbox_3d, NOCS_params = bbox_3d_coord
        R, T, S = NOCS_params
        scaler = math.sqrt(S[0]**2 + S[1]**2 + S[2]**2)
        NOCS_RTS_dict[part_ins_cnt] = {'R': R, 'T': T, 'S': S, 'scaler': scaler}

    for part_ins_cnt in NOCS_RTS_dict.keys():
        mask = ins_seg_map == part_ins_cnt
        part_npcs = npcs_map[mask]
        part_world = (part_npcs * NOCS_RTS_dict[part_ins_cnt]['scaler']
                      ) @ NOCS_RTS_dict[part_ins_cnt]['R'] + NOCS_RTS_dict[part_ins_cnt]['T']

        pcd_1 = o3d.geometry.PointCloud()
        pcd_1.points = o3d.utility.Vector3dVector(part_world)
        pcd_2 = o3d.geometry.PointCloud()
        pcd_2.points = o3d.utility.Vector3dVector(part_npcs)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        o3d.visualization.draw_geometries([pcd_1, pcd_2, coord_frame])


def visu_NPCS_map(npcs_map, depth_map):
    height = npcs_map.shape[0]
    width = npcs_map.shape[1]

    npcs_image = npcs_map + np.array([0.5, 0.5, 0.5])
    assert (npcs_image > 0).all(), 'NPCS map error!'
    assert (npcs_image < 1).all(), 'NPCS map error!'
    empty_mask = (abs(depth_map) < 1e-6)
    npcs_image[empty_mask] = np.array([0, 0, 0])
    npcs_image = (np.clip(npcs_image, 0, 1) * 255).astype('uint8')

    return npcs_image
