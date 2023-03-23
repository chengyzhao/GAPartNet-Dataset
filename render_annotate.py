import os
import sys
from os.path import join as pjoin
import numpy as np
from argparse import ArgumentParser
import time
import shutil

sys.path.append('./utils')
from utils.render_utils import get_cam_pos,get_light_decay,get_camera_rotate, set_all_scene, render_rgb_image, render_depth_map, get_mapping_dict, get_recovery_point_cloud_per_part, get_pose_bbox, get_link_pose_dict, get_joint_pose_dict, refine_qpos_meta, add_background_color_for_image, check_object_in_image, get_2D_segmentation, get_2D_3D_bbox, get_NPCS_map, get_final_anno_bbox_pose, \
    recover_pose_bbox_to_static_pose
from utils.urdf_utils import get_semantic_info, get_urdf_mobility, get_fixed_handle_info, remove_fixed_handle_from_urdf, \
    check_urdf_ins_if_missing, create_new_links_joints, modify_urdf_file_add_remove, modify_semantic_info, create_link_annos
from utils.config_utils import CAMERA_ROTATE, get_fixed_handle_configs, get_id_label, save_rgb_image, save_depth_map, save_anno_dict, save_meta, DATASET_PATH, FIXED_HANDLE_STRATEGY, SAVE_PATH, TARGET_PARTS_FIRST_STAGE, TARGET_PARTS_SECOND_STAGE, BACKGROUND_RGB, RERENDER_MAX, RENDERED_DATA_PATH, \
    load_meta, load_anno_dict, NUM_RENDER, RENDERED_DATA_PATH, CAMERA_POSITION_RANGE

if __name__ == "__main__":
    # 加载配置参数
    parser = ArgumentParser()
    parser.add_argument('--model_id', type=int)
    parser.add_argument('--category', type=str)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)

    CONFS = parser.parse_args()

    MODEL_ID = CONFS.model_id
    CATEGORY = CONFS.category
    WIDTH = CONFS.width
    HEIGHT = CONFS.height
    
    print(f'------> Current Run: {CATEGORY} : {MODEL_ID}')
    
    # 配置路径
    model_label = get_id_label(MODEL_ID)
    assert model_label in ['original', 'modified'], f'Error! model_label {model_label} is not in [original, modified]'
    anno_data_path = pjoin(DATASET_PATH, model_label, str(MODEL_ID))
    
    # copy data to save path
    save_path = SAVE_PATH
    anno_new_path = pjoin(save_path, str(MODEL_ID))
    if os.path.exists(anno_new_path):
        shutil.rmtree(anno_new_path)
    shutil.copytree(anno_data_path, anno_new_path)
    anno_data_path = anno_new_path

    # load rendered metas and annos
    num_camera_position = len(CAMERA_POSITION_RANGE[CATEGORY])
    num_render = NUM_RENDER
    # anno_history = {}
    meta_history = {}
    for i_cam in range(num_camera_position):
        # anno_history[i_cam] = {}
        meta_history[i_cam] = {}
        for i_render in range(num_render):
            filename = '{}_{}_{:02d}_{:03d}'.format(CATEGORY, MODEL_ID, i_cam, i_render)
            metafile = load_meta(pjoin(RENDERED_DATA_PATH, 'metafile'), filename)
            meta_history[i_cam][i_render] = metafile
            # anno_dict = load_anno_dict(pjoin(RENDERED_DATA_PATH, 'annotation'), filename)
    
    print('Load history metafile and annotation done!')

    engine = None
    
    for i_cam in range(num_camera_position):
        for i_render in range(num_render):
            print(f'------> Current Run: {CATEGORY} : {MODEL_ID} : {i_cam} : {i_render}')
            
            meta_his = meta_history[i_cam][i_render]
            cam_pos = np.array(meta_his['camera_position'], dtype=np.float32).reshape(-1)
            
            scene, camera, joint_qpos, metafile_raw, engine = set_all_scene(data_path=anno_data_path,
                                                                    cam_pos=cam_pos,
                                                                    width=WIDTH,
                                                                    height=HEIGHT,
                                                                    model_id=MODEL_ID,
                                                                    category=CATEGORY,
                                                                    engine=engine)
            rgb_image = render_rgb_image(camera=camera)
            depth_map = render_depth_map(camera=camera)

            final_rgb_image = rgb_image
            final_depth_map = depth_map

            valid_flag = check_object_in_image(depth_map, metafile_raw)
            if not valid_flag:
                print(
                    f'Error! Fail in boundary check stage in rendering {CATEGORY} : {MODEL_ID} : {i_cam} : {i_render}')
                exit(-1)
            
            link_name_list = get_semantic_info(anno_data_path)
            urdf_ins = get_urdf_mobility(anno_data_path)
            fixed_handle_ins = get_fixed_handle_info(anno_data_path)
            urdf_ins = remove_fixed_handle_from_urdf(urdf_ins, fixed_handle_ins)
            fixed_handle_render_configs = get_fixed_handle_configs(FIXED_HANDLE_STRATEGY)
            metafile = refine_qpos_meta(meta=metafile_raw, urdf_ins=urdf_ins)
            link_pose_dict = get_link_pose_dict(scene=scene)
            joint_pose_dict = get_joint_pose_dict(link_pose_dict=link_pose_dict, urdf_ins=urdf_ins)
            visId2instName, instName2catId = get_mapping_dict(scene=scene,
                                                            linkId2catName=link_name_list,
                                                            target_parts_list=TARGET_PARTS_FIRST_STAGE)
            part_list, point_cloud, per_point_rgb = get_recovery_point_cloud_per_part(camera=camera,
                                                                                    rgb_image=rgb_image,
                                                                                    depth_map=depth_map,
                                                                                    meta=metafile,
                                                                                    visId2instName=visId2instName,
                                                                                    instName2catId=instName2catId)
            part_bbox_list, fixed_handle_anno_flag, urdf_ins = get_pose_bbox(part_list=part_list,
                                                                target_parts_list=TARGET_PARTS_FIRST_STAGE,
                                                                final_target_parts_list=TARGET_PARTS_SECOND_STAGE,
                                                                scene=scene,
                                                                urdf_ins=urdf_ins,
                                                                link_pose_dict=link_pose_dict,
                                                                joint_pose_dict=joint_pose_dict,
                                                                fixed_handle_config=fixed_handle_render_configs,
                                                                meta=metafile)
            
            # 1. modify urdf file
            urdf_ins_original = get_urdf_mobility(anno_data_path)
            check_urdf_ins_if_missing(urdf_ins_original, urdf_ins)
            new_links, new_joints, modified_links, instname2newlinkname = create_new_links_joints(urdf_ins)
            p_new_urdf_file = modify_urdf_file_add_remove(anno_data_path, new_links, new_joints, modified_links)
            print("Save new urdf file done: ", p_new_urdf_file)
            
            # 2. create semantic label
            p_new_semantic_file, all_link_names = modify_semantic_info(anno_data_path, new_links)
            print("Save new semantic file done: ", p_new_semantic_file)
            
            # 3. create gapart semantic and pose annotation
            # 3.1 init gapart semantic annotation w/o pose
            link_annotations, instname2linkname = create_link_annos(instName2catId, instname2newlinkname, TARGET_PARTS_FIRST_STAGE, all_link_names)
            print(f'link_annotations: {link_annotations}')
            # 3.2 add pose to gapart annotation
            static_part_pose_bbox_list = recover_pose_bbox_to_static_pose(part_bbox_list, urdf_ins, joint_pose_dict, TARGET_PARTS_SECOND_STAGE, metafile)
            print(f'static_part_pose_bbox_list: {static_part_pose_bbox_list}')
            
            