import os
import numpy as np
import sapien.core as sapien
import math
from itertools import groupby
import transforms3d.euler as t
import transforms3d.axangles as tax
import copy

from pose_utils import get_slider_button_pose, get_hinge_door_pose, get_slider_drawer_pose, get_hinge_lid_pose, get_slider_lid_pose, get_hinge_knob_pose, get_fixed_handle_pose, get_hinge_handle_pose
from urdf_utils import add_fixed_handle_to_urdf
from config_utils import CAMERA_ROTATE


# * 生成2D segmentation mask的函数（COCO格式）
def binary_mask_to_rle(binary_mask):
    '''from a numpy-array-like binary mask to RLE-form mask.
    written by helin, needed by annotating 2D segmentation masks.'''
    rle = {"counts": [], "size": list(binary_mask.shape)}
    counts = rle.get("counts")
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order="F"))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


# * 生成一个随机的camera position
def get_cam_pos(theta_min, theta_max, phi_min, phi_max, dis_min, dis_max):
    theta = np.random.uniform(low=theta_min, high=theta_max)
    phi = np.random.uniform(low=phi_min, high=phi_max)
    distance = np.random.uniform(low=dis_min, high=dis_max)
    x = math.sin(math.pi / 180 * theta) * math.cos(math.pi / 180 * phi) * distance
    y = math.sin(math.pi / 180 * theta) * math.sin(math.pi / 180 * phi) * distance
    z = math.cos(math.pi / 180 * theta) * distance
    return np.array([x, y, z])

def get_light_decay():
    return np.random.uniform(low=0.1,high=0.9)

def get_camera_rotate():
    return np.random.uniform(low=0-CAMERA_ROTATE,high=CAMERA_ROTATE)


def set_all_scene(data_path,
                  cam_pos,
                  width,
                  height,
                  model_id,
                  category,
                  engine,
                  use_raytracing=False,
                  light_decay=None,
                  camera_rotate=None,
                  joint_qpos=None,
                  meta=None,
                  de=None):
    if meta is None:
        meta = {}

    # set the sapien environment
    if engine is None:
        engine = sapien.Engine()
        if use_raytracing:
            config = sapien.KuafuConfig()
            config.spp = 256
            config.use_denoiser = True
            renderer = sapien.KuafuRenderer(config)
        else:
            renderer = sapien.VulkanRenderer(offscreen_only=True)
        engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    # load model
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    urdf_path = os.path.join(data_path, 'mobility_relabel.urdf')
    robot = loader.load_kinematic(urdf_path)
    assert robot, 'URDF not loaded.'

    # 设置joint的pose
    if joint_qpos is None:
        qlimits = robot.get_qlimits()
        qlimits[qlimits == np.inf] = 10000
        qlimits[qlimits == -np.inf] = -10000
        qlimits_l = np.concatenate([limits[..., 0].reshape(-1, 1) for limits in qlimits])
        qlimits_r = np.concatenate([limits[..., 1].reshape(-1, 1) for limits in qlimits])
        if de is None:
            delta = np.random.rand(*qlimits_l.shape)
        else:
            delta = np.random.rand(*qlimits_l.shape)
            delta = np.ones_like(delta) * de
        qpos = delta * qlimits_l + (1 - delta) * qlimits_r
    else:
        qpos = joint_qpos
    robot.set_qpos(qpos=qpos)

    # TODO: different in server and local (sapien version issue)
    if light_decay is None:
        scene.set_ambient_light([0.5, 0.5, 0.5])
    else:
        scene.set_ambient_light([1*light_decay,1*light_decay,1*light_decay])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)
    
    # rscene = scene.get_renderer_scene()
    # if light_decay is None:
    #     rscene.set_ambient_light([0.5, 0.5, 0.5])
    # else:
    #     rscene.set_ambient_light([1*light_decay,1*light_decay,1*light_decay])
    # rscene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    # rscene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    # rscene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    # rscene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera = scene.add_mounted_camera(
        name="camera",
        actor=camera_mount_actor,
        pose=sapien.Pose(),  # relative to the mounted actor
        width=width,
        height=height,
        fovx=np.deg2rad(35.0),
        fovy=np.deg2rad(35.0),
        near=0.1,
        far=100.0,
    )

    forward = -cam_pos / np.linalg.norm(cam_pos)
    left = np.cross([0, 0, 1], forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)

    if camera_rotate is not None:
        rad=camera_rotate/180*np.pi
        cam_rot_mat=tax.axangle2mat(forward.reshape(-1).tolist(),rad).T
        left=left@cam_rot_mat
        left = left / np.linalg.norm(left)
        up=up@cam_rot_mat
        up = up / np.linalg.norm(up)

    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

    scene.step()
    scene.update_render()
    camera.take_picture()

    # config metafile
    meta['category'] = category
    meta['model_id'] = model_id
    meta['joint_qpos'] = qpos.reshape(-1).tolist()

    if light_decay is None:
        meta['light_decay']=0.5
    else:
        meta['light_decay']=light_decay

    if camera_rotate is None:
        meta['camera_rotate']=0
    else:
        meta['camera_rotate']=camera_rotate

    meta['camera_position'] = cam_pos.reshape(-1).tolist()
    meta['image_width'] = width
    meta['image_height'] = height

    # 相机内参矩阵
    K = camera.get_camera_matrix()[:3, :3]
    meta['camera_intrinsic'] = K.reshape(-1).tolist()

    # 世界坐标系变换矩阵
    Rtilt = camera.get_model_matrix()
    Rtilt_rot = Rtilt[:3, :3] @ np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    Rtilt_trl = Rtilt[:3, 3]
    meta['world2camera_rotation'] = Rtilt_rot.reshape(-1).tolist()
    meta['camera2world_translation'] = Rtilt_trl.reshape(-1).tolist()

    return scene, camera, qpos, meta, engine


def render_rgb_image(camera):
    rgba = camera.get_float_texture('Color')
    rgb = rgba[:, :, :3]
    rgb_img = (rgb * 255).clip(0, 255).astype("uint8")
    return rgb_img


def render_depth_map(camera):
    position = camera.get_float_texture('Position')
    depth_map = -position[..., 2]
    return depth_map


def get_mapping_dict(scene, linkId2catName, target_parts_list):
    # map visual id to instance name
    visId2instName = {}
    # map instance name to category id(index+1, 0 for others)
    instName2catId = {}
    for articulation in scene.get_all_articulations():
        for link in articulation.get_links():
            link_name = link.get_name()
            if link_name == 'base':
                continue
            link_id = int(link_name.split('_')[-1]) + 1
            for visual in link.get_visual_bodies():
                visual_name = visual.get_name()
                if visual_name.find('handle') != -1 and linkId2catName[link_id].find('handle') == -1:
                    # visial name handle; link name not handle: fixed handle!
                    inst_name = link_name + ':' + linkId2catName[link_id] + '/' + visual_name.split(
                        '-')[0] + ':' + 'fixed_handle'
                    visual_id = visual.get_visual_id()
                    visId2instName[visual_id] = inst_name
                    if inst_name not in instName2catId.keys():
                        instName2catId[inst_name] = target_parts_list.index('fixed_handle') + 1
                elif linkId2catName[link_id] in target_parts_list:
                    inst_name = link_name + ':' + linkId2catName[link_id]
                    visual_id = visual.get_visual_id()
                    visId2instName[visual_id] = inst_name
                    if inst_name not in instName2catId.keys():
                        instName2catId[inst_name] = target_parts_list.index(linkId2catName[link_id]) + 1
                else:
                    inst_name = 'others'
                    visual_id = visual.get_visual_id()
                    visId2instName[visual_id] = inst_name
                    if inst_name not in instName2catId.keys():
                        instName2catId[inst_name] = 0
    return visId2instName, instName2catId


def get_annotation_2d_3d_old(camera, rgb_image, depth_image, meta, target_parts_list, visId2instName, instName2catId):
    seg_labels = camera.get_uint32_texture("Segmentation")
    seg_labels_by_visual_id = seg_labels[..., 0].astype(np.uint16)  # H x W, save each pixel's visual id

    category = meta['category']
    model_id = meta['model_id']
    render_cnt = meta['render_count']
    ins_index = meta['index']
    direct_code = meta['direction_code']
    state = meta['state']
    width = meta['image_width']
    height = meta['image_height']
    K = meta['camera_K']
    K = np.array(K).reshape(3, 3)
    Rtilt_rot = meta['Rtilt_rotation']
    Rtilt_rot = np.array(Rtilt_rot).reshape(3, 3)
    Rtilt_trl = meta['Rtilt_translation']
    Rtilt_trl = np.array(Rtilt_trl).reshape(1, 3)

    # * 2D：COCO format annotation框架
    id_image = int(model_id * 1000000 + render_cnt * 10000 + ins_index * 100 + direct_code)
    img = {
        "file_name": f'{state}_{category}_{model_id}_{render_cnt}_{ins_index}_{direct_code}.png',
        "height": height,
        "width": width,
        "id": id_image
    }
    jf = {
        "info": {},
        "license": [],
        "images": [],
        "annotations": [],
        "categories": ([{
            'id': 0,
            'name': 'others'
        }] + [{
            'id': x + 1,
            'name': target_parts_list[x]
        } for x in range(len(target_parts_list))]),
    }
    jf["images"].append(img)

    # * 分不同的part instance逐部分还原point cloud
    parts = []
    point_cloud = []
    per_point_rgb = []
    per_point_sem_label = []
    per_point_ins_label = []
    bounding_boxes = []
    bounding_boxes_world = []

    local_part_id = 1  # 标记当前part的全局编号，方便索引
    for inst_name in instName2catId.keys():

        # * get 2d mask for each instance
        mask = np.zeros((height, width))
        for vis_id in visId2instName.keys():
            if visId2instName[vis_id] != inst_name:
                continue
            # for all the vis id that belongs to this instance(not link! pay attention to the difference between link & instance: fixed handle is instance.)
            mask += (seg_labels_by_visual_id == vis_id)  # H*W binary mask.
        area = int(sum(sum(mask != 0)))
        if area == 0:
            continue

        # * 2D bbox
        y, x = (mask != 0).nonzero()  # for bbox detection
        # 类似sunrgbd的2d bbox标注，w=width-1(x_max-x_min，不加1),h同理
        w = int(max(x) - min(x))
        h = int(max(y) - min(y))
        xmin = int(min(x))
        ymin = int(min(y))

        # * 2D segmentation
        rle = binary_mask_to_rle(mask)

        # * save 2D annotation as COCO format
        category_id = instName2catId[inst_name]
        id_anno = id_image * 1000 + local_part_id  # need unique index for each mask in the whole dataset!
        anno = {
            "id": id_anno,
            "image_id": id_image,  # the index of image that this mask belongs to.
            "category_id": category_id,  # part category index.
            "segmentation": rle,
            "area": area,
            "bbox": [xmin, ymin, w, h],
            "iscrowd": 0,
        }
        jf["annotations"].append(anno)

        # * 2D over

        # * 3D begin
        # recover part instance point cloud
        part_pcs = []
        part_rgb = []
        for y_, x_ in zip(y, x):
            if depth_image[y_][x_] == 0:
                continue
            z_new = float(depth_image[y_][x_]) / 4000
            x_new = (x_ - K[0][2]) * z_new / K[0][0]
            y_new = (y_ - K[1][2]) * z_new / K[1][1]
            # point cloud N*3
            part_pcs.append([x_new, y_new, z_new])
            # rgb N*3
            part_rgb.append([
                float(rgb_image[y_][x_][0]) / 255,
                float(rgb_image[y_][x_][1]) / 255,
                float(rgb_image[y_][x_][2]) / 255
            ])

        # * 3D bbox in camera space
        # l,w,h correspond to dx,dy,dz
        # TODO 相比SUNRGBD格式调整保存顺序，wlh->lwh
        xcenter_3d = float(max(np.array(part_pcs)[:, 0]) + min(np.array(part_pcs)[:, 0])) / 2
        ycenter_3d = float(max(np.array(part_pcs)[:, 1]) + min(np.array(part_pcs)[:, 1])) / 2
        zcenter_3d = float(max(np.array(part_pcs)[:, 2]) + min(np.array(part_pcs)[:, 2])) / 2
        l_3d = float(max(np.array(part_pcs)[:, 0]) - min(np.array(part_pcs)[:, 0]))
        w_3d = float(max(np.array(part_pcs)[:, 1]) - min(np.array(part_pcs)[:, 1]))
        h_3d = float(max(np.array(part_pcs)[:, 2]) - min(np.array(part_pcs)[:, 2]))

        bbox = [category_id, xmin, ymin, w, h, xcenter_3d, ycenter_3d, zcenter_3d, l_3d, w_3d, h_3d]
        bounding_boxes.append(bbox)

        # * 3D bbox in world space(fake?)
        part_pcs_world = np.array(part_pcs) @ Rtilt_rot.T + Rtilt_trl
        xcenter_3d_world = float(max(part_pcs_world[:, 0]) + min(part_pcs_world[:, 0])) / 2
        ycenter_3d_world = float(max(part_pcs_world[:, 1]) + min(part_pcs_world[:, 1])) / 2
        zcenter_3d_world = float(max(part_pcs_world[:, 2]) + min(part_pcs_world[:, 2])) / 2
        l_3d_world = float(max(part_pcs_world[:, 0]) - min(part_pcs_world[:, 0]))
        w_3d_world = float(max(part_pcs_world[:, 1]) - min(part_pcs_world[:, 1]))
        h_3d_world = float(max(part_pcs_world[:, 2]) - min(part_pcs_world[:, 2]))

        bbox_world = [
            category_id, xmin, ymin, w, h, xcenter_3d_world, ycenter_3d_world, zcenter_3d_world, l_3d_world, w_3d_world,
            h_3d_world
        ]
        bounding_boxes_world.append(bbox_world)

        # * 3D segmentation
        # semantic segmentation
        part_sem_label = [category_id for _ in part_pcs]
        # instance segmentation
        if category_id != 0:
            part_ins_label = [local_part_id for _ in part_pcs]
        else:
            part_ins_label = [0 for _ in part_pcs]

        part_meta = {
            'name': inst_name,
            'category': 'others' if category_id == 0 else target_parts_list[category_id - 1],
            'category_id': category_id,
            'instance_id': 0 if category_id == 0 else local_part_id
        }
        parts.append(part_meta)
        meta['parts'] = parts

        point_cloud += part_pcs
        per_point_rgb += part_rgb
        per_point_sem_label += part_sem_label
        per_point_ins_label += part_ins_label

        if category_id != 0:
            local_part_id += 1

    ret_dict = {}
    ret_dict['point_cloud'] = np.array(point_cloud)
    ret_dict['per_point_rgb'] = np.array(per_point_rgb)
    ret_dict['bbox_camera'] = bounding_boxes
    ret_dict['bbox_world'] = bounding_boxes_world
    ret_dict['2d_annotation'] = jf
    ret_dict['3d_annotation'] = (np.array(per_point_sem_label), np.array(per_point_ins_label))
    ret_dict['meta'] = meta

    return ret_dict


def get_recovery_point_cloud_per_part(camera, rgb_image, depth_map, meta, visId2instName, instName2catId):
    seg_labels = camera.get_uint32_texture("Segmentation")
    seg_labels_by_visual_id = seg_labels[..., 0].astype(np.uint16)  # H x W, save each pixel's visual id

    width = meta['image_width']
    height = meta['image_height']
    K = meta['camera_intrinsic']
    K = np.array(K).reshape(3, 3)

    parts = []
    point_cloud = []
    per_point_rgb = []

    for inst_name in instName2catId.keys():
        mask = np.zeros((height, width))
        for vis_id in visId2instName.keys():
            if visId2instName[vis_id] != inst_name:
                continue
            # for all the vis id that belongs to this instance(not link! pay attention to the difference between link & instance: fixed handle is instance.)
            mask += (seg_labels_by_visual_id == vis_id)  # H*W binary mask.
        area = int(sum(sum(mask != 0)))
        if area == 0:
            continue
        y, x = (mask != 0).nonzero()
        # recover part instance point cloud
        part_pcs = []
        part_rgb = []
        for y_, x_ in zip(y, x):
            if abs(depth_map[y_][x_]) < 1e-6:
                continue
            z_new = float(depth_map[y_][x_])
            x_new = (x_ - K[0][2]) * z_new / K[0][0]
            y_new = (y_ - K[1][2]) * z_new / K[1][1]
            # point cloud N*3
            part_pcs.append([x_new, y_new, z_new])
            # rgb N*3
            part_rgb.append([
                float(rgb_image[y_][x_][0]) / 255,
                float(rgb_image[y_][x_][1]) / 255,
                float(rgb_image[y_][x_][2]) / 255
            ])
        category_id = instName2catId[inst_name]

        parts.append((inst_name, category_id, np.array(part_pcs), np.array(part_rgb)))
        point_cloud += part_pcs
        per_point_rgb += part_rgb

    return parts, np.array(point_cloud), np.array(per_point_rgb)


def get_recovery_point_cloud_whole(rgb_image, depth_map, meta):
    width = meta['image_width']
    height = meta['image_height']
    K = meta['camera_intrinsic']
    K = np.array(K).reshape(3, 3)

    point_cloud = []
    per_point_rgb = []

    for y_ in range(height):
        for x_ in range(width):
            if abs(depth_map[y_][x_]) < 1e-6:
                continue
            z_new = float(depth_map[y_][x_])
            x_new = (x_ - K[0][2]) * z_new / K[0][0]
            y_new = (y_ - K[1][2]) * z_new / K[1][1]
            point_cloud.append([x_new, y_new, z_new])
            per_point_rgb.append([
                float(rgb_image[y_][x_][0]) / 255,
                float(rgb_image[y_][x_][1]) / 255,
                float(rgb_image[y_][x_][2]) / 255
            ])
    return np.array(point_cloud), np.array(per_point_rgb)


def get_link_pose_dict(scene):
    links_pose_dict = {}
    for articulation in scene.get_all_articulations():
        for link in articulation.get_links():
            link_name = link.get_name()
            link_pose = link.get_pose()
            if link_name == 'base':
                link_id = 0
            else:
                link_id = int(link_name.split('_')[-1]) + 1
            links_pose_dict[link_id] = {'p': link_pose.p, 'q': link_pose.q}
    return links_pose_dict


def get_joint_pose_dict(link_pose_dict, urdf_ins):
    # get joint pose from link pose and link-joint relative relationship from urdf
    joint_info = urdf_ins['joint']
    joint_pose_dict = {}
    for i in range(urdf_ins['num_links'] - 1):
        if joint_info['type'][i] == 'fixed':  # skip the fixed joint
            continue
        child_id = joint_info['child'][i]
        child_p = link_pose_dict[child_id]['p']
        child_q = link_pose_dict[child_id]['q']
        joint_pose_xyz = [child_p[0], child_p[1], child_p[2]]
        joint_pose_mat = np.dot(
            t.quat2mat(child_q),
            t.euler2mat(joint_info['rpy'][i][0], joint_info['rpy'][i][1], joint_info['rpy'][i][2]).T)
        joint_pose_quat = t.euler2quat(*t.mat2euler(joint_pose_mat))
        axis_pose_vector = np.dot(np.array(joint_info['axis'][i]).reshape(1, 3),
                                  t.quat2mat(joint_pose_quat).T).reshape(3, ).tolist()
        joint_pose_dict[i] = {'xyz': joint_pose_xyz, 'axis': axis_pose_vector}

    return joint_pose_dict


def refine_qpos_meta(meta, urdf_ins):
    qpos = meta['joint_qpos']
    qpos_refined = [0] + qpos
    joint_info = urdf_ins['joint']

    class TreeNode(object):
        def __init__(self, link_id=None):
            super().__init__()
            self.id = link_id
            self.child = []
            self.joint = []

    treenode_dict = {}
    for i in range(urdf_ins['num_links'] - 1):
        cur_parent = joint_info['parent'][i]
        cur_child = joint_info['child'][i]
        if cur_parent not in treenode_dict.keys():
            treenode_dict[cur_parent] = TreeNode(cur_parent)
        if cur_child not in treenode_dict.keys():
            treenode_dict[cur_child] = TreeNode(cur_child)
        treenode_dict[cur_parent].child.append(cur_child)
        treenode_dict[cur_parent].joint.append(i)

    root = treenode_dict[0]
    joint_id_idx = []

    def walkTree(node):
        if node.child != []:
            for i in range(len(node.child)):
                joint_id_idx.append(node.joint[i])
                walkTree(treenode_dict[node.child[i]])

    walkTree(root)
    idx = np.argsort(joint_id_idx)
    qpos_new = [qpos_refined[i] for i in idx]
    meta_new=copy.deepcopy(meta)
    meta_new['joint_qpos'] = qpos_new

    return meta_new


def get_pose_bbox(part_list, target_parts_list, final_target_parts_list, scene, urdf_ins, link_pose_dict,
                  joint_pose_dict, fixed_handle_config, meta):

    object_category = meta['category']
    model_id = meta['model_id']

    fixed_handle_anno_flag = True

    # 三种不单独标注当前instance的handle的情况
    if object_category not in fixed_handle_config.keys():
        fixed_handle_anno_flag = False
    elif str(model_id) not in fixed_handle_config[object_category].keys():
        fixed_handle_anno_flag = False
    elif fixed_handle_config[object_category][str(model_id)] == -1:
        fixed_handle_anno_flag = False

    # 若不单独标注当前instance的fixed handle，则将handle的obj恢复到其link中
    # ! 由于获取bbox时是对obj的顶点取包围盒，故这样即可自动将handle的point cloud框进去，也就不再需要将handle的point cloud也恢复到其link中
    if not fixed_handle_anno_flag:
        urdf_ins = add_fixed_handle_to_urdf(urdf_ins)
    urdf_ins_output = copy.deepcopy(urdf_ins)

    bboxes = []
    for part_tuple in part_list:
        inst_name, category_id, part_pcs, part_rgb = part_tuple
        if category_id == 0:  # others
            continue
        category = target_parts_list[category_id - 1]
        print(category)
        if category == 'slider_button':
            bbox = get_slider_button_pose(inst_name=inst_name,
                                          pcs=part_pcs,
                                          scene=scene,
                                          urdf_ins=urdf_ins,
                                          link_pose_dict=link_pose_dict,
                                          joint_pose_dict=joint_pose_dict,
                                          meta=meta)
            final_category_id = final_target_parts_list.index(target_parts_list[category_id - 1]) + 1
            bboxes.append((final_category_id, bbox, inst_name))
        elif category == 'hinge_door':
            bbox = get_hinge_door_pose(inst_name=inst_name,
                                       pcs=part_pcs,
                                       scene=scene,
                                       urdf_ins=urdf_ins,
                                       link_pose_dict=link_pose_dict,
                                       joint_pose_dict=joint_pose_dict,
                                       meta=meta)
            final_category_id = final_target_parts_list.index(target_parts_list[category_id - 1]) + 1
            bboxes.append((final_category_id, bbox, inst_name))
        elif category == 'slider_drawer':
            bbox = get_slider_drawer_pose(inst_name=inst_name,
                                          pcs=part_pcs,
                                          scene=scene,
                                          urdf_ins=urdf_ins,
                                          link_pose_dict=link_pose_dict,
                                          joint_pose_dict=joint_pose_dict,
                                          meta=meta)
            final_category_id = final_target_parts_list.index(target_parts_list[category_id - 1]) + 1
            bboxes.append((final_category_id, bbox, inst_name))
        elif category == 'slider_lid':
            bbox = get_slider_lid_pose(inst_name=inst_name,
                                       pcs=part_pcs,
                                       scene=scene,
                                       urdf_ins=urdf_ins,
                                       link_pose_dict=link_pose_dict,
                                       joint_pose_dict=joint_pose_dict,
                                       meta=meta)
            final_category_id = final_target_parts_list.index(target_parts_list[category_id - 1]) + 1
            bboxes.append((final_category_id, bbox, inst_name))
        elif category == 'hinge_lid':
            bbox = get_hinge_lid_pose(inst_name=inst_name,
                                      pcs=part_pcs,
                                      scene=scene,
                                      urdf_ins=urdf_ins,
                                      link_pose_dict=link_pose_dict,
                                      joint_pose_dict=joint_pose_dict,
                                      meta=meta)
            final_category_id = final_target_parts_list.index(target_parts_list[category_id - 1]) + 1
            bboxes.append((final_category_id, bbox, inst_name))
        elif category == 'hinge_knob':
            bbox = get_hinge_knob_pose(inst_name=inst_name,
                                       pcs=part_pcs,
                                       scene=scene,
                                       urdf_ins=urdf_ins,
                                       link_pose_dict=link_pose_dict,
                                       joint_pose_dict=joint_pose_dict,
                                       meta=meta)
            final_category_id = final_target_parts_list.index(target_parts_list[category_id - 1]) + 1
            bboxes.append((final_category_id, bbox, inst_name))
        elif category == 'hinge_handle':
            bbox = get_hinge_handle_pose(inst_name=inst_name,
                                       pcs=part_pcs,
                                       scene=scene,
                                       urdf_ins=urdf_ins,
                                       link_pose_dict=link_pose_dict,
                                       joint_pose_dict=joint_pose_dict,
                                       meta=meta)
            final_category_id = final_target_parts_list.index(target_parts_list[category_id - 1]) + 1
            bboxes.append((final_category_id, bbox, inst_name))
        elif category == 'hinge_handleline':
            bbox = get_hinge_handle_pose(inst_name=inst_name,
                                       pcs=part_pcs,
                                       scene=scene,
                                       urdf_ins=urdf_ins,
                                       link_pose_dict=link_pose_dict,
                                       joint_pose_dict=joint_pose_dict,
                                       meta=meta)
            final_category_id = final_target_parts_list.index('line_fixed_handle') + 1
            bboxes.append((final_category_id, bbox, inst_name))
        elif category == 'hinge_handleround':
            bbox = get_hinge_handle_pose(inst_name=inst_name,
                                       pcs=part_pcs,
                                       scene=scene,
                                       urdf_ins=urdf_ins,
                                       link_pose_dict=link_pose_dict,
                                       joint_pose_dict=joint_pose_dict,
                                       meta=meta)
            final_category_id = final_target_parts_list.index('round_fixed_handle') + 1
            bboxes.append((final_category_id, bbox, inst_name))
        elif category == 'fixed_handle' and fixed_handle_anno_flag:
            bbox, tag = get_fixed_handle_pose(inst_name=inst_name,
                                              pcs=part_pcs,
                                              scene=scene,
                                              urdf_ins=urdf_ins,
                                              link_pose_dict=link_pose_dict,
                                              joint_pose_dict=joint_pose_dict,
                                              anno_config=fixed_handle_config,
                                              meta=meta)
            final_category_id = final_target_parts_list.index(tag + '_fixed_handle') + 1
            bboxes.append((final_category_id, bbox, inst_name))
    return bboxes, fixed_handle_anno_flag, urdf_ins_output


def add_background_color_for_image(rgb_image, depth_map, meta, background_rgb):
    width = meta['image_width']
    height = meta['image_height']
    for y_ in range(height):
        for x_ in range(width):
            if abs(depth_map[y_][x_]) < 1e-6:
                rgb_image[y_][x_] = background_rgb
    return rgb_image


def check_object_in_image(depth_map, meta):
    width = meta['image_width']
    height = meta['image_height']
    for i in range(height):
        if abs(depth_map[i][0]) >= 1e-6:
            return False
        if abs(depth_map[i][width - 1]) >= 1e-6:
            return False
    for i in range(width):
        if abs(depth_map[0][i]) >= 1e-6:
            return False
        if abs(depth_map[height - 1][i]) >= 1e-6:
            return False
    return True


def get_2D_segmentation(camera, depth_map, part_bbox_list, fixed_handle_anno_flag, meta, visId2instName,
                        instName2catId):
    # 通过merge rgb_data的depth_map 和 anno_data的visual_id 来确定在rgb_data上的segmentation
    seg_labels = camera.get_uint32_texture("Segmentation")
    seg_labels_by_visual_id = seg_labels[..., 0].astype(np.uint16)  # H x W, save each pixel's visual id

    width = meta['image_width']
    height = meta['image_height']

    sem_seg_map = np.zeros((height, width))  # -1表示empty，0表示others，从1开始计数
    ins_seg_map = np.zeros((height, width))  # -1表示empty，0表示others，从1开始计数

    mapping = []
    part_ins_cnt = 1
    for final_category_id, bbox, inst_name in part_bbox_list:
        inst_name_list = [inst_name]
        if not fixed_handle_anno_flag:
            for key in instName2catId.keys():
                if key.startswith(inst_name) and key != inst_name:  # 不标handle，则merge同一个link下的所有inst_name
                    inst_name_list.append(key)
        mask = np.zeros((height, width))
        for sub_inst in inst_name_list:
            for vis_id in visId2instName.keys():
                if visId2instName[vis_id] != sub_inst:
                    continue
                # for all the vis id that belongs to this instance(not link! pay attention to the difference between link & instance: fixed handle is instance.)
                mask += (seg_labels_by_visual_id == vis_id)  # H*W binary mask.
        sem_seg_map[mask > 0] = final_category_id
        ins_seg_map[mask > 0] = part_ins_cnt
        mapping.append((final_category_id, part_ins_cnt, bbox, inst_name_list))
        part_ins_cnt += 1

    empty_mask = abs(depth_map) < 1e-6
    sem_seg_map[empty_mask] = -1
    ins_seg_map[empty_mask] = -1

    return sem_seg_map, ins_seg_map, mapping


def get_2D_3D_bbox(sem_seg_map, ins_seg_map, anno_mapping):
    bbox_list = []
    for final_category_id, part_ins_cnt, bbox_3d_coord, inst_name_list in anno_mapping:
        bbox_3d, NOCS_params = bbox_3d_coord
        part_mask = (ins_seg_map == part_ins_cnt)
        y, x = part_mask.nonzero()  # for bbox detection
        bbox_2d_coord = np.array([[min(y) - 1, min(x) - 1], [max(y) + 1, max(x) + 1]])  # 边界检查保证了-1和+1不会出边界
        bbox_list.append((final_category_id, part_ins_cnt, bbox_2d_coord, bbox_3d, NOCS_params))
    return bbox_list


def get_NPCS_map(depth_map, ins_seg_map, anno_mapping, meta):
    width = meta['image_width']
    height = meta['image_height']
    K = meta['camera_intrinsic']
    Rtilt_rot = np.array(meta['world2camera_rotation']).reshape(3, 3)
    Rtilt_trl = np.array(meta['camera2world_translation']).reshape(1, 3)
    K = np.array(K).reshape(3, 3)

    NOCS_RTS_dict = {}
    for final_category_id, part_ins_cnt, bbox_3d_coord, inst_name_list in anno_mapping:
        bbox_3d, NOCS_params = bbox_3d_coord
        R, T, S = NOCS_params
        scaler = math.sqrt(S[0]**2 + S[1]**2 + S[2]**2)
        NOCS_RTS_dict[part_ins_cnt] = {'R': R, 'T': T, 'S': S, 'scaler': scaler}

    # print(NOCS_RTS_dict)

    world_position_map = np.zeros((height, width, 3))
    canon_position_map = np.zeros((height, width, 3))

    for y_ in range(height):
        for x_ in range(width):
            if abs(depth_map[y_][x_]) < 1e-6:
                continue
            z_new = float(depth_map[y_][x_])
            x_new = (x_ - K[0][2]) * z_new / K[0][0]
            y_new = (y_ - K[1][2]) * z_new / K[1][1]
            pixel_camera_position = np.array([x_new, y_new, z_new])
            pixel_world_position = pixel_camera_position @ Rtilt_rot.T + Rtilt_trl
            world_position_map[y_][x_] = pixel_world_position
            pixel_ins_cnt = int(ins_seg_map[y_][x_])
            if pixel_ins_cnt == 0:  # others
                continue
            pixel_nocs_position = (pixel_world_position - NOCS_RTS_dict[pixel_ins_cnt]['T']) @ (
                NOCS_RTS_dict[pixel_ins_cnt]['R'].T) / NOCS_RTS_dict[pixel_ins_cnt]['scaler']
            canon_position_map[y_][x_] = pixel_nocs_position

    # for part_ins_cnt in NOCS_RTS_dict.keys():
    #     mask = ins_seg_map == part_ins_cnt
    #     part_npcs_map = canon_position_map[mask]
    #     print(np.min(part_npcs_map))

    return canon_position_map


def get_final_anno_bbox_pose(bbox_list, target_part_list):
    anno_list = []
    for final_category_id, part_ins_cnt, bbox_2d_coord, bbox_3d_coord, NOCS_params in bbox_list:
        part_anno = {}
        part_anno['category'] = target_part_list[final_category_id - 1]
        part_anno['category_id'] = final_category_id
        part_anno['instance_id'] = part_ins_cnt
        part_anno['bbox_2d'] = bbox_2d_coord
        part_anno['bbox_3d'] = bbox_3d_coord
        part_anno['NPCS_RTS'] = NOCS_params

        # 添加pose frame gt
        pose_x = np.array([1, 0, 0]) @ NOCS_params[0]
        pose_x = pose_x / np.linalg.norm(pose_x)
        pose_y = np.array([0, 1, 0]) @ NOCS_params[0]
        pose_y = pose_y / np.linalg.norm(pose_y)
        pose_z = np.array([0, 0, 1]) @ NOCS_params[0]
        pose_z = pose_z / np.linalg.norm(pose_z)
        part_anno['pose_frame'] = (pose_x, pose_y, pose_z)

        anno_list.append(part_anno)

    return anno_list


def recover_pose_bbox_to_static_pose(part_bbox_list, urdf_ins, joint_pose_dict, target_parts_list, meta):
    output_part_bbox_list = []
    
    for part_tuple in part_bbox_list:
        category = target_parts_list[part_tuple[0] - 1] # final category in annotation
        bbox = part_tuple[1][0].copy()
        inst_name = part_tuple[2]
        print('recovering', category, inst_name)
        
        joint_info = urdf_ins['joint']
        link_name = inst_name.split(':')[0]
        if link_name == 'base':
            link_id = 0
        else:
            link_id = int(link_name.split('_')[-1]) + 1

        joint_id_list = []
        last_link_id = link_id
        end_flag = False
        while (not end_flag):
            for i in range(urdf_ins['num_links'] - 1):
                if joint_info['child'][
                        i] == last_link_id:  # !!! the joint's index should be the same as its child's index - 1
                    joint_id_list.append(i)
                    if joint_info['parent'][i] == 0:
                        end_flag = True
                        break
                    else:
                        last_link_id = joint_info['parent'][i]
                        break
        print('joint_id_list', joint_id_list)
        joint_id_list = joint_id_list[:-1]
        
        joint_qpos_list = np.array(meta['joint_qpos']).reshape(-1, 1)
        for joint_id in joint_id_list:
            joint_qpos = joint_qpos_list[joint_id]
            axis_start_point = np.array(joint_pose_dict[joint_id]['xyz']).reshape(1, 3)
            axis_direction_vector = np.array(joint_pose_dict[joint_id]['axis']).reshape(1, 3)
            axis_direction_vector = axis_direction_vector / np.linalg.norm(axis_direction_vector)
            joint_type = joint_info['type'][joint_id]
            
            if joint_type == 'prismatic':
                bbox = bbox - axis_direction_vector * joint_qpos
                continue
                
            elif joint_type == 'revolute':
                rotation_mat = t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), joint_qpos * (-1)).T
                bbox = np.dot(bbox - axis_start_point, rotation_mat) + axis_start_point
                continue
            
            elif joint_type == 'continuous':
                rotation_mat = t.axangle2mat(axis_direction_vector.reshape(-1).tolist(), joint_qpos * (-1)).T
                bbox = np.dot(bbox - axis_start_point, rotation_mat) + axis_start_point
                continue
            
            elif joint_type == 'fixed':
                continue
            
            else:
                raise ValueError('joint type error: ', joint_type)
        
        output_part_bbox_list.append((category, bbox, inst_name))
    
    return output_part_bbox_list


def merge_bbox_into_annotation(link_annos, part_bbox_list, name_mapping):
    for cat_name, bbox, inst_name in part_bbox_list:
        link_name = name_mapping[inst_name]
        assert link_annos[link_name]['is_gapart'] == True
        
        link_annos[link_name]['bbox'] = bbox.copy()
        
        assert link_annos[link_name]['category'] != 'hinge_handle_null'
        assert cat_name != 'hinge_handle_null'
        if link_annos[link_name]['category'] == 'fixed_handle':
            assert cat_name == 'line_fixed_handle' or cat_name == 'round_fixed_handle'
        elif link_annos[link_name]['category'] == 'hinge_handleline':
            assert cat_name == 'line_fixed_handle'
        elif link_annos[link_name]['category'] == 'hinge_handleround':
            assert cat_name == 'round_fixed_handle'
        else:
            assert link_annos[link_name]['category'] == cat_name
        
        link_annos[link_name]['category'] = cat_name
    
    return link_annos


def merge_link_annotation(all_link_annos, curr_link_annos):
    link_names = list(all_link_annos.keys())
    
    for _name in link_names:
        if all_link_annos[_name]['is_gapart'] == False:
            assert curr_link_annos[_name]['is_gapart'] == False
            continue
        
        if all_link_annos[_name]['bbox'] is not None:
            if curr_link_annos[_name]['bbox'] is not None:
                assert all_link_annos[_name]['category'] == curr_link_annos[_name]['category']
            # TODO: maybe need to check the bbox here (need to tolerate some error)
            continue
        
        if curr_link_annos[_name]['bbox'] is not None:
            all_link_annos[_name]['bbox'] = curr_link_annos[_name]['bbox'].copy()
            all_link_annos[_name]['category'] = curr_link_annos[_name]['category']
            continue
        
    return all_link_annos


def check_annotation_all_filled(all_link_annos, target_name_list):
    link_names = list(all_link_annos.keys())
    
    finished = True
    for _name in link_names:
        if all_link_annos[_name]['is_gapart'] == False:
            continue
        
        if all_link_annos[_name]['bbox'] is None:
            finished = False
            break
        
        assert all_link_annos[_name]['category'] in target_name_list
        assert all_link_annos[_name]['category'] != 'hinge_handle_null'
    
    return finished


def correct_hinge_knob_orientation(link_annos, urdf_file):
    # Use heuristic to correct the orientation of hinge knob
    
    link_names = list(link_annos.keys())
    
    for _name in link_names:
        if link_annos[_name]['is_gapart'] == False:
            continue
        
        if link_annos[_name]['category'] != 'hinge_knob':
            continue
        
        bbox = link_annos[_name]['bbox']
        zs = bbox[:4, :] - bbox[4:, :]
        z_axis = np.mean(zs, axis=0)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        if abs(z_axis[2]) > 0.1:
            if z_axis[2] < 0:
                bbox = bbox[[6, 7, 4, 5, 2, 3, 0, 1], :] # flip the bbox
        else:
            center = np.mean(bbox, axis=0)
            center[2] = 0 # project to the xy plane
            center = center / np.linalg.norm(center)
            z_axis[2] = 0 # project to the xy plane
            z_axis = z_axis / np.linalg.norm(z_axis)
            if np.dot(center, z_axis) < 0:
                bbox = bbox[[6, 7, 4, 5, 2, 3, 0, 1], :] # flip the bbox
        
        link_annos[_name]['bbox'] = bbox
        
    return link_annos

