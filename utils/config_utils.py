import os
import sys
from os.path import join as pjoin
import json
from PIL import Image
import numpy as np
import pickle


# TODO to modify!
DATASET_PATH = '/data/chengyang/GAPartNet_src'
SAVE_PATH = '/data/chengyang/GAPartNet_modified_test'
NUM_RENDER = 32


FIXED_HANDLE_STRATEGY = './configs/fixed_handle.json'
ID_PATH = './configs/partnet_part1_id_split.txt'
# ID_PATH = './configs/partnet_part2_id_split.txt'
RENDERED_DATA_PATH = '/data/chengyang/data/source_data_2/valid'
# RENDERED_DATA_PATH = '/data/chengyang/data/new_render/partnet'

TARGET_PARTS_FIRST_STAGE = [
    'fixed_handle', 'hinge_handle_null', 'slider_button', 'hinge_door', 'slider_drawer', 'slider_lid', 'hinge_lid',
    'hinge_knob', 'hinge_handle', 'hinge_handleline', 'hinge_handleround'
]

TARGET_PARTS_SECOND_STAGE = [
    'line_fixed_handle', 'round_fixed_handle', 'hinge_handle_null', 'slider_button', 'hinge_door', 'slider_drawer',
    'slider_lid', 'hinge_lid', 'hinge_knob', 'hinge_handle'
]

OBJECT_CATEGORIES = [
    'Box', 'Camera', 'CoffeeMachine', 'Dishwasher', 'KitchenPot', 'Microwave', 'Oven', 'Phone', 'Refrigerator',
    'Remote', 'Safe', 'StorageFurniture', 'Table', 'Toaster', 'TrashCan', 'WashingMachine', 'Keyboard', 'Laptop', 'Door', 'Printer',
    'Suitcase', 'Bucket', 'Toilet'
]

CAMERA_POSITION_RANGE = {
    'Box': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.5
    }],
    'Camera': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.5
    }, {
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': -60.0,
        'phi_max': 60.0,
        'distance_min': 3.5,
        'distance_max': 4.5
    }],
    'CoffeeMachine': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Dishwasher': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'KitchenPot': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Microwave': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Oven': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Phone': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Refrigerator': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Remote': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Safe': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'StorageFurniture': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 4.1,
        'distance_max': 5.2
    }],
    'Table': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.8,
        'distance_max': 4.5
    }],
    'Toaster': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'TrashCan': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 4,
        'distance_max': 5.5
    }],
    'WashingMachine': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Keyboard': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3,
        'distance_max': 3.5
    }],
    'Laptop': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Door': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Printer': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Suitcase': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Bucket': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }],
    'Toilet': [{
        'theta_min': 30.0,
        'theta_max': 80.0,
        'phi_min': 120.0,
        'phi_max': 240.0,
        'distance_min': 3.5,
        'distance_max': 4.1
    }]
}

BACKGROUND_RGB = np.array([216, 206, 189], dtype=np.uint8)
CAMERA_ROTATE = 5
RERENDER_MAX = 10

HEIGHT = 800
WIDTH = 800


def get_fixed_handle_configs(inpath):
    with open(inpath, 'r') as fd:
        config_dict = json.load(fd)
    return config_dict


# 获取当前model id对应的label，original or modified
def get_id_label(target_id):
    label = None
    with open(ID_PATH, 'r') as fd:
        for line in fd:
            category = line.rstrip('\n').split(' ')[0]
            id = int(line.rstrip('\n').split(' ')[1])
            state = line.rstrip('\n').split(' ')[2]
            if id == target_id:
                label = state
                break
    if label != None:
        return label
    else:
        print('Error! Could not get label of id {}'.format(target_id))
        exit(-1)


def save_rgb_image(rgb_img, save_path, filename):
    new_image = Image.fromarray(rgb_img)
    new_image.save(pjoin(save_path, f'{filename}.png'))


def save_depth_map(depth_map, save_path, filename):
    # np.save(pjoin(save_path, f'{filename}.npy'), depth_map)

    np.savez_compressed(pjoin(save_path, f'{filename}.npz'), depth_map=depth_map)


def save_anno_dict(anno_dict, save_path, filename):
    # with open(pjoin(save_path, f'{filename}.pkl'), 'wb') as fd:
    #     pickle.dump(anno_dict, fd)

    seg_path = pjoin(save_path, 'segmentation')
    bbox_path = pjoin(save_path, 'bbox')
    npcs_path = pjoin(save_path, 'npcs')

    if not os.path.exists(seg_path): os.mkdir(seg_path)
    if not os.path.exists(bbox_path): os.mkdir(bbox_path)
    if not os.path.exists(npcs_path): os.mkdir(npcs_path)

    np.savez_compressed(pjoin(seg_path, f'{filename}.npz'),
                        semantic_segmentation=anno_dict['semantic_segmentation'],
                        instance_segmentation=anno_dict['instance_segmentation'])

    np.savez_compressed(pjoin(npcs_path, f'{filename}.npz'), npcs_map=anno_dict['npcs_map'])

    with open(pjoin(bbox_path, f'{filename}.pkl'), 'wb') as fd:
        bbox_dict = {'bboxes_with_pose': anno_dict['bboxes_with_pose']}
        pickle.dump(bbox_dict, fd)


def save_meta(meta, save_path, filename):
    with open(pjoin(save_path, f'{filename}.json'), 'w') as fd:
        json.dump(meta, fd)


def load_rgb_image(save_path, filename):
    img = Image.open(pjoin(save_path, f'{filename}.png'))
    return np.array(img)


def load_depth_map(save_path, filename):
    # depth_map = np.load(pjoin(save_path, f'{filename}.npy'))

    depth_dict = np.load(pjoin(save_path, f'{filename}.npz'))
    depth_map = depth_dict['depth_map']
    return depth_map


def load_anno_dict(save_path, filename):
    # with open(pjoin(save_path, f'{filename}.pkl'), 'rb') as fd:
    #     anno_dict = pickle.load(fd)

    anno_dict = {}
    seg_path = pjoin(save_path, 'segmentation')
    bbox_path = pjoin(save_path, 'bbox')
    npcs_path = pjoin(save_path, 'npcs')

    seg_dict = np.load(pjoin(seg_path, f'{filename}.npz'))
    anno_dict['semantic_segmentation'] = seg_dict['semantic_segmentation']
    anno_dict['instance_segmentation'] = seg_dict['instance_segmentation']

    npcs_dict = np.load(pjoin(npcs_path, f'{filename}.npz'))
    anno_dict['npcs_map'] = npcs_dict['npcs_map']

    with open(pjoin(bbox_path, f'{filename}.pkl'), 'rb') as fd:
        bbox_dict = pickle.load(fd)
    anno_dict['bboxes_with_pose'] = bbox_dict['bboxes_with_pose']

    return anno_dict


def load_meta(save_path, filename):
    with open(pjoin(save_path, f'{filename}.json'), 'r') as fd:
        meta = json.load(fd)
    return meta
