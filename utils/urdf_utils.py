import os
from os.path import join as pjoin
import xml.etree.ElementTree as ET


# map link name to its object category name
def get_semantic_info(in_path, category):
    semantics_path = os.path.join(in_path, 'semantics_relabel.txt')
    linkName2catName = {}
    with open(semantics_path, 'r') as fd:
        linkName2catName = {}
        for line in fd:
            _id = int(line.rstrip().split(' ')[0].split('_')[-1]) + 1
            linkName2catName[_id] = line.rstrip().split(' ')[-2] + '_' + line.rstrip().split(' ')[-1]
            if category != "Bucket" and category != "Suitcase":
                if linkName2catName[_id] == 'hinge_handle':
                    linkName2catName[_id] = linkName2catName[_id] + '_ignore'
        linkName2catName[0] = 'base'
    link_name = [None] * len(linkName2catName.keys())
    for key in linkName2catName.keys():
        link_name[key] = linkName2catName[key]
    return link_name


# get all info from urdf
def get_urdf_mobility(inpath, verbose=False):
    if not inpath.endswith(".urdf"):
        urdf_name = inpath + "/mobility_relabel.urdf"
    else:
        urdf_name = inpath
        inpath = '/'.join(inpath.split('/')[:-1])

    urdf_ins = {}
    tree_urdf = ET.parse(urdf_name)
    num_real_links = len(tree_urdf.findall('link'))
    root_urdf = tree_urdf.getroot()
    rpy_xyz = {}
    list_xyz = [None] * num_real_links
    list_rpy = [None] * num_real_links
    list_obj = [None] * num_real_links
    # ['obj'] ['link/joint']['xyz/rpy'] [0, 1, 2, 3, 4]
    num_links = 0
    for link in root_urdf.iter('link'):
        num_links += 1
        if link.attrib['name'] == 'base':
            index_link = 0
        else:
            index_link = int(link.attrib['name'].split('_')[1]) + 1  # since the name is base, link_0, link_1
        list_xyz[index_link] = []
        list_rpy[index_link] = []
        list_obj[index_link] = []
        for visual in link.iter('visual'):
            for origin in visual.iter('origin'):
                if 'xyz' in origin.attrib:
                    list_xyz[index_link].append([float(x) for x in origin.attrib['xyz'].split()])
                else:
                    list_xyz[index_link].append([0, 0, 0])
                if 'rpy' in origin.attrib:
                    list_rpy[index_link].append([float(x) for x in origin.attrib['rpy'].split()])
                else:
                    list_rpy[index_link].append([0, 0, 0])
            for geometry in visual.iter('geometry'):
                for mesh in geometry.iter('mesh'):
                    if 'home' in mesh.attrib['filename'] or 'work' in mesh.attrib['filename']:
                        list_obj[index_link].append(mesh.attrib['filename'])
                    else:
                        list_obj[index_link].append(inpath + '/' + mesh.attrib['filename'])

    rpy_xyz['xyz'] = list_xyz
    rpy_xyz['rpy'] = list_rpy  # here it is empty list
    urdf_ins['link'] = rpy_xyz
    urdf_ins['obj_name'] = list_obj

    rpy_xyz = {}
    list_type = [None] * (num_real_links - 1)
    list_parent = [None] * (num_real_links - 1)
    list_child = [None] * (num_real_links - 1)
    list_xyz = [None] * (num_real_links - 1)
    list_rpy = [None] * (num_real_links - 1)
    list_axis = [None] * (num_real_links - 1)
    list_limit = [[0, 0]] * (num_real_links - 1)
    # here we still have to read the URDF file
    for joint in root_urdf.iter('joint'):
        """
        joint_index = int(joint.attrib['name'].split('_')[1])
        list_type[joint_index] = joint.attrib['type']
        """

        for child in joint.iter('child'):
            link_name = child.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            joint_index = link_index - 1  # !!! the joint's index should be the same as its child's index - 1
            list_child[joint_index] = link_index

        list_type[joint_index] = joint.attrib['type']

        for parent in joint.iter('parent'):
            link_name = parent.attrib['link']
            if link_name == 'base':
                link_index = 0
            else:
                link_index = int(link_name.split('_')[1]) + 1
            list_parent[joint_index] = link_index

        for origin in joint.iter('origin'):
            if 'xyz' in origin.attrib:
                list_xyz[joint_index] = [float(x) for x in origin.attrib['xyz'].split()]
            else:
                list_xyz[joint_index] = [0, 0, 0]
            if 'rpy' in origin.attrib:
                list_rpy[joint_index] = [float(x) for x in origin.attrib['rpy'].split()]
            else:
                list_rpy[joint_index] = [0, 0, 0]
        for axis in joint.iter('axis'):  # we must have
            list_axis[joint_index] = [float(x) for x in axis.attrib['xyz'].split()]
        for limit in joint.iter('limit'):
            list_limit[joint_index] = [float(limit.attrib['lower']), float(limit.attrib['upper'])]
        # 特殊处理continuous的上下限，和render_utils的处理保持一致
        if joint.attrib['type'] == 'continuous':
            list_limit[joint_index] = [-10000.0, 10000.0]

    rpy_xyz['type'] = list_type
    rpy_xyz['parent'] = list_parent
    rpy_xyz['child'] = list_child
    rpy_xyz['xyz'] = list_xyz
    rpy_xyz['rpy'] = list_rpy
    rpy_xyz['axis'] = list_axis
    rpy_xyz['limit'] = list_limit

    urdf_ins['joint'] = rpy_xyz
    urdf_ins['num_links'] = num_real_links
    if verbose:
        for j, pos in enumerate(urdf_ins['link']['xyz']):
            if len(pos) > 3:
                print('link {} xyz: '.format(j), pos[0])
            else:
                print('link {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['link']['rpy']):
            if len(orient) > 3:
                print('link {} rpy: '.format(j), orient[0])
            else:
                print('link {} rpy: '.format(j), orient)
        # for joint
        for j, pos in enumerate(urdf_ins['joint']['xyz']):
            print('joint {} xyz: '.format(j), pos)
        for j, orient in enumerate(urdf_ins['joint']['rpy']):
            print('joint {} rpy: '.format(j), orient)
        for j, orient in enumerate(urdf_ins['joint']['axis']):
            print('joint {} axis: '.format(j), orient)
        for j, child in enumerate(urdf_ins['joint']['child']):
            print('joint {} has child link: '.format(j), child)
        for j, parent in enumerate(urdf_ins['joint']['parent']):
            print('joint {} has parent link: '.format(j), parent)

    return urdf_ins


def get_fixed_handle_info(inpath, category):
    if not inpath.endswith(".urdf"):
        urdf_name = inpath + "/mobility_relabel.urdf"
    else:
        urdf_name = inpath
        inpath = '/'.join(inpath.split('/')[:-1])

    linkName2catName = get_semantic_info(inpath, category)

    urdf_ins = {}
    tree_urdf = ET.parse(urdf_name)
    num_real_links = len(tree_urdf.findall('link'))
    root_urdf = tree_urdf.getroot()

    list_handle_of_link = [None] * num_real_links

    for link in root_urdf.iter('link'):
        link_name = link.attrib['name']
        if link_name == 'base':
            index_link = 0
        else:
            index_link = int(link_name.split('_')[1]) + 1
        handle_dict = {}
        for visual in link.iter('visual'):
            visual_name = visual.attrib['name']
            if visual_name.find('handle') != -1 and linkName2catName[index_link].find('handle') == -1:
                inst_name = link_name + ':' + linkName2catName[index_link] + '/' + visual_name.split(
                    '-')[0] + ':' + 'fixed_handle'
                if inst_name not in handle_dict.keys():
                    handle_dict[inst_name] = []
                for geometry in visual.iter('geometry'):
                    for mesh in geometry.iter('mesh'):
                        if 'home' in mesh.attrib['filename'] or 'work' in mesh.attrib['filename']:
                            handle_dict[inst_name].append(mesh.attrib['filename'])
                        else:
                            handle_dict[inst_name].append(inpath + '/' + mesh.attrib['filename'])
        if handle_dict == {}:
            continue
        list_handle_of_link[index_link] = []
        for inst_name in handle_dict.keys():
            handle_obj = handle_dict[inst_name]
            list_handle_of_link[index_link].append((index_link, inst_name, handle_obj))

    return list_handle_of_link


def remove_fixed_handle_from_urdf(urdf_ins, list_handle_of_link):
    list_obj = urdf_ins['obj_name']
    num_real_links = urdf_ins['num_links']
    for link_id in range(num_real_links):
        if list_handle_of_link[link_id] == None:
            continue
        for handle in list_handle_of_link[link_id]:
            _, _, handle_obj = handle
            for obj in handle_obj:
                list_obj[link_id].remove(obj)
    urdf_ins['obj_name'] = list_obj
    urdf_ins['fixed_handle_ins'] = list_handle_of_link
    return urdf_ins


def add_fixed_handle_to_urdf(urdf_ins):
    fixed_handle_list = urdf_ins['fixed_handle_ins']
    list_obj = urdf_ins['obj_name']
    num_real_links = urdf_ins['num_links']
    for link_id in range(num_real_links):
        if fixed_handle_list[link_id] == None:
            continue
        for _, inst_name, objs in fixed_handle_list[link_id]:
            list_obj[link_id] += objs
    urdf_ins['obj_name'] = list_obj
    del urdf_ins['fixed_handle_ins']
    return urdf_ins


def check_urdf_ins_if_missing(urdf_ins_original, urdf_ins):
    num_real_links = urdf_ins['num_links']
    assert len(urdf_ins['obj_name']) == num_real_links
    if 'fixed_handle_ins' not in urdf_ins:
        for i in range(num_real_links):
            assert set(urdf_ins_original['obj_name'][i]) == set(urdf_ins['obj_name'][i])
    else:
        assert len(urdf_ins['fixed_handle_ins']) == num_real_links
        for i in range(num_real_links):
            handle_objs = []
            if urdf_ins['fixed_handle_ins'][i] is not None:
                for _, _, objs in urdf_ins['fixed_handle_ins'][i]:
                    handle_objs += objs
            assert set(urdf_ins_original['obj_name'][i]) == set(urdf_ins['obj_name'][i] + handle_objs)


def create_new_links_joints(urdf_ins):
    new_links = []
    new_joints = []
    modified_links = []
    name_mapping = {}
    if 'fixed_handle_ins' not in urdf_ins:
        return new_links, new_joints, modified_links, name_mapping
    
    fixed_handle_ins = urdf_ins['fixed_handle_ins']
    num_real_links = urdf_ins['num_links']
    num_new_links = 0
    for link_id in range(num_real_links):
        if fixed_handle_ins[link_id] is None:
            continue
        for _, handle_inst_name, handle_objs in fixed_handle_ins[link_id]:
            new_link_name = 'link_' + str(num_real_links + num_new_links - 1)
            link_visual_objs = []
            for obj in handle_objs:
                assert 'textured_objs' in obj
                obj_name = obj[obj.find('textured_objs'):]
                link_visual_objs.append(obj_name)
            new_links.append((new_link_name, 'handle', link_visual_objs))
            
            new_joint_name = 'joint_' + str(num_real_links + num_new_links - 1)
            new_joints.append((new_joint_name, new_link_name, handle_inst_name.split(':')[0], 'fixed'))
            
            modified_links.append((handle_inst_name.split(':')[0], link_visual_objs))
            
            name_mapping[handle_inst_name] = new_link_name
            
            num_new_links += 1
    
    return new_links, new_joints, modified_links, name_mapping


def create_link_visual_collision_joint_indexs(urdf_lines):
    link_prefix = '<link name="'
    link_end = '</link>'
    link_indexs = {}
    for idx_l, line in enumerate(urdf_lines):
        if link_prefix in line:
            left_idx = line.find(link_prefix) + len(link_prefix)
            right_idx = line.find('"', left_idx)
            link_name = line[left_idx:right_idx]
            start_idx = idx_l
            end_idx = None
            if '/' in line[right_idx:]:
                end_idx = start_idx + 1
            else:
                for idx_l2, line2 in enumerate(urdf_lines[start_idx + 1:]):
                    if link_end in line2:
                        end_idx = start_idx + 1 + idx_l2 + 1
                        break
            assert end_idx is not None
            link_indexs[link_name] = (start_idx, end_idx)
    
    visual_prefix = '<visual name="'
    visual_end = '</visual>'
    mesh_prefix = 'mesh filename="'
    visual_indexs = {}
    for idx_l, line in enumerate(urdf_lines):
        if visual_prefix in line:
            start_idx = idx_l
            mesh_name = None
            end_idx = None
            for idx_l2, line2 in enumerate(urdf_lines[start_idx + 1:]):
                if mesh_prefix in line2:
                    left_idx = line2.find(mesh_prefix) + len(mesh_prefix)
                    right_idx = line2.find('"', left_idx)
                    assert mesh_name is None
                    mesh_name = line2[left_idx:right_idx]
                if visual_end in line2:
                    end_idx = start_idx + 1 + idx_l2 + 1
                    break
            assert end_idx is not None
            visual_indexs[mesh_name] = (start_idx, end_idx)
    
    collision_prefix = '<collision>'
    collision_end = '</collision>'
    mesh_prefix = 'mesh filename="'
    collision_indexs = {}
    for idx_l, line in enumerate(urdf_lines):
        if collision_prefix in line:
            start_idx = idx_l
            mesh_name = None
            end_idx = None
            for idx_l2, line2 in enumerate(urdf_lines[start_idx + 1:]):
                if mesh_prefix in line2:
                    left_idx = line2.find(mesh_prefix) + len(mesh_prefix)
                    right_idx = line2.find('"', left_idx)
                    assert mesh_name is None
                    mesh_name = line2[left_idx:right_idx]
                if collision_end in line2:
                    end_idx = start_idx + 1 + idx_l2 + 1
                    break
            assert end_idx is not None
            collision_indexs[mesh_name] = (start_idx, end_idx)
    
    joint_prefix = '<joint name="'
    joint_end = '</joint>'
    joint_indexs = {}
    for idx_l, line in enumerate(urdf_lines):
        if joint_prefix in line:
            left_idx = line.find(joint_prefix) + len(joint_prefix)
            right_idx = line.find('"', left_idx)
            joint_name = line[left_idx:right_idx]
            start_idx = idx_l
            end_idx = None
            if '/' in line[right_idx:]:
                end_idx = start_idx + 1
            else:
                for idx_l2, line2 in enumerate(urdf_lines[start_idx + 1:]):
                    if joint_end in line2:
                        end_idx = start_idx + 1 + idx_l2 + 1
                        break
            assert end_idx is not None
            joint_indexs[joint_name] = (start_idx, end_idx)
    
    return link_indexs, visual_indexs, collision_indexs, joint_indexs
    

def create_visual_xyz(urdf_lines):
    visual_prefix = '<visual name="'
    visual_end = '</visual>'
    mesh_prefix = 'mesh filename="'
    origin_xyz_prefix = '<origin xyz="'
    visual_xyz = {}
    for idx_l, line in enumerate(urdf_lines):
        if visual_prefix in line:
            start_idx = idx_l
            origin_xyz = None
            mesh_name = None
            end_idx = None
            for idx_l2, line2 in enumerate(urdf_lines[start_idx + 1:]):
                if origin_xyz_prefix in line2:
                    assert 'rpy' not in line2
                    left_idx = line2.find(origin_xyz_prefix) + len(origin_xyz_prefix)
                    right_idx = line2.find('"', left_idx)
                    assert origin_xyz is None
                    origin_xyz = line2[left_idx:right_idx]
                if mesh_prefix in line2:
                    left_idx = line2.find(mesh_prefix) + len(mesh_prefix)
                    right_idx = line2.find('"', left_idx)
                    assert mesh_name is None
                    mesh_name = line2[left_idx:right_idx]
                if visual_end in line2:
                    end_idx = start_idx + 1 + idx_l2 + 1
                    break
            assert mesh_name is not None
            assert origin_xyz is not None
            assert end_idx is not None
            visual_xyz[mesh_name] = origin_xyz
    
    return visual_xyz


def modify_urdf_file_add_remove(inpath, new_links, new_joints, modified_links):
    if not inpath.endswith(".urdf"):
        urdf_name = inpath + "/mobility_relabel.urdf"
    else:
        urdf_name = inpath
        inpath = '/'.join(inpath.split('/')[:-1])
        
    with open(urdf_name, 'r') as f:
        urdf_lines = f.readlines()
    
    assert "</robot>" in urdf_lines[-1]
    obj_xyz = create_visual_xyz(urdf_lines)
    
    # remove objs from links
    for link_name, link_objs in modified_links:
        for obj_name in link_objs:
            # remove visual
            link_indexs, visual_indexs, collision_indexs, joint_indexs = create_link_visual_collision_joint_indexs(urdf_lines)
            assert obj_name in visual_indexs
            start_idx, end_idx = visual_indexs[obj_name]
            urdf_lines = urdf_lines[:start_idx] + urdf_lines[end_idx:]
            
            # remove collision
            link_indexs, visual_indexs, collision_indexs, joint_indexs = create_link_visual_collision_joint_indexs(urdf_lines)
            assert obj_name in collision_indexs
            start_idx, end_idx = collision_indexs[obj_name]
            urdf_lines = urdf_lines[:start_idx] + urdf_lines[end_idx:]
        
    # creating index for links, visuals, collisions and joints
    link_indexs, visual_indexs, collision_indexs, joint_indexs = create_link_visual_collision_joint_indexs(urdf_lines)
    assert len(new_links) == len(new_joints)
    if len(new_links) > 0:
        assert len(modified_links) > 0
    if len(modified_links) > 0:
        assert len(new_links) > 0
    for new_idx, (new_link, new_joint) in enumerate(zip(new_links, new_joints)):
        new_link_name, new_visual_name, new_link_objs = new_link
        new_joint_name, new_joint_child, new_joint_parent, new_joint_type = new_joint
        assert new_link_name == new_joint_child
        assert new_joint_name.split('_')[-1] == new_link_name.split('_')[-1]
        
        # add new link and new joint
        new_visual_name = new_visual_name + '-1'
        urdf_lines.insert(-1, '\t<link name="{}">\n'.format(new_link_name))
        for obj_name in new_link_objs:
            urdf_lines.insert(-1, '\t\t<visual name="{}">\n'.format(new_visual_name))
            urdf_lines.insert(-1, '\t\t\t<origin xyz="{}"/>\n'.format(obj_xyz[obj_name]))
            urdf_lines.insert(-1, '\t\t\t<geometry>\n')
            urdf_lines.insert(-1, '\t\t\t\t<mesh filename="{}"/>\n'.format(obj_name))
            urdf_lines.insert(-1, '\t\t\t</geometry>\n')
            urdf_lines.insert(-1, '\t\t</visual>\n')
        for obj_name in new_link_objs:
            urdf_lines.insert(-1, '\t\t<collision>\n')
            urdf_lines.insert(-1, '\t\t\t<origin xyz="{}"/>\n'.format(obj_xyz[obj_name]))
            urdf_lines.insert(-1, '\t\t\t<geometry>\n')
            urdf_lines.insert(-1, '\t\t\t\t<mesh filename="{}"/>\n'.format(obj_name))
            urdf_lines.insert(-1, '\t\t\t</geometry>\n')
            urdf_lines.insert(-1, '\t\t</collision>\n')
        urdf_lines.insert(-1, '\t</link>\n')
        
        urdf_lines.insert(-1, '\t<joint name="{}" type="{}">\n'.format(new_joint_name, new_joint_type))
        urdf_lines.insert(-1, '\t\t<origin rpy="0 0 0" xyz="0 0 0"/>\n')
        urdf_lines.insert(-1, '\t\t<child link="{}"/>\n'.format(new_joint_child))
        urdf_lines.insert(-1, '\t\t<parent link="{}"/>\n'.format(new_joint_parent))
        urdf_lines.insert(-1, '\t</joint>\n')
    
    # write new urdf file
    new_urdf_name = pjoin(inpath, 'mobility_relabel_gapartnet.urdf')
    if os.path.exists(new_urdf_name):
        os.remove(new_urdf_name)
    with open(new_urdf_name, 'w') as f:
        f.writelines(urdf_lines)
    
    return new_urdf_name


def modify_semantic_info(in_path, new_links):
    semantics_path = os.path.join(in_path, 'semantics_relabel.txt')
    semantics = []
    with open(semantics_path, 'r') as fd:
        for line in fd:
            semantics.append(line.strip().split(' '))
    
    new_link_names = [new_link[0] for new_link in new_links]
    new_link_names = sorted(new_link_names, key=lambda x: int(x.split('_')[-1]))
    for new_link_name in new_link_names:
        semantics.append([new_link_name, 'fixed', 'handle'])
    
    new_semantic_name = os.path.join(in_path, 'semantics_relabel_gapartnet.txt')
    with open(new_semantic_name, 'w') as fd:
        for line in semantics:
            fd.write(' '.join(line) + '\n')
    
    return new_semantic_name, [x[0] for x in semantics]


def create_link_annos(instname2catid, instname2newlink, mapping, all_link_names):
    annos = {x: {} for x in all_link_names}
    instname2linkname = {}
    
    for instname in instname2catid.keys():
        if instname == "others":
            continue
        
        if instname not in instname2newlink.keys():
            if '/' in instname:
                continue
            link_name = instname.split(':')[0]
            cat_name = instname.split(':')[1]
            assert not cat_name.endswith('_null')
            assert mapping[instname2catid[instname] - 1] == cat_name
            annos[link_name]['is_gapart'] = True
            annos[link_name]['category'] = cat_name
            annos[link_name]['bbox'] = None
            instname2linkname[instname] = link_name
            
        else:
            assert '/' in instname
            link_name = instname2newlink[instname]
            cat_name = instname.split(':')[-1]
            assert not cat_name.endswith('_null')
            assert cat_name == "fixed_handle"
            assert mapping[instname2catid[instname] - 1] == cat_name
            annos[link_name]['is_gapart'] = True
            annos[link_name]['category'] = cat_name
            annos[link_name]['bbox'] = None
            instname2linkname[instname] = link_name
    
    for link_name in all_link_names:
        if 'is_gapart' not in annos[link_name].keys():
            annos[link_name]['is_gapart'] = False
    
    return annos, instname2linkname


def fix_hinge_handle_joint_of_link_in_urdf_and_semantics(annotations, urdf_path, semantics_path):
    target_joints = []
    target_links = []
    all_link_names = [x for x in annotations.keys() if annotations[x]['is_gapart']]
    
    for link_name in all_link_names:
        category = annotations[link_name]['category']
        if category.endswith('fixed_handle'):
            target_joints.append('joint_{}'.format(link_name.split('_')[-1]))
            target_links.append(link_name)
    
    with open(urdf_path, 'r') as f:
        urdf_lines = f.readlines()
    
    for joint_name in target_joints:
        link_indexs, visual_indexs, collision_indexs, joint_indexs = create_link_visual_collision_joint_indexs(urdf_lines)
        start_idx, end_idx = joint_indexs[joint_name]
        joint_type = urdf_lines[start_idx].split('type="')[1].split('"')[0]
        urdf_lines[start_idx] = urdf_lines[start_idx].replace(joint_type, 'fixed')
        removed_idxs = []
        for idx in range(start_idx, end_idx):
            if "limit" in urdf_lines[idx]:
                removed_idxs.append(idx)
        print("fixed joint {} with {} removed lines".format(joint_name, len(removed_idxs)))
        urdf_lines = [x for idx, x in enumerate(urdf_lines) if idx not in removed_idxs]
    
    os.remove(urdf_path)
    with open(urdf_path, 'w') as f:
        f.writelines(urdf_lines)
    
    semantics = []
    with open(semantics_path, 'r') as fd:
        for line in fd:
            ls = line.strip().split(' ')
            if ls[0] in target_links:
                ls[1] = 'fixed'
                ls[2] = 'handle'
            semantics.append(ls)
    
    os.remove(semantics_path)
    with open(semantics_path, 'w') as fd:
        for line in semantics:
            fd.write(' '.join(line) + '\n')
    
    return urdf_path, semantics_path
    

if __name__ == '__main__':
    data_path = '/Users/proalec/Desktop/Part Detection/test/test_pose/example_data/'
    id_path = pjoin(data_path, '100051')
    print(get_semantic_info(id_path))