import os
import os.path as osp
import sys
import json
import shutil


dataset_path_part1 = "/data/chengyang/data/chengyang/data/dataset_v1.0"
dataset_path_part2 = "/data/chengyang/data/chengyang/data/partnet"
dataset_path_before_all = "/data/chengyang/data/chengyang/partnet_all_src"

split_txt_part1 = "/data/chengyang/GAPartNet/GAPartNet-Dataset/configs/partnet_part1_id_split.txt"
split_txt_part2 = "/data/chengyang/GAPartNet/GAPartNet-Dataset/configs/partnet_part2_id_split.txt"
split_txt_before_all = "/data/chengyang/GAPartNet/GAPartNet-Dataset/configs/partnet_all_id_split.txt"

if __name__ == "__main__":
    
    dataset_path_dict = {
        1: dataset_path_part1,
        2: dataset_path_part2
    }
    
    all_data_split = []
    
    with open(split_txt_part1, "r") as f:
        for line in f:
            ls = line.strip().split(' ')
            cat = ls[0]
            model_id = ls[1]
            tag = ls[2]
            all_data_split.append((1, cat, model_id, tag))
        
    with open(split_txt_part2, "r") as f:
        for line in f:
            ls = line.strip().split(' ')
            cat = ls[0]
            model_id = ls[1]
            tag = ls[2]
            all_data_split.append((2, cat, model_id, tag))
            
    for _i, data_split in enumerate(all_data_split):
        _dataset_path = dataset_path_dict[data_split[0]]
        _cat = data_split[1]
        _model_id = data_split[2]
        _tag = data_split[3]
        assert _tag in ['original', 'modified'], "tag should be original or modified"
        
        assert not osp.exists(osp.join(dataset_path_before_all, 'original', _model_id)), "original model should not exist"
        shutil.copytree(osp.join(_dataset_path, 'original', _model_id), osp.join(dataset_path_before_all, 'original', _model_id))
        
        if _tag == 'modified':
            assert not osp.exists(osp.join(dataset_path_before_all, _tag, _model_id)), "modified model should not exist"
            shutil.copytree(osp.join(_dataset_path, _tag, _model_id), osp.join(dataset_path_before_all, _tag, _model_id))
        
        print("copying {} / {}".format(_i, len(all_data_split)))
    
    print("done")
    
    all_data_split_dict = {}
    for _, cat, model_id, tag in all_data_split:
        if cat not in all_data_split_dict:
            all_data_split_dict[cat] = []
        assert model_id not in all_data_split_dict[cat], "model_id should not exist"
        all_data_split_dict[cat].append((int(model_id), tag))
    
    for key in all_data_split_dict:
        all_data_split_dict[key] = sorted(all_data_split_dict[key], key=lambda x: x[0])
    
    keys = list(all_data_split_dict.keys())
    keys = sorted(keys)
    
    with open(split_txt_before_all, "w") as f:
        for key in keys:
            for model_id, tag in all_data_split_dict[key]:
                f.write("{} {} {}\n".format(key, model_id, tag))
            print("writing {} / {}".format(key, len(all_data_split_dict[key])))
    
    print("done")
        
