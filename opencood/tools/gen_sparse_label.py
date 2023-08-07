import os
import glob
import random
import tqdm

import numpy as np

from opencood.hypes_yaml.yaml_utils import load_yaml, save_yaml


def gen_sparse_label(path):
    scenarios = sorted(os.listdir(path))
    for scenario in tqdm.tqdm(scenarios):
        spath = os.path.join(path, scenario)
        cavs = sorted([x for x in os.listdir(spath) if os.path.isdir(os.path.join(spath, x))])

        for cav in cavs:
            cav_path = os.path.join(path, scenario, cav)
            yaml_files = glob.glob(os.path.join(cav_path, '*.yaml'))
            yaml_files = [x for x in yaml_files if 'sparse_gt' not in x]
            for yf in yaml_files:
                params = load_yaml(yf)
                vehicles = params['vehicles']
                keys = list(vehicles.keys())
                if len(keys) > 1:
                    key = random.choice(keys)
                    params['vehicles'] = {key: vehicles[key]}
                out_file = yf.replace('.yaml', 'sparse_gt.yaml')
                if os.path.exists(out_file):
                    os.remove(out_file)
                out_file = yf.replace('.yaml', '_sparse_gt.yaml')
                save_yaml(params, save_name=out_file)


if __name__=="__main__":
    gen_sparse_label("/koko/OPV2V/train")
    gen_sparse_label("/koko/OPV2V/test")