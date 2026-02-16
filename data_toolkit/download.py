import os
import copy
import sys
import importlib
import argparse
import pandas as pd
from easydict import EasyDict as edict

if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--output_dir', type=str, default=None,   
                        help='Directory to save the metadata')
    parser.add_argument('--download_root', type=str, default=None,
                        help='Directory to download the objects')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--check_only', action='store_true',
                        help='Only check if the objects are already downloaded')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.output_dir = opt.output_dir or opt.root
    opt.download_root = opt.download_root or opt.root

    os.makedirs(opt.root, exist_ok=True)
    os.makedirs(opt.download_root, exist_ok=True)
    os.makedirs(os.path.join(opt.download_root, 'raw', 'new_records'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.download_root, 'raw', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.download_root, 'raw', 'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        if 'local_path' in metadata.columns:
            metadata = metadata[metadata['local_path'].isna()]
    else:
        if os.path.exists(opt.instances):
            with open(opt.instances, 'r') as f:
                instances = f.read().splitlines()
        else:
            instances = opt.instances.split(',')
        metadata = metadata[metadata['sha256'].isin(instances)]

    start = len(metadata) * opt.rank // opt.world_size
    end = len(metadata) * (opt.rank + 1) // opt.world_size
    metadata = metadata[start:end]
                
    print(f'Processing {len(metadata)} objects...')

    # process objects
    downloaded = dataset_utils.download(metadata, **opt)
    downloaded.to_csv(os.path.join(opt.download_root, 'raw', 'new_records', f'part_{opt.rank}.csv'), index=False)
