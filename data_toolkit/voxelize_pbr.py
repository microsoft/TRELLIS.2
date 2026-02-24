import os
import copy
import sys
import importlib
import argparse
import pandas as pd
import pickle
import numpy as np
import torch
from easydict import EasyDict as edict
from functools import partial
import o_voxel


def _pbr_voxelize(file, metadatum, pbr_dump_root, root):
    sha256 = metadatum['sha256']
    try:
        pack = {'sha256': sha256}
        dump = None
        for res in opt.resolution:
            need_process = False

            # check if already processed
            if os.path.exists(os.path.join(root, f'pbr_voxels_{res}', f'{sha256}.vxz')):
                try:
                    info = o_voxel.io.read_vxz_info(os.path.join(root, f'pbr_voxels_{res}', f'{sha256}.vxz'))
                    pack[f'pbr_voxelized_{res}'] = True
                    pack[f'num_pbr_voxels_{res}'] = info['num_voxel']
                except Exception as e:
                    print(f'Error reading {sha256}.vxz: {e}')
                    need_process = True
            else:
                need_process = True

            # process if necessary
            if need_process:
                if dump is None:
                    with open(os.path.join(pbr_dump_root, 'pbr_dumps', f'{sha256}.pickle'), 'rb') as f:
                        dump = pickle.load(f)
                    # Fix dump alpha map
                    for mat in dump['materials']:
                        if mat['alphaTexture'] is not None and mat['alphaMode'] == 'OPAQUE':
                            mat['alphaMode'] = 'BLEND'
                    dump['materials'].append({
                        "baseColorFactor": [0.8, 0.8, 0.8],
                        "alphaFactor": 1.0,
                        "metallicFactor": 0.0,
                        "roughnessFactor": 0.5,
                        "alphaMode": "OPAQUE",
                        "alphaCutoff": 0.5,
                        "baseColorTexture": None,
                        "alphaTexture": None,
                        "metallicTexture": None,
                        "roughnessTexture": None,
                    })      # append default material
                    dump['objects'] = [
                        obj for obj in dump['objects']
                        if obj['vertices'].size != 0 and obj['faces'].size != 0
                    ]
                    vertices = torch.from_numpy(np.concatenate([obj['vertices'] for obj in dump['objects']], axis=0)).float()
                    vertices_min = vertices.min(dim=0)[0]
                    vertices_max = vertices.max(dim=0)[0]
                    center = (vertices_min + vertices_max) / 2
                    scale = 0.99999 / (vertices_max - vertices_min).max()
                    for obj in dump['objects']:
                        obj['vertices'] = (torch.from_numpy(obj['vertices']).float() - center) * scale
                        obj['vertices'] = obj['vertices'].numpy()
                        obj['mat_ids'][obj['mat_ids'] == -1] = len(dump['materials']) - 1
                        assert np.all(obj['mat_ids'] >= 0), 'invalid mat_ids'
                        assert np.all(obj['vertices'] >= -0.5) and np.all(obj['vertices'] <= 0.5), 'vertices out of range'

                coord, attr = o_voxel.convert.blender_dump_to_volumetric_attr(dump, grid_size=res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                                                                              mip_level_offset=0, verbose=False, timing=False)
                del attr['normal']
                del attr['emissive']
                o_voxel.io.write_vxz(os.path.join(root, f'pbr_voxels_{res}', f'{sha256}.vxz'), coord, attr)
                pack[f'pbr_voxelized_{res}'] = True
                pack[f'num_pbr_voxels_{res}'] = len(coord)

        return pack
    except Exception as e:
        print(f'Error voxelizing {sha256}: {e}')
        return {'sha256': sha256, 'error': str(e)}


if __name__ == '__main__':
    dataset_utils = importlib.import_module(f'datasets.{sys.argv[1]}')

    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--pbr_dump_root', type=str, default=None,
                        help='Directory to load mesh dumps')
    parser.add_argument('--pbr_voxel_root', type=str, default=None,
                        help='Directory to save voxelized pbr attributes')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    dataset_utils.add_args(parser)
    parser.add_argument('--resolution', type=str, default=1024)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--max_workers', type=int, default=0)
    opt = parser.parse_args(sys.argv[2:])
    opt = edict(vars(opt))
    opt.resolution = sorted([int(x) for x in opt.resolution.split(',')], reverse=True)
    opt.pbr_dump_root = opt.pbr_dump_root or opt.root
    opt.pbr_voxel_root = opt.pbr_voxel_root or opt.root

    for res in opt.resolution:
        os.makedirs(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{res}', 'new_records'), exist_ok=True)

    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.pbr_dump_root, 'pbr_dumps', 'metadata.csv')).set_index('sha256'))
    for res in opt.resolution:
        if os.path.exists(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{res}', 'metadata.csv')):
            pbr_voxel_metadata = pd.read_csv(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{res}','metadata.csv')).set_index('sha256')
            pbr_voxel_metadata = pbr_voxel_metadata.rename(columns={'pbr_voxelized': f'pbr_voxelized_{res}', 'num_pbr_voxels': f'num_pbr_voxels_{res}'})
            metadata = metadata.combine_first(pbr_voxel_metadata)
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['pbr_dumped'] == True]
        mask = np.zeros(len(metadata), dtype=bool)
        for res in opt.resolution:
            if f'pbr_voxelized_{res}' in metadata.columns:
                mask |= metadata[f'pbr_voxelized_{res}'] != True
            else:
                mask[:] = True
                break
        metadata = metadata[mask]
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
    func = partial(_pbr_voxelize, pbr_dump_root=opt.pbr_dump_root, root=opt.pbr_voxel_root)
    pbr_voxelized = dataset_utils.foreach_instance(metadata, None, func, max_workers=opt.max_workers, no_file=True, desc='Voxelizing')
    if 'error' in pbr_voxelized.columns:
        errors = pbr_voxelized[pbr_voxelized['error'].notna()]
        with open('errors.txt', 'w') as f:
            f.write('\n'.join(errors['sha256'].tolist()))
    for res in opt.resolution:
        if f'pbr_voxelized_{res}' in pbr_voxelized.columns:
            pbr_voxel_metadata = pbr_voxelized[pbr_voxelized[f'pbr_voxelized_{res}'] == True]
            if len(pbr_voxel_metadata) > 0:
                pbr_voxel_metadata = pbr_voxel_metadata[['sha256', f'pbr_voxelized_{res}', f'num_pbr_voxels_{res}']]
                pbr_voxel_metadata = pbr_voxel_metadata.rename(columns={f'pbr_voxelized_{res}': 'pbr_voxelized', f'num_pbr_voxels_{res}': 'num_pbr_voxels'})
                pbr_voxel_metadata.to_csv(os.path.join(opt.pbr_voxel_root, f'pbr_voxels_{res}', 'new_records', f'part_{opt.rank}.csv'), index=False)
