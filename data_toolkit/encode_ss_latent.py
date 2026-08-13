import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import json
import argparse
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from easydict import EasyDict as edict
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

import trellis2.models as models

torch.set_grad_enabled(False)

def is_valid_sparse_tensor(tensor):
    return torch.isfinite(tensor.feats).all() and torch.isfinite(tensor.coords).all()

def clear_cuda_error():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='Directory to save the metadata')
    parser.add_argument('--shape_latent_root', type=str, default=None,
                        help='Directory to save the shape latent files')
    parser.add_argument('--ss_latent_root', type=str, default=None,
                        help='Directory to save the shape latent files')
    parser.add_argument('--filter_low_aesthetic_score', type=float, default=None,
                        help='Filter objects with aesthetic score lower than this value')
    parser.add_argument('--resolution', type=int, default=64,
                        help='Sparse voxel resolution')
    parser.add_argument('--shape_latent_name', type=str, required=True,
                        help='Name of the shape latent files')
    parser.add_argument('--enc_pretrained', type=str, default='microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16',
                        help='Pretrained encoder model')
    parser.add_argument('--model_root', type=str,
                        help='Root directory of models')
    parser.add_argument('--enc_model', type=str,
                        help='Encoder model. if specified, use this model instead of pretrained model')
    parser.add_argument('--ckpt', type=str,
                        help='Checkpoint to load')
    parser.add_argument('--instances', type=str, default=None,
                        help='Instances to process')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    opt = parser.parse_args()
    opt = edict(vars(opt))
    opt.shape_latent_root = opt.shape_latent_root or opt.root
    opt.ss_latent_root = opt.ss_latent_root or opt.root

    if opt.enc_model is None:
        latent_name = f'{opt.enc_pretrained.split("/")[-1]}_{opt.resolution}'
        encoder = models.from_pretrained(opt.enc_pretrained).eval().cuda()
    else:
        latent_name = f'{opt.enc_model.split("/")[-1]}_{opt.ckpt}_{opt.resolution}'
        cfg = edict(json.load(open(os.path.join(opt.model_root, opt.enc_model, 'config.json'), 'r')))
        encoder = getattr(models, cfg.models.encoder.name)(**cfg.models.encoder.args).cuda()
        ckpt_path = os.path.join(opt.model_root, opt.enc_model, 'ckpts', f'encoder_{opt.ckpt}.pt')
        encoder.load_state_dict(torch.load(ckpt_path), strict=False)
        encoder.eval()
        print(f'Loaded model from {ckpt_path}')
    
    os.makedirs(os.path.join(opt.ss_latent_root, 'ss_latents', latent_name, 'new_records'), exist_ok=True)
    
    # get file list
    if not os.path.exists(os.path.join(opt.root, 'metadata.csv')):
        raise ValueError('metadata.csv not found')
    metadata = pd.read_csv(os.path.join(opt.root, 'metadata.csv')).set_index('sha256')
    if os.path.exists(os.path.join(opt.root, 'aesthetic_scores', 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.root, 'aesthetic_scores','metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.shape_latent_root, 'shape_latents', opt.shape_latent_name, 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.shape_latent_root, 'shape_latents', opt.shape_latent_name,'metadata.csv')).set_index('sha256'))
    if os.path.exists(os.path.join(opt.ss_latent_root,'ss_latents', latent_name, 'metadata.csv')):
        metadata = metadata.combine_first(pd.read_csv(os.path.join(opt.ss_latent_root,'ss_latents', latent_name,'metadata.csv')).set_index('sha256'))
    metadata = metadata.reset_index()
    if opt.instances is None:
        if opt.filter_low_aesthetic_score is not None:
            metadata = metadata[metadata['aesthetic_score'] >= opt.filter_low_aesthetic_score]
        metadata = metadata[metadata['shape_latent_encoded'] == True]
        if 'ss_latent_encoded' in metadata.columns:
            metadata = metadata[metadata['ss_latent_encoded'] != True]
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
    records = []
    
    # filter out objects that are already processed
    sha256_list = os.listdir(os.path.join(opt.ss_latent_root, 'ss_latents', latent_name))
    sha256_list = [os.path.splitext(f)[0] for f in sha256_list if f.endswith('.npz')]
    for sha256 in sha256_list:
        records.append({'sha256': sha256, 'ss_latent_encoded': True})
    print(f'Found {len(sha256_list)} processed objects')
    metadata = metadata[~metadata['sha256'].isin(sha256_list)]
        
    print(f'Processing {len(metadata)} objects...')
    
    sha256s = list(metadata['sha256'].values)
    load_queue = Queue(maxsize=32)
    with ThreadPoolExecutor(max_workers=32) as loader_executor, \
         ThreadPoolExecutor(max_workers=32) as saver_executor:

        def loader(sha256):
            try:
                coords = np.load(os.path.join(opt.shape_latent_root, 'shape_latents', opt.shape_latent_name, f'{sha256}.npz'))['coords']
                assert np.all(coords < opt.resolution), f"{sha256}: Invalid coords"
                coords = torch.from_numpy(coords).long()
                ss = torch.zeros(1, opt.resolution, opt.resolution, opt.resolution, dtype=torch.long)
                ss[:, coords[:, 0], coords[:, 1], coords[:, 2]] = 1
                load_queue.put((sha256, ss))
            except Exception as e:
                print(f"[Loader Error] {sha256}: {e}")
                load_queue.put((sha256, None))

        loader_executor.map(loader, sha256s)
        
        def saver(sha256, pack):
            save_path = os.path.join(opt.ss_latent_root, 'ss_latents', latent_name, f'{sha256}.npz')
            np.savez_compressed(save_path, **pack)
            records.append({'sha256': sha256, 'ss_latent_encoded': True})
            
        for _ in tqdm(range(len(sha256s)), desc="Extracting latents"):
            try:
                sha256, ss = load_queue.get()
                if ss is None:
                    print(f"[Skip] {sha256}: Failed to load input")
                    continue
                
                ss = ss.cuda()[None].float()
                z = encoder(ss, sample_posterior=False)
                torch.cuda.synchronize()

                if not torch.isfinite(z).all():
                    print(f"[Skip] {sha256}: Non-finite latent")
                    clear_cuda_error()
                    continue

                pack = {
                    'z': z[0].cpu().numpy(),
                }
                saver_executor.submit(saver, sha256, pack)

            except Exception as e:
                print(f"[Error] {sha256}: {e}")
                clear_cuda_error()
                continue
            
        saver_executor.shutdown(wait=True)
        
    records = pd.DataFrame.from_records(records)
    records.to_csv(os.path.join(opt.ss_latent_root, 'ss_latents', latent_name, 'new_records', f'part_{opt.rank}.csv'), index=False)
