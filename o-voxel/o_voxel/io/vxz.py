from typing import *
import os
import json
import struct
import torch
import numpy as np
import zlib
import lzma
import zstandard
from concurrent.futures import ThreadPoolExecutor
from ..serialize import encode_seq, decode_seq
from .. import _C


__all__ = [
    "read_vxz",
    "read_vxz_info",
    "write_vxz",
]


"""
VXZ format

Header:
- file type (3 bytes) - 'VXZ'
- version (1 byte) - 0
- binary start offset (4 bytes)
- structure (json) -
{
    "num_voxel": int,
    "chunk_size": int,
    "filter": str,
    "compression": str,
    "compression_level": int,
    "raw_size": int,
    "compressed_size": int,
    "compress_ratio": float,
    "attr_interleave": str,
    "attr": [
        {"name": str, "chs": int},
        ...
    ]
    "chunks": [
        {
            "ptr": [offset, length],        # offset from global binary start
            "svo": [offset, length],        # offset from this chunk start
            "attr": [offset, length],       # offset from this chunk start
        },
        ...
    ]
}
- binary data
"""

DEFAULT_COMPRESION_LEVEL = {
    'none': 0,
    'deflate': 9,
    'lzma': 9,
    'zstd': 22,
}


def _compress(data: bytes, algo: Literal['none', 'deflate', 'lzma', 'zstd'], level: int) -> bytes:
    if algo == 'none':
        return data
    if level is None:
        level = DEFAULT_COMPRESION_LEVEL[algo]
    if algo == 'deflate':
        compresser = zlib.compressobj(level, wbits=-15)
        return compresser.compress(data) + compresser.flush()
    if algo == 'lzma':
        compresser = lzma.LZMACompressor(format=lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA2, 'preset': level}])
        return compresser.compress(data) + compresser.flush()
    if algo == 'zstd':
        compresser = zstandard.ZstdCompressor(level=level, write_checksum=False, write_content_size=True, threads=-1)
        return compresser.compress(data)
    raise ValueError(f"Invalid compression algorithm: {algo}")


def _decompress(data: bytes, algo: Literal['none', 'deflate', 'lzma', 'zstd'], level: int) -> bytes:
    if algo == 'none':
        return data
    if level is None:
        level = DEFAULT_COMPRESION_LEVEL[algo]
    if algo == 'deflate':
        decompresser = zlib.decompressobj(wbits=-15)
        return decompresser.decompress(data) + decompresser.flush()
    if algo == 'lzma':
        decompresser = lzma.LZMADecompressor(format=lzma.FORMAT_RAW, filters=[{'id': lzma.FILTER_LZMA2, 'preset': level}])
        return decompresser.decompress(data)
    if algo == 'zstd':
        decompresser = zstandard.ZstdDecompressor(format=zstandard.FORMAT_ZSTD1)
        return decompresser.decompress(data)
    raise ValueError(f"Invalid compression algorithm: {algo}")


def read_vxz_info(file) -> Dict:
    """
    Read the header of a VXZ file without decompressing the binary data.
    
    Args:
        file_path: Path or file-like object to the VXZ file.
        
    Returns:
        Dict: the header of the VXZ file.
    """
    if isinstance(file, str):
        with open(file, 'rb') as f:
            file_data = f.read()
    else:
        file_data = file.read()
        
    assert file_data[:3] == b'VXZ', "Invalid file type"
    version = file_data[3]
    assert version == 0, "Invalid file version"
    
    bin_start = struct.unpack('>I', file_data[4:8])[0]
    structure_data = json.loads(file_data[8:bin_start].decode())
    return structure_data


def read_vxz(file, num_threads: int = -1) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Read a VXZ file containing voxels.
    
    Args:
        file_path: Path or file-like object to the VXZ file.
        num_threads: the number of threads to use for reading the file.
        
    Returns:
        torch.Tensor: the coordinates of the voxels.
        Dict[str, torch.Tensor]: the attributes of the voxels.
    """
    if isinstance(file, str):
        with open(file, 'rb') as f:
            file_data = f.read()
    else:
        file_data = file.read()
        
    num_threads = num_threads if num_threads > 0 else os.cpu_count()
    
    # Parse header
    assert file_data[:3] == b'VXZ', "Invalid file type"
    version = file_data[3]
    assert version == 0, "Invalid file version"
    
    bin_start = struct.unpack('>I', file_data[4:8])[0]
    structure_data = json.loads(file_data[8:bin_start].decode())
    bin_data = file_data[bin_start:]
    
    # Decode chunks
    chunk_size = structure_data['chunk_size']
    chunk_depth = np.log2(chunk_size)
    assert chunk_depth.is_integer(), f"Chunk size must be a power of 2, got {chunk_size}"
    chunk_depth = int(chunk_depth)
    
    def worker(chunk_info):
        decompressed = {}
        chunk_data = bin_data[chunk_info['ptr'][0]:chunk_info['ptr'][0]+chunk_info['ptr'][1]]
        for k, v in chunk_info.items():
            if k in ['ptr', 'idx']:
                continue
            decompressed[k] = np.frombuffer(_decompress(chunk_data[v[0]:v[0]+v[1]], structure_data['compression'], structure_data['compression_level']), dtype=np.uint8)
        svo = torch.tensor(np.frombuffer(decompressed['svo'], dtype=np.uint8))
        morton_code = _C.decode_sparse_voxel_octree_cpu(svo, chunk_depth)
        coord = decode_seq(morton_code.int()).cpu()
        
        # deinterleave attributes
        if structure_data['attr_interleave'] == 'none':
            all_attr = []
            for k, chs in structure_data['attr']:
                for i in range(chs):
                    all_attr.append(torch.tensor(decompressed[f'{k}_{i}']))
            all_attr = torch.stack(all_attr, dim=1)     
        elif structure_data['attr_interleave'] == 'as_is':
            all_attr = []
            for k, chs in structure_data['attr']:
                all_attr.append(torch.tensor(decompressed[k].reshape(-1, chs)))
            all_attr = torch.cat(all_attr, dim=1)
        elif structure_data['attr_interleave'] == 'all':
            all_chs = sum(chs for k, chs in structure_data['attr'])
            all_attr = decompressed['attr'].reshape(-1, all_chs)
        
        # unfilter
        if structure_data['filter'] == 'none':
            pass
        elif structure_data['filter'] == 'parent':
            all_attr = _C.decode_sparse_voxel_octree_attr_parent_cpu(svo, chunk_depth, all_attr)
        elif structure_data['filter'] == 'neighbor':
            all_attr = _C.decode_sparse_voxel_octree_attr_neighbor_cpu(coord, chunk_size, all_attr)
        
        # final
        attr = {}
        ch = 0
        for k, chs in structure_data['attr']:
            attr[k] = all_attr[:, ch:ch+chs]
            ch += chs
        return {
            'coord': coord,
            'attr': attr,
        }
            
    if num_threads == 1:
        chunks = [worker(info) for info in structure_data['chunks']]
    else:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            chunks = list(executor.map(worker, structure_data['chunks']))
    
    # Combine chunks
    coord = []
    attr = {k: [] for k, _ in structure_data['attr']}
    for info, chunk in zip(structure_data['chunks'], chunks):
        coord.append(chunk['coord'] + torch.tensor([[info['idx'][0] * chunk_size, info['idx'][1] * chunk_size, info['idx'][2] * chunk_size]]).int())
        for k, v in chunk['attr'].items():
            attr[k].append(v)
    coord = torch.cat(coord, dim=0)
    for k, v in attr.items():
        attr[k] = torch.cat(v, dim=0)
    return coord, attr


def write_vxz(
    file,
    coord: torch.Tensor,
    attr: Dict[str, torch.Tensor],
    chunk_size: int = 256,
    filter: Literal['none', 'parent', 'neighbor'] = 'none',
    compression: Literal['none', 'deflate', 'lzma', 'zstd'] = 'lzma',
    compression_level: Optional[int] = None,
    attr_interleave: Literal['none', 'as_is', 'all'] = 'as_is',
    num_threads: int = -1,
):
    """
    Write a VXZ file containing voxels.
    
    Args:
        file: Path or file-like object to the VXZ file.
        coord: the coordinates of the voxels.
        attr: the attributes of the voxels.
        chunk_size: the size of each chunk.
        filter: the filter to apply to the voxels.
        compression: the compression algorithm to use.
        compression_level: the level of compression.
        attr_interleave: how to interleave the attributes.
        num_threads: the number of threads to use for compression.
    """
    # Check
    for k, v in attr.items():
        assert coord.shape[0] == v.shape[0], f"Number of coordinates and attributes do not match for key {k}"
        assert v.dtype == torch.uint8, f"Attributes must be uint8, got {v.dtype} for key {k}"
    assert attr_interleave in ['none', 'as_is', 'all'], f"Invalid attr_interleave value: {attr_interleave}"
    
    compression_level = compression_level or DEFAULT_COMPRESION_LEVEL[compression]
    num_threads = num_threads if num_threads > 0 else os.cpu_count()
    
    file_info = {
        'num_voxel': coord.shape[0],
        'chunk_size': chunk_size,
        'filter': filter,
        'compression': compression,
        'compression_level': compression_level,
        'raw_size': sum([coord.numel() * 4] + [v.numel() for v in attr.values()]),
        'compressed_size': 0,
        'compress_ratio': 0.0,
        'attr_interleave': attr_interleave,
        'attr': [[k, v.shape[1]] for k, v in attr.items()],
        'chunks': [],
    }
    bin_data = b''
    
    # Split into chunks
    chunk_depth = np.log2(chunk_size)
    assert chunk_depth.is_integer(), f"Chunk size must be a power of 2, got {chunk_size}"
    chunk_depth = int(chunk_depth)
    
    chunk_coord = coord // chunk_size 
    coord = coord % chunk_size
    unique_chunk_coord, inverse = torch.unique(chunk_coord, dim=0, return_inverse=True)
    
    chunks = []
    for idx, chunk_xyz in enumerate(unique_chunk_coord.tolist()):
        chunk_mask = (inverse == idx)
        chunks.append({
            'idx': chunk_xyz,
            'coord': coord[chunk_mask],
            'attr': {k: v[chunk_mask] for k, v in attr.items()},
        })
    
    # Compress each chunk
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        def worker(chunk):
            ## compress to binary
            coord = chunk['coord']
            morton_code = encode_seq(coord)
            sorted_idx = morton_code.argsort().cpu()
            coord = coord.cpu()[sorted_idx]
            morton_code = morton_code.cpu()[sorted_idx]
            attr = torch.cat([v.cpu()[sorted_idx] for v in chunk['attr'].values()], dim=1)
            svo = _C.encode_sparse_voxel_octree_cpu(morton_code, chunk_depth)
            svo_bytes = _compress(svo.numpy().tobytes(), compression, compression_level)
            
            # filter
            if filter == 'none':
                attr = attr.numpy()
            elif filter == 'parent':
                attr = _C.encode_sparse_voxel_octree_attr_parent_cpu(svo, chunk_depth, attr).numpy()
            elif filter == 'neighbor':
                attr = _C.encode_sparse_voxel_octree_attr_neighbor_cpu(coord, chunk_size, attr).numpy()
            
            # interleave attributes
            attr_bytes = {}
            if attr_interleave == 'none':
                ch = 0
                for k, chs in file_info['attr']:
                    for i in range(chs):
                        attr_bytes[f'{k}_{i}'] = _compress(attr[:, ch].tobytes(), compression, compression_level)
                        ch += 1
            elif attr_interleave == 'as_is':
                ch = 0
                for k, chs in file_info['attr']:
                    attr_bytes[k] = _compress(attr[:, ch:ch+chs].tobytes(), compression, compression_level)
                    ch += chs
            elif attr_interleave == 'all':
                attr_bytes['attr'] = _compress(attr.tobytes(), compression, compression_level)
            
            ## buffer for each chunk
            chunk_info = {'idx': chunk['idx']}            
            bin_data = b''
            
            ### svo
            chunk_info['svo'] = [len(bin_data), len(svo_bytes)]
            bin_data += svo_bytes
            
            ### attr
            for k, v in attr_bytes.items():
                chunk_info[k] = [len(bin_data), len(v)]
                bin_data += v
            
            return chunk_info, bin_data
            
        chunks = list(executor.map(worker, chunks))
    
    for chunk_info, chunk_data in chunks:
        chunk_info['ptr'] = [len(bin_data), len(chunk_data)]
        bin_data += chunk_data
        file_info['chunks'].append(chunk_info)
        
    file_info['compressed_size'] = len(bin_data)
    file_info['compress_ratio'] = file_info['raw_size'] / file_info['compressed_size']
    
    # File parts
    structure_data = json.dumps(file_info).encode()
    header = b'VXZ\x00' + struct.pack('>I', len(structure_data) + 8)
    
    # Write to file
    if isinstance(file, str):
        with open(file, 'wb') as f:
            f.write(header)
            f.write(structure_data)
            f.write(bin_data)
    else:
        file.write(header)
        file.write(structure_data)
        file.write(bin_data)
