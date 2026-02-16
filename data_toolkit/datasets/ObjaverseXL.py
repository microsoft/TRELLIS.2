import os
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import objaverse.xl as oxl
from utils import get_file_hash
import tempfile
import zipfile



def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--source', type=str, default='sketchfab',
                        help='Data source to download annotations from (github, sketchfab)')


def get_metadata(source, **kwargs):
    if source == 'sketchfab':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_sketchfab.csv")
    elif source == 'github':
        metadata = pd.read_csv("hf://datasets/JeffreyXiang/TRELLIS-500K/ObjaverseXL_github.csv")
    else:
        raise ValueError(f"Invalid source: {source}")
    return metadata
        

def download(metadata, output_dir, **kwargs):    
    os.makedirs(os.path.join(output_dir, 'raw'), exist_ok=True)

    # download annotations
    annotations = oxl.get_annotations()
    annotations = annotations[annotations['sha256'].isin(metadata['sha256'].values)]
    
    # download and render objects
    file_paths = oxl.download_objects(
        annotations,
        download_dir=os.path.join(output_dir, "raw"),
        save_repo_format="zip",
    )
    
    downloaded = {}
    metadata = metadata.set_index("file_identifier")
    for k, v in file_paths.items():
        sha256 = metadata.loc[k, "sha256"]
        downloaded[sha256] = os.path.relpath(v, output_dir)

    return pd.DataFrame(downloaded.items(), columns=['sha256', 'local_path'])

def foreach_instance(metadata, output_dir, func, max_workers=None, desc='Processing objects', no_file=False):
    records = []
    if max_workers is None or max_workers <= 0:
        max_workers = os.cpu_count()

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor, \
            tqdm(total=len(metadata), desc=desc) as pbar:
            
            def worker(metadatum):
                try:
                    sha256 = metadatum['sha256']
                    if no_file:
                        record = func(None, metadatum)
                    else:
                        local_path = metadatum['local_path']
                        if local_path.startswith('raw/github/repos/'):
                            path_parts = local_path.split('/')
                            file_name = os.path.join(*path_parts[5:])
                            zip_file = os.path.join(output_dir, *path_parts[:5])
                            import tempfile, zipfile
                            with tempfile.TemporaryDirectory() as tmp_dir:
                                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                                    zip_ref.extractall(tmp_dir)
                                file = os.path.join(tmp_dir, file_name)
                                record = func(file, metadatum) 
                        else:
                            file = os.path.join(output_dir, local_path)
                            record = func(file, metadatum) 

                    if record is not None:
                        records.append(record)
                    pbar.update()
                except Exception as e:
                    print(f"Error processing object {metadatum.get('sha256', 'unknown')}: {e}")
                    pbar.update()
            
            for metadatum in metadata.to_dict('records'):
                executor.submit(worker, metadatum)
            
            executor.shutdown(wait=True)
    except Exception as e:
        print(f"Error happened during processing: {e}")
        
    return pd.DataFrame.from_records(records)