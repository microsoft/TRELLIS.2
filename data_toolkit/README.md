# Dataset Preparation Toolkit

This toolkit provides a comprehensive pipeline for preparing 3D datasets, including downloading, processing, voxelizing, and latent encoding for SC-VAE and Flow Model training.

### Step 1: Install Dependencies

Initialize the environment and install necessary dependencies:

```bash
. ./data_toolkit/setup.sh
```

### Step 2: Initialize Metadata

Before processing, load the dataset metadata.

```bash
python data_toolkit/build_metadata.py <SUBSET> --root <ROOT> [--source <SOURCE>]
```

**Arguments:**
- `SUBSET`: Target dataset subset. Options: `ObjaverseXL`, `ABO`, `HSSD`, `TexVerse` (Training sets); `SketchfabPicked`, `Toys4k` (Test sets).
- `ROOT`: Root directory to save the data.
- `SOURCE`: Data source (Required if `SUBSET` is `ObjaverseXL`). Options: `sketchfab`, `github`.

**Example:**
Load metadata for `ObjaverseXL` (sketchfab) and save to `datasets/ObjaverseXL_sketchfab`:
```bash
python data_toolkit/build_metadata.py ObjaverseXL --source sketchfab --root datasets/ObjaverseXL_sketchfab
```

### Step 3: Download Data

Download the 3D assets to the local storage.

```bash
python data_toolkit/download.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Arguments:**
- `RANK` / `WORLD_SIZE`: Parameters for multi-node distributed downloading.

**Example:**
To download the `ObjaverseXL` subset:

> **Note:** The example below sets a large `WORLD_SIZE` (160,000) for demonstration purposes, meaning only a tiny fraction of the dataset will be downloaded by this single process.

```bash
python data_toolkit/download.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --world_size 160000
```

*Attention: Some datasets may require an interactive Hugging Face login or manual steps. Please follow any on-screen instructions.*

**Update Metadata:**
After downloading, update the metadata registry:
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### Step 4: Process Mesh and PBR Textures

Standardize 3D assets by dumping mesh and PBR textures.
*Note: This process utilizes the CPU.*

```bash
# Dump Meshes
python data_toolkit/dump_mesh.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]

# Dump PBR Textures
python data_toolkit/dump_pbr.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]

# Get statistics of the asset
python data_toolkit/asset_stats.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Example:**
```bash
python data_toolkit/dump_mesh.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/dump_pbr.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
python data_toolkit/asset_stats.py --root datasets/ObjaverseXL_sketchfab
```

**Update Metadata:**
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### Step 5: Convert to O-Voxels

Convert the processed meshes and textures into O-Voxels format.
*Note: This process utilizes the CPU.*

```bash
python data_toolkit/dual_grid.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>]

python data_toolkit/voxelize_pbr.py <SUBSET> --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>]
```

**Arguments:**
- `RESOLUTION`: Target resolutions for O-Voxels, comma-separated (e.g., `256,512,1024`). Default is `256` for `dual_grid.py` and `1024` for `voxelize_pbr.py`.

**Example:**
Convert `ObjaverseXL` to resolutions 256, 512, and 1024:
```bash
python data_toolkit/dual_grid.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256,512,1024
python data_toolkit/voxelize_pbr.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab --resolution 256,512,1024
```


### At this point, the dataset is ready for SC-VAE Training

### Step 6: Encode Latents

Encode sparse structures into latents to train the first-stage generator.

```bash
# 1. Encode Shape Latents
python data_toolkit/encode_shape_latent.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>]

# 2. Encode PBR Latents
python data_toolkit/encode_pbr_latent.py --root <ROOT> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <RESOLUTION>]

# 3. Update Metadata (Required before next step)
python data_toolkit/build_metadata.py <SUBSET> --root <ROOT>

# 4. Encode Sparse Structure (SS) Latents
python data_toolkit/encode_ss_latent.py --root <ROOT> --shape_latent_name <SHAPE_LATENT_NAME> [--rank <RANK> --world_size <WORLD_SIZE>] [--resolution <SS_RESOLUTION>] 
```

**Arguments:**
- `RESOLUTION`: Input O-Voxel resolution. Default is `1024`.
- `SS_RESOLUTION`: Resolution for sparse structures. Default is `64`.
- `SHAPE_LATENT_NAME`: The specific version name of the shape latent.

**Example:**
```bash
python data_toolkit/encode_shape_latent.py --root datasets/ObjaverseXL_sketchfab --resolution 512
python data_toolkit/encode_pbr_latent.py --root datasets/ObjaverseXL_sketchfab --resolution 512
python data_toolkit/encode_shape_latent.py --root datasets/ObjaverseXL_sketchfab --resolution 1024
python data_toolkit/encode_pbr_latent.py --root datasets/ObjaverseXL_sketchfab --resolution 1024

# Update metadata
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab

# Encode SS Latents
python data_toolkit/encode_ss_latent.py --root datasets/ObjaverseXL_sketchfab --shape_latent_name shape_enc_next_dc_f16c32_fp16_1024 --resolution 64

# Final Metadata Update
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

### Step 7: Render Image Conditions

Render multi-view images to train the image-conditioned generator.
*Note: This process may utilize the CPU.*

```bash
python data_toolkit/render_cond.py <SUBSET> --root <ROOT> [--num_cond_views <NUM_COND_VIEWS>] [--rank <RANK> --world_size <WORLD_SIZE>]
```

**Arguments:**
- `NUM_COND_VIEWS`: Number of views to render per asset. Default is `16`.

**Example:**
```bash
python data_toolkit/render_cond.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```

**Final Metadata Update:**
```bash
python data_toolkit/build_metadata.py ObjaverseXL --root datasets/ObjaverseXL_sketchfab
```