"""
Run TRELLIS.2 texturing on an arbitrary 3D mesh.

Usage:
    python texture_mesh.py <mesh_path> <image_path> [options]

Examples:
    python texture_mesh.py model.ply photo.png
    python texture_mesh.py model.glb photo.webp --output out.glb --resolution 1024 --texture-size 2048
    python texture_mesh.py model.obj photo.jpg --steps 20 --guidance 3.0 --seed 42
"""
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import sys
import time
import trimesh
from PIL import Image
from trellis2.pipelines import Trellis2TexturingPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply TRELLIS.2 PBR texturing to a 3D mesh",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("mesh", help="Input mesh file (.ply, .obj, .glb, .gltf)")
    parser.add_argument("image", help="Reference image for texturing (.png, .jpg, .webp)")
    parser.add_argument("--output", default=None,
                        help="Output GLB path (default: <mesh_stem>_textured.glb)")
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B",
                        help="HuggingFace model ID or local path")
    parser.add_argument("--resolution", type=int, default=1024, choices=[512, 1024, 1536],
                        help="Voxel resolution for geometry encoding")
    parser.add_argument("--texture-size", type=int, default=2048,
                        help="Output texture map resolution (pixels)")
    parser.add_argument("--steps", type=int, default=12,
                        help="Number of flow ODE solver steps")
    parser.add_argument("--guidance", type=float, default=1.0,
                        help="Classifier-free guidance strength")
    parser.add_argument("--guidance-rescale", type=float, default=0.0,
                        help="Guidance rescale factor")
    parser.add_argument("--rescale-t", type=float, default=3.0,
                        help="Time rescaling for ODE solver")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--no-preprocess", action="store_true",
                        help="Skip image preprocessing (background removal). "
                             "Use if image already has alpha channel or clean background.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate inputs
    if not os.path.isfile(args.mesh):
        print(f"Error: mesh file not found: {args.mesh}")
        sys.exit(1)
    if not os.path.isfile(args.image):
        print(f"Error: image file not found: {args.image}")
        sys.exit(1)

    # Default output path
    if args.output is None:
        stem = os.path.splitext(os.path.basename(args.mesh))[0]
        args.output = os.path.join(os.path.dirname(args.mesh) or ".", f"{stem}_textured.glb")

    print(f"Mesh:        {args.mesh}")
    print(f"Image:       {args.image}")
    print(f"Output:      {args.output}")
    print(f"Resolution:  {args.resolution}")
    print(f"Texture size:{args.texture_size}")
    print(f"Steps:       {args.steps}")
    print(f"Guidance:    {args.guidance}")
    print(f"Seed:        {args.seed}")
    print()

    # Load pipeline
    print("Loading pipeline...")
    pipeline = Trellis2TexturingPipeline.from_pretrained(
        args.model, config_file="texturing_pipeline.json"
    )
    pipeline.cuda()

    # Load inputs
    print("Loading mesh...")
    mesh = trimesh.load(args.mesh, force="mesh")
    print(f"  Vertices: {len(mesh.vertices):,}  Faces: {len(mesh.faces):,}")

    print("Loading image...")
    image = Image.open(args.image)
    print(f"  Size: {image.size}  Mode: {image.mode}")

    # Run texturing
    print("Running texturing...")
    t0 = time.perf_counter()
    output = pipeline.run(
        mesh,
        image,
        seed=args.seed,
        preprocess_image=not args.no_preprocess,
        tex_slat_sampler_params={
            "steps": args.steps,
            "guidance_strength": args.guidance,
            "guidance_rescale": args.guidance_rescale,
            "rescale_t": args.rescale_t,
        },
        resolution=args.resolution,
        texture_size=args.texture_size,
    )
    t1 = time.perf_counter()
    print(f"  Texturing: {t1 - t0:.1f}s")

    # Export
    print(f"Exporting to {args.output} ...")
    t2 = time.perf_counter()
    output.export(args.output, extension_webp=True)
    t3 = time.perf_counter()
    print(f"  Export:    {t3 - t2:.1f}s")
    print(f"  Total:     {t3 - t0:.1f}s")


if __name__ == "__main__":
    main()
