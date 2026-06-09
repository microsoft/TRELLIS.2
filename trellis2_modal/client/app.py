"""
Gradio-based local client for TRELLIS.2 3D generation via Modal.

Provides a user interface for:
- Image upload and preprocessing
- Resolution and generation parameter controls
- Video preview of generated 3D model
- GLB extraction and download
"""

from __future__ import annotations

import base64
import os
import tempfile

import gradio as gr
import numpy as np

from .api import APIError, TRELLIS2APIClient


MAX_SEED = np.iinfo(np.int32).max


def get_client() -> TRELLIS2APIClient | None:
    """
    Create API client from environment variables or secrets file.

    Requires TRELLIS2_API_URL to be set.
    Credentials are loaded from (in order of priority):
    1. Environment variables: TRELLIS2_MODAL_KEY and TRELLIS2_MODAL_SECRET
    2. Secrets file: ~/.trellis2_modal_secrets.json

    Returns:
        TRELLIS2APIClient if configured, None otherwise
    """
    api_url = os.environ.get("TRELLIS2_API_URL")
    if not api_url:
        return None

    try:
        # Credentials loaded automatically from env vars or secrets file
        return TRELLIS2APIClient(base_url=api_url)
    except ValueError:
        # No credentials found
        return None


def get_seed(randomize_seed: bool, seed: int) -> int:
    """Get the random seed."""
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def generate_3d(
    image: str | None,
    seed: int,
    resolution: str,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    shape_slat_guidance_strength: float,
    shape_slat_sampling_steps: int,
    tex_slat_guidance_strength: float,
    tex_slat_sampling_steps: int,
) -> tuple[str | None, str | None, str]:
    """
    Generate 3D from uploaded image via Modal API.

    Args:
        image: Path to uploaded image
        seed: Random seed
        resolution: "512", "1024", or "1536"
        ss_*: Sparse structure params
        shape_slat_*: Shape SLAT params
        tex_slat_*: Texture SLAT params

    Returns:
        Tuple of (video_path, state_b64, status_message)
    """
    if image is None:
        return None, None, "⚠️ Please upload an image first."

    client = get_client()
    if client is None:
        return None, None, "❌ Error: TRELLIS2_API_URL not set or credentials missing."

    # Map resolution to pipeline_type
    pipeline_type = {
        "512": "512",
        "1024": "1024_cascade",
        "1536": "1536_cascade",
    }.get(resolution, "1024_cascade")

    try:
        result = client.generate(
            image_path=image,
            seed=seed,
            pipeline_type=pipeline_type,
            ss_sampling_steps=ss_sampling_steps,
            ss_guidance_strength=ss_guidance_strength,
            shape_slat_sampling_steps=shape_slat_sampling_steps,
            shape_slat_guidance_strength=shape_slat_guidance_strength,
            tex_slat_sampling_steps=tex_slat_sampling_steps,
            tex_slat_guidance_strength=tex_slat_guidance_strength,
        )

        # Decode video and save to temp file
        video_bytes = base64.b64decode(result["video"])
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            video_path = f.name

        # Build status message
        elapsed = client.last_request_elapsed or 0
        cold_start_msg = " (cold start)" if client.was_cold_start() else ""
        status = f"✅ Generated in {elapsed:.1f}s{cold_start_msg}"

        return video_path, result["state"], status

    except APIError as e:
        return None, None, f"❌ API Error: {e.code} - {e.message}"
    except FileNotFoundError as e:
        return None, None, f"❌ {e}"
    except Exception as e:
        return None, None, f"❌ Error: {e}"


def extract_glb(
    state: str | None,
    decimation_target: int,
    texture_size: int,
) -> tuple[str | None, str | None, str]:
    """
    Extract GLB mesh from generation state via Modal API.

    Args:
        state: Base64 compressed state from generate()
        decimation_target: Target vertex count
        texture_size: Texture resolution

    Returns:
        Tuple of (glb_path, download_path, status_message)
    """
    if state is None:
        return None, None, "⚠️ Please generate a 3D model first."

    client = get_client()
    if client is None:
        return None, None, "❌ Error: TRELLIS2_API_URL not set or credentials missing."

    try:
        # Create temp file for output
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            output_path = f.name

        client.extract_glb(
            state=state,
            output_path=output_path,
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1.0,
            remesh_project=0.0,
        )

        elapsed = client.last_request_elapsed or 0
        return output_path, output_path, f"✅ Extracted in {elapsed:.1f}s"

    except APIError as e:
        return None, None, f"❌ API Error: {e.code} - {e.message}"
    except Exception as e:
        return None, None, f"❌ Error: {e}"


def create_interface() -> gr.Blocks:
    """Create and configure the Gradio interface."""
    with gr.Blocks(title="TRELLIS.2 3D Generator") as demo:
        gr.Markdown("""
        ## Image to 3D Asset with [TRELLIS.2](https://microsoft.github.io/TRELLIS.2) via Modal
        
        * Upload an image and click **Generate** to create a 3D asset.
        * Click **Extract GLB** to export and download the GLB file.
        * Resolution affects quality and generation time:
          - **512**: ~10s, fast preview
          - **1024**: ~30s, good quality (recommended)
          - **1536**: ~90s, maximum quality
        """)

        # State for storing generation result
        generation_state = gr.State(value=None)

        with gr.Row():
            # Left column: Input and controls
            with gr.Column(scale=1, min_width=360):
                image_input = gr.Image(
                    label="Image Prompt",
                    type="filepath",
                    image_mode="RGBA",
                    height=400,
                )

                resolution = gr.Radio(
                    ["512", "1024", "1536"],
                    label="Resolution",
                    value="1024",
                )

                with gr.Row():
                    seed = gr.Slider(
                        0,
                        MAX_SEED,
                        label="Seed",
                        value=0,
                        step=1,
                    )
                    randomize_seed = gr.Checkbox(
                        label="Randomize",
                        value=True,
                    )

                generate_btn = gr.Button("Generate", variant="primary")
                gen_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    show_label=False,
                )

                with gr.Accordion(label="Advanced Settings", open=False):
                    gr.Markdown("**Stage 1: Sparse Structure Generation**")
                    with gr.Row():
                        ss_guidance_strength = gr.Slider(
                            1.0,
                            10.0,
                            label="Guidance Strength",
                            value=7.5,
                            step=0.1,
                        )
                        ss_sampling_steps = gr.Slider(
                            1,
                            50,
                            label="Sampling Steps",
                            value=12,
                            step=1,
                        )

                    gr.Markdown("**Stage 2: Shape Generation**")
                    with gr.Row():
                        shape_slat_guidance_strength = gr.Slider(
                            1.0,
                            10.0,
                            label="Guidance Strength",
                            value=7.5,
                            step=0.1,
                        )
                        shape_slat_sampling_steps = gr.Slider(
                            1,
                            50,
                            label="Sampling Steps",
                            value=12,
                            step=1,
                        )

                    gr.Markdown("**Stage 3: Material Generation**")
                    with gr.Row():
                        tex_slat_guidance_strength = gr.Slider(
                            1.0,
                            10.0,
                            label="Guidance Strength",
                            value=1.0,
                            step=0.1,
                        )
                        tex_slat_sampling_steps = gr.Slider(
                            1,
                            50,
                            label="Sampling Steps",
                            value=12,
                            step=1,
                        )

            # Right column: Output
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Preview"):
                        video_output = gr.Video(
                            label="3D Preview",
                            height=500,
                            autoplay=True,
                            loop=True,
                        )

                    with gr.TabItem("Extract GLB"):
                        with gr.Row():
                            decimation_target = gr.Slider(
                                100000,
                                1000000,
                                label="Decimation Target (vertices)",
                                value=500000,
                                step=10000,
                            )
                            texture_size = gr.Dropdown(
                                [1024, 2048, 4096],
                                label="Texture Size",
                                value=2048,
                            )

                        extract_btn = gr.Button("Extract GLB")
                        extract_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            show_label=False,
                        )

                        glb_output = gr.Model3D(
                            label="3D Model",
                            height=400,
                            clear_color=(0.25, 0.25, 0.25, 1.0),
                        )
                        download_btn = gr.DownloadButton(
                            label="Download GLB",
                            visible=True,
                        )

        # Wire up events
        generate_btn.click(
            fn=get_seed,
            inputs=[randomize_seed, seed],
            outputs=[seed],
        ).then(
            fn=generate_3d,
            inputs=[
                image_input,
                seed,
                resolution,
                ss_guidance_strength,
                ss_sampling_steps,
                shape_slat_guidance_strength,
                shape_slat_sampling_steps,
                tex_slat_guidance_strength,
                tex_slat_sampling_steps,
            ],
            outputs=[video_output, generation_state, gen_status],
        )

        extract_btn.click(
            fn=extract_glb,
            inputs=[generation_state, decimation_target, texture_size],
            outputs=[glb_output, download_btn, extract_status],
        )

    return demo


def main() -> None:
    """Entry point for the client application."""
    api_url = os.environ.get("TRELLIS2_API_URL")

    if not api_url:
        print("=" * 60)
        print("TRELLIS.2 Modal Client")
        print("=" * 60)
        print()
        print("⚠️  Warning: TRELLIS2_API_URL not set.")
        print()
        print("Set the endpoint URL:")
        print("  export TRELLIS2_API_URL=https://your-app--generate.modal.run")
        print()
        print("Credentials are loaded from (in order):")
        print("  1. TRELLIS2_MODAL_KEY and TRELLIS2_MODAL_SECRET env vars")
        print("  2. ~/.trellis2_modal_secrets.json")
        print()
        print("Starting anyway (will show error when generating)...")
        print()

    demo = create_interface()
    demo.launch()


if __name__ == "__main__":
    main()
