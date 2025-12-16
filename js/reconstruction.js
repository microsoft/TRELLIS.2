var recon_items = [
    {
        video: "jet_engine_turbine.mp4",
        video_scale: 1.4, video_offset_y: -30,
        alt: "A jet engine turbine"
    },
    {
        video: "gun_mech.mp4",
        video_scale: 1.2, video_offset_y: 0,
        alt: "A gun mechanism"
    },
    {
        video: "plant_pot_stand.mp4",
        video_scale: 1.3, video_offset_y: 0,
        alt: "A plant pot stand"
    },
    {
        video: "shopping_cart.mp4",
        video_scale: 1.2, video_offset_y: 0,
        alt: "A shopping cart"
    },
    {
        video: "rusty_ferris_wheel.mp4",
        video_scale: 1.25, video_offset_y: 0,
        alt: "A rusty ferris wheel"
    },
    {
        video: "retro_fridge_tv.mp4",
        video_scale: 1.3, video_offset_y: -30,
        alt: "A retro fridge and TV"
    },
    {
        video: "steampunk_crystal_device.mp4",
        video_scale: 1.25, video_offset_y: 0,
        alt: "A steampunk crystal device"
    },
    {
        video: "sony_walkman.mp4",
        video_scale: 1.75, video_offset_y: -40,
        alt: "A Sony Walkman"
    },
];


function recon_carousel_item_template(item) {
    return `<div class="x-card" style="height: 70vh")\'>
                <div style="max-width: 100%; max-height: 100%; aspect-ratio: 1; pointer-events: none;">
                    <video autoplay playsinline loop muted height="100%" style="position: relative; max-height: 800px; z-index: -2; transform: scale(${item.video_scale}) translateY(${item.video_offset_y}px)" src="assets/reconstruction/${item.video}"></video>
                </div>
            </div>`;
}
