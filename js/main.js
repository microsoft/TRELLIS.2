window.onload = function() {
    initWindow();
    make_carousel('results-gen', gen_carousel_item_template, gen_items, 1, 1);
    make_carousel('results-recon', recon_carousel_item_template, recon_items, 1, 1);
};
