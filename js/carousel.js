var carousel_objects = {};


function make_carousel(carousel_id, item_template, items, rows_to_display, cols_to_display) {
    carousel_objects[carousel_id] = {};
    carousel_objects[carousel_id].item_template = item_template;
    carousel_objects[carousel_id].items = items;
    carousel_objects[carousel_id].active_items = Array.from({ length: items.length }, (_, i) => i);
    carousel_objects[carousel_id].rows_to_display = rows_to_display;
    carousel_objects[carousel_id].cols_to_display = cols_to_display;
    carousel_objects[carousel_id].num_to_display = rows_to_display * cols_to_display;
    carousel_objects[carousel_id].num_pages = Math.ceil(items.length / carousel_objects[carousel_id].num_to_display);
    carousel_objects[carousel_id].current_page = 0;
    carousel_objects[carousel_id].has_tags = items.some(item => item.tags && item.tags.length > 0);
    carousel_objects[carousel_id].tags = ["All"].concat(Array.from(new Set(items.flatMap(item => item.tags ?? []))));
    carousel_init(carousel_id);
}


function carousel_init(carousel_id) {
    let carousel = document.getElementById(carousel_id);
    let html = "";
    if (carousel_objects[carousel_id].has_tags) {
        html += '<div class="x-carousel-tags">';
        for (let tag of carousel_objects[carousel_id].tags) {
            if (tag === "All") {
                html += `<div class="x-carousel-tag active" onclick="carousel_tag_select('${carousel_id}', '${tag}')">${tag}</div>`;
            }
            else {
                html += `<div class="x-carousel-tag" onclick="carousel_tag_select('${carousel_id}', '${tag}')">${tag}</div>`;
            }
        }
        html += '</div>';
    }
    html += '<div class="x-carousel-slider">'
    for (let i = 0; i < carousel_objects[carousel_id].num_to_display; i++) {
        html += `<div class="x-carousel-slider-item" style="flex-basis: calc(100% / ${carousel_objects[carousel_id].cols_to_display});"></div>`;
    }
    html += '</div>';
    html += '<div class="x-carousel-nav">';
    html += `<div class="x-carousel-switch" onclick="carousel_prev('${carousel_id}')">&lt;</div>`;
    html += '<div class="x-carousel-pages">';
    for (let i = 0; i < carousel_objects[carousel_id].num_pages; i++) {
        html += `<div class="x-carousel-page${i === 0 ? ' x-carousel-page-active' : ''}" onclick="carousel_page('${carousel_id}', ${i})"></div>`;
    }
    html += '</div>';
    html += `<div class="x-carousel-switch" onclick="carousel_next('${carousel_id}')">&gt;</div>`;
    html += '</div>';
    carousel.innerHTML = html;
    carousel_render(carousel_id);
}


function carousel_render(carousel_id) {
    let carousel = document.getElementById(carousel_id);
    let slider = carousel.querySelector('.x-carousel-slider');
    num_to_display = carousel_objects[carousel_id].num_to_display;
    let current_page = carousel_objects[carousel_id].current_page;
    start_idx = current_page * num_to_display;
    for (let i = 0; i < num_to_display; i++) {
        let item_idx = start_idx + i;
        let item = slider.children[i];
        let info = { item_idx: item_idx, page_idx: current_page, display_idx: i };
        if (item_idx < carousel_objects[carousel_id].active_items.length) {
            item.innerHTML = carousel_objects[carousel_id].item_template(
                carousel_objects[carousel_id].items[carousel_objects[carousel_id].active_items[item_idx]],
                info
            );
        } else {
            item.innerHTML = "";
        }
    }
}


function carousel_resize(carousel_id) {
    let carousel = document.getElementById(carousel_id);
    let slider = carousel.querySelector('.x-carousel-nav');
    html = "";
    html += `<div class="x-carousel-switch" onclick="carousel_prev('${carousel_id}')">&lt;</div>`;
    html += '<div class="x-carousel-pages">';
    for (let i = 0; i < carousel_objects[carousel_id].num_pages; i++) {
        html += `<div class="x-carousel-page${i === 0 ? ' x-carousel-page-active' : ''}" onclick="carousel_page('${carousel_id}', ${i})"></div>`;
    }
    html += '</div>';
    html += `<div class="x-carousel-switch" onclick="carousel_next('${carousel_id}')">&gt;</div>`;
    slider.innerHTML = html;
}


function carousel_page(carousel_id, page) {
    let carousel = document.getElementById(carousel_id);
    carousel_objects[carousel_id].current_page = page;
    carousel.querySelector('.x-carousel-page-active').classList.remove('x-carousel-page-active');
    carousel.querySelector('.x-carousel-pages').children[page].classList.add('x-carousel-page-active');
    carousel_render(carousel_id);
}


function carousel_prev(carousel_id) {
    page = carousel_objects[carousel_id].current_page - 1;
    if (page < 0) page += carousel_objects[carousel_id].num_pages;
    carousel_page(carousel_id, page);
}

function carousel_next(carousel_id) {
    page = carousel_objects[carousel_id].current_page + 1;
    if (page >= carousel_objects[carousel_id].num_pages) page -= carousel_objects[carousel_id].num_pages;
    carousel_page(carousel_id, page);
}


function carousel_tag_select(carousel_id, tag) {
    let carousel = document.getElementById(carousel_id);
    let tags = carousel.querySelectorAll('.x-carousel-tag');
    for (let i = 0; i < tags.length; i++) {
        tags[i].classList.remove('active');
    }
    if (tag === "All") {
        tags[0].classList.add('active');
    }
    else {
        for (let i = 0; i < tags.length; i++) {
            if (tags[i].textContent === tag) {
                tags[i].classList.add('active');
                break;
            }
        }
    }
    let items = carousel_objects[carousel_id].items;
    if (tag === "All") {
        carousel_objects[carousel_id].active_items = Array.from({ length: items.length }, (_, i) => i);
    }
    else {
        let active_items = [];
        for (let i = 0; i < items.length; i++) {
            if (items[i].tags && items[i].tags.includes(tag)) {
                active_items.push(i);
            }
        }
        carousel_objects[carousel_id].active_items = active_items;
    }
    carousel_objects[carousel_id].num_pages = Math.ceil(carousel_objects[carousel_id].active_items.length / carousel_objects[carousel_id].num_to_display);
    carousel_objects[carousel_id].current_page = 0;
    carousel_resize(carousel_id);
    carousel_render(carousel_id);
}
