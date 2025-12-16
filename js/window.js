var fullscreenElement = null;


function preventDefault(e) {
    e.preventDefault();
}


function stopPropagation(e) {
    e.stopPropagation();
}


function initWindow() {
    fullscreenElement = document.createElement('div');
    fullscreenElement.id = 'fullscreen';
    fullscreenElement.innerHTML = `
        <div id="window">
            <div id="close" onclick="closeWindow()">✕</div>
            <div id="content"></div>
        </div>
    `;
    document.body.appendChild(fullscreenElement);
    fullscreenElement.addEventListener('wheel', preventDefault);
    fullscreenElement.addEventListener('touchmove', preventDefault);
}


function openWindow(content) {
    let contentElement = fullscreenElement.querySelector('#content');
    contentElement.innerHTML = content;
    fullscreenElement.style.display = 'flex';
    setTimeout(() => {
        fullscreenElement.style.opacity = 1;
    }, 100);
    
    // if copntent height is greater than window height, enable scroll
    let contentHeight = contentElement.clientHeight;
    let innerHeight = contentElement.children[0].clientHeight;
    if (innerHeight > contentHeight) {
        contentElement.addEventListener('wheel', stopPropagation);
        contentElement.addEventListener('touchmove', stopPropagation);
    }
    else {
        contentElement.removeEventListener('wheel', stopPropagation);
        contentElement.removeEventListener('touchmove', stopPropagation);
    }
}


function closeWindow() {
    window_state = {};
    fullscreenElement.style.opacity = 0;
    setTimeout(() => {
        fullscreenElement.style.display = 'none';
    }, 250);
}


var window_state = {};


function hideTexture() {
    let appearanceButton = document.getElementById('appearance-button');
    let geometryButton = document.getElementById('geometry-button');
    appearanceButton.classList.remove('checked');
    geometryButton.classList.add('checked');
    let modelViewer = document.getElementById('modelviewer');
    if (modelViewer.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture === null) return;
    window_state.textures = [];
    window_state.mr_textures = [];
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        window_state.textures.push(modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.texture);
        window_state.mr_textures.push(modelViewer.model.materials[i].pbrMetallicRoughness.metallicRoughnessTexture.texture);
    }
    window_state.exposure = modelViewer.exposure;
    modelViewer.environmentImage = 'assets/env_maps/gradient.jpg';
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(null);
        modelViewer.model.materials[i].pbrMetallicRoughness.metallicRoughnessTexture.setTexture(null);
    }
    modelViewer.exposure = 5;
}


function showTexture() {
    let appearanceButton = document.getElementById('appearance-button');
    let geometryButton = document.getElementById('geometry-button');
    appearanceButton.classList.add('checked');
    geometryButton.classList.remove('checked');
    let modelViewer = document.getElementById('modelviewer');
    if (modelViewer.model.materials[0].pbrMetallicRoughness.baseColorTexture.texture !== null) return;
    modelViewer.environmentImage = undefined;
    for (let i = 0; i < modelViewer.model.materials.length; i++) {
        modelViewer.model.materials[i].pbrMetallicRoughness.baseColorTexture.setTexture(window_state.textures[i]);
        modelViewer.model.materials[i].pbrMetallicRoughness.metallicRoughnessTexture.setTexture(window_state.mr_textures[i]);
    }
    modelViewer.exposure = 1;
}


function downloadGLB() {
    let modelViewer = document.getElementById('modelviewer');
    window.open(modelViewer.src);
}


function asset_panel_template(prompt) {
    return `
        <div class="x-small-header" style="margin: 16px auto;">Prompt</div>
        <div class="modelviewer-panel-prompt">
            ${prompt}
        </div>
        <div class="x-small-header" style="margin: 16px auto; margin-top: 32px;">Display Mode</div>
        <div class="x-left-align">
            <div id="appearance-button" class="modelviewer-panel-button small checked" onclick="showTexture()">Appearance</div>
            <div id="geometry-button" class="modelviewer-panel-button small" onclick="hideTexture()">Geometry</div>
        </div>
        <div class="x-flex-spacer"></div>
        <div class="x-row">
            <div id="download-button" class="modelviewer-panel-button" onclick="downloadGLB()">Download GLB</div>
        </div>
    `;
}

function modelviewer_window_template(item, panel, config) {
    let viewer_size = config && config.viewer_size || 500;
    let panel_size = config && config.panel_size || 300;
    let show_annotations = config && config.show_annotations || false;
    html = `<div class="x-row" style="align-items: stretch; flex-wrap: wrap; width: ${viewer_size + panel_size + 32}px; max-width: calc(100vw - 32px);">
                <div class="modelviewer-container" style="position: relative; width: ${viewer_size}px; background: #222222;">
                    <div style="position: absolute; bottom: 16px; left: 16px; font-size: 14px; font-weight: 500; color: #555555ff;">
                        NOTE: Mesh simplified for online viewing. Rendered without alpha channel (not well-supported by web 3D viewer).
                    </div>
                    <model-viewer
                        id="modelviewer"
                        src="${item.model}"
                        camera-controls
                        tone-mapping="natural"
                        shadow-intensity="1"
                        exposure="1"
                        >`
    if (show_annotations) {
        window_state.assets = item.assets;
        window_state.prompt_template = item.prompt_template;
        for (let i = 0; i < item.assets.length; i++) {
            html += `<button slot="hotspot-${i}" data-position="${item.assets[i].position.join(' ')}">${item.assets[i].name}</button>`;
        }
    }
    html += `        </model-viewer>
                </div>
                <div class="modelviewer-panel" style="flex: 1 1 ${panel_size}px;">
                    ${panel}
                </div>
            </div>`;
    return html;
}
