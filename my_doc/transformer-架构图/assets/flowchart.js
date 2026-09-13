// ========== 状态 ==========
let currentView = 'main';
let currentMode = 'drag';
let activeContainer = null;
let viewState = {};

// ========== Mermaid 初始化 ==========
mermaid.initialize({
    startOnLoad: false, theme: 'base',
    themeVariables: {
        primaryColor: '#cc785c', primaryTextColor: '#141413', primaryBorderColor: '#a9583e',
        lineColor: '#8e8b82', secondaryColor: '#5db8a6', tertiaryColor: '#e8a55a',
        fontFamily: 'Inter, sans-serif', fontSize: '14px'
    },
    flowchart: { useMaxWidth: false, htmlLabels: true, curve: 'basis', padding: 15, nodeSpacing: 30, rankSpacing: 45 }
});

// ========== 工具模式 ==========
function setMode(mode) {
    currentMode = mode;
    document.getElementById('btnZoom').classList.toggle('active', mode === 'zoom');
    document.getElementById('btnDrag').classList.toggle('active', mode === 'drag');
}
function resetView() {
    if (!activeContainer) return;
    const id = activeContainer.id;
    if (viewState[id]) { viewState[id] = { zoom: 1, tx: 0, ty: 0 }; applyTransform(activeContainer); }
}
function applyTransform(c) {
    const div = c.querySelector('.mermaid');
    const s = viewState[c.id];
    if (div && s) div.style.transform = `translate(${s.tx}px, ${s.ty}px) scale(${s.zoom})`;
}

// ========== 点击事件 ==========
function addClickEvents(container, flowId) {
    container.querySelectorAll('.mermaid .node').forEach(node => {
        node.style.cursor = 'pointer';
        const texts = [];
        node.querySelectorAll('text, tspan, .nodeLabel').forEach(el => { if (el.textContent) texts.push(el.textContent); });
        const fullText = texts.join(' ');

        if (flowId === 'mainFlow') {
            for (const [kw, id] of Object.entries(CONFIG.nodeToMainSection)) {
                if (fullText.includes(kw)) { node.addEventListener('click', () => jumpTo('section' + id)); break; }
            }
            for (const [kw, id] of Object.entries(CONFIG.nodeToSubFlow)) {
                if (fullText.includes(kw)) {
                    const shape = node.querySelector('rect, polygon, circle');
                    if (shape) { shape.setAttribute('stroke-dasharray', '5,3'); shape.setAttribute('stroke-width', '3'); }
                    node.addEventListener('dblclick', e => { e.stopPropagation(); showSubFlow(id); });
                    break;
                }
            }
        } else {
            const map = CONFIG.subNodeToSection[flowId];
            if (map) {
                for (const [kw, id] of Object.entries(map)) {
                    if (fullText.includes(kw)) { node.addEventListener('click', () => jumpTo(id)); break; }
                }
            }
        }
        node.addEventListener('mouseenter', function() { this.style.opacity = '0.8'; });
        node.addEventListener('mouseleave', function() { this.style.opacity = '1'; });
    });
}

// ========== 视图切换 ==========
function showSubFlow(subId) {
    currentView = subId;
    CONFIG.allFlows.forEach(id => document.getElementById(id).style.display = 'none');
    CONFIG.allTitles.forEach(id => { const e = document.getElementById(id); if (e) e.style.display = 'none'; });
    document.getElementById(subId).style.display = 'block';
    const te = document.getElementById(subId + 'Title'); if (te) te.style.display = 'block';
    document.getElementById('flowTitle').textContent = CONFIG.subTitles[subId] || '';
    document.getElementById('backBtn').classList.add('visible');
    CONFIG.allDetails.forEach(id => document.getElementById(id).style.display = 'none');
    const di = CONFIG.subFlowDetails[subId]; if (di) document.getElementById(di).style.display = 'block';
    const md = document.querySelector('#' + subId + ' .mermaid');
    if (md && !md.getAttribute('data-processed')) {
        mermaid.run({ nodes: [md] }).then(() => {
            md.setAttribute('data-processed', 'true');
            addClickEvents(document.getElementById(subId), subId);
            setupZoomForContainer(document.getElementById(subId));
        });
    } else { setupZoomForContainer(document.getElementById(subId)); }
}

function showMain() {
    currentView = 'main';
    CONFIG.allFlows.forEach(id => document.getElementById(id).style.display = 'none');
    CONFIG.allTitles.forEach(id => { const e = document.getElementById(id); if (e) e.style.display = 'none'; });
    document.getElementById('mainFlow').style.display = 'block';
    document.getElementById('flowTitle').textContent = CONFIG.mainTitle;
    document.getElementById('flowTitle').style.display = 'block';
    document.getElementById('backBtn').classList.remove('visible');
    CONFIG.allDetails.forEach(id => document.getElementById(id).style.display = 'none');
    document.getElementById('mainDetails').style.display = 'block';
    setupZoomForContainer(document.getElementById('mainFlow'));
}

function jumpTo(sectionId) {
    document.querySelectorAll('.code-section').forEach(s => s.classList.remove('highlight'));
    const s = document.getElementById(sectionId);
    if (!s) return;
    s.classList.add('highlight');
    setTimeout(() => s.scrollIntoView({ behavior: 'smooth', block: 'start' }), 100);
}

// ========== 代码块折叠 ==========
function toggleSection(id) {
    const c = document.getElementById('section' + id) || document.getElementById(id);
    if (c) { const ct = c.querySelector('.code-content'); if (ct) ct.classList.toggle('active'); }
}

// ========== 分隔条拖动 ==========
(function() {
    const r = document.getElementById('resizer'), lp = document.getElementById('leftPanel');
    let ir = false;
    r.addEventListener('mousedown', () => { ir = true; r.classList.add('active'); document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none'; });
    document.addEventListener('mousemove', e => { if (!ir) return; const rb = document.querySelector('.container').getBoundingClientRect(); lp.style.width = Math.max(300, Math.min(700, e.clientX - rb.left)) + 'px'; });
    document.addEventListener('mouseup', () => { ir = false; r.classList.remove('active'); document.body.style.cursor = ''; document.body.style.userSelect = ''; });
})();

// ========== 缩放与拖拽 ==========
function setupZoomForContainer(container) {
    const div = container.querySelector('.mermaid'); if (!div) return;
    if (!viewState[container.id]) viewState[container.id] = { zoom: 1, tx: 0, ty: 0 };
    activeContainer = container;
    container.onwheel = e => {
        if (currentMode !== 'zoom') return;
        e.preventDefault();
        const st = viewState[container.id];
        const rc = container.getBoundingClientRect();
        const mx = e.clientX - rc.left, my = e.clientY - rc.top;
        const nz = Math.max(.3, Math.min(3, st.zoom * (1 - e.deltaY * .005)));
        st.tx = mx - (nz / st.zoom) * (mx - st.tx);
        st.ty = my - (nz / st.zoom) * (my - st.ty);
        st.zoom = nz;
        applyTransform(container);
    };
    let panning = false, sx = 0, sy = 0, ptx = 0, pty = 0;
    container.onmousedown = e => {
        if (currentMode !== 'drag' || e.target.closest('.node') || e.target.closest('.nodeLabel')) return;
        panning = true; sx = e.clientX; sy = e.clientY;
        ptx = viewState[container.id].tx; pty = viewState[container.id].ty;
        container.classList.add('grabbing'); e.preventDefault();
    };
    document.onmousemove = e => { if (!panning) return; viewState[container.id].tx = ptx + (e.clientX - sx); viewState[container.id].ty = pty + (e.clientY - sy); applyTransform(container); };
    document.onmouseup = () => { panning = false; container.classList.remove('grabbing'); };
}

// ========== 页面加载 ==========
document.addEventListener('DOMContentLoaded', () => {
    const mm = document.querySelector('#mainFlow .mermaid');
    mermaid.run({ nodes: [mm] }).then(() => {
        mm.setAttribute('data-processed', 'true');
        addClickEvents(document.getElementById('mainFlow'), 'mainFlow');
        setupZoomForContainer(document.getElementById('mainFlow'));
    });
});
