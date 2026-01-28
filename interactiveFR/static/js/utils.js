// ───────────────────────────── BBox Utilities ────────────────────────────
export const setBBoxPos = (el, bbox, w, h) => {
    const aspect = 16 / 9;
    let rw = w, rh = h;
    if (h / w > 9 / 16) rh = w / aspect;
    else rw = h * aspect;

    const offL = (w - rw) / 2, offT = (h - rh) / 2;
    const left = bbox[0] * rw, top = bbox[1] * rh;
    const bw = (bbox[2] - bbox[0]) * rw, bh = (bbox[3] - bbox[1]) * rh;

    el.style.left = `${Math.max(offL, left + offL) - 5}px`;
    el.style.top = `${Math.max(offT, top + offT) - 5}px`;
    el.style.width = `${Math.min(bw, rw - left)}px`;
    el.style.height = `${Math.min(bh, rh - top)}px`;
};

export const clearBBoxes = (container) => container.querySelectorAll(".bbox").forEach((el) => el.remove());

export const updateBBoxes = (container, detections, opts = {}) => {
    const { showLabels = false, showUnknown = true } = opts;
    const existing = container.querySelectorAll(".bbox");
    const bboxData = [];

    const valid = detections.filter((d) => d.bbox && (showUnknown || d.label !== "Unknown"));

    valid.forEach((d, i) => {
        const isUnknown = d.label === "Unknown";
        const isTarget = d.is_target === true;
        const el = i < existing.length ? existing[i] : document.createElement("div");
        if (i >= existing.length) container.appendChild(el);

        el.className = `bbox ${isTarget ? "bbox-target" : "bbox-identified"}`;

        if (showLabels && !isUnknown) {
            const cls = `bbox-label ${isTarget ? "bbox-label-target" : "bbox-label-identified"}`;
            const score = d.score != null ? d.score.toFixed(2) : "";
            el.innerHTML = `<p class="${cls}">${d.label}${score ? ` <span class=\"bbox-score\">${score}</span>` : ""}</p>`;
        } else {
            el.innerHTML = "";
        }

        setBBoxPos(el, d.bbox, container.offsetWidth, container.offsetHeight);
        bboxData.push(d.bbox);
    });

    for (let i = valid.length; i < existing.length; i++) existing[i].remove();
    return bboxData;
};

export const delay = (ms) => new Promise((r) => setTimeout(r, ms));
