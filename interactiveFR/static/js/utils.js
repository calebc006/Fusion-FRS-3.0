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
    const SCORE_THRESHOLD = 0.9

    // Filter detections with bboxes
    const detectionsWithBbox = detections.filter((d) => {
        if (!d.bbox) return false;
        if (!d.score || d.score > SCORE_THRESHOLD) return false;
        if (!showUnknown && d.label === "Unknown") return false;
        return true;
    });

    detectionsWithBbox.forEach((detection, idx) => {
        const isUnknown = detection.label === "Unknown";
        const isTarget = detection.is_target === true;
        let bboxEl;

        if (idx < existingBoxes.length) {
            // Reuse existing element
            bboxEl = existingBoxes[idx];
        } else {
            // Create new element only if needed
            bboxEl = document.createElement("div");
            videoContainer.appendChild(bboxEl);
        }

        // Update classes: bbox + target/unknown/identified
        if (isTarget) {
            bboxEl.className = "bbox bbox-target";
        } else {
            bboxEl.className = isUnknown
                ? "bbox bbox-unknown"
                : "bbox bbox-identified";
        }

        // Update label content with matching classes
        if (showLabels) {
            if (isTarget) {
                bboxEl.innerHTML = `<p class="bbox-label bbox-label-target">TARGET</p>`;
            } else if (isUnknown) {
                bboxEl.innerHTML = `<p class="bbox-label bbox-label-unknown">${detection.label} <span class="bbox-score">${detection.score?.toFixed(2) || ""}</span></p>`;
            } else {
                bboxEl.innerHTML = `<p class="bbox-label bbox-label-identified">${detection.label} <span class="bbox-score">${detection.score?.toFixed(2) || ""}</span></p>`;
            }
        } else {
            bboxEl.innerHTML = "";
        }

        if (onBBoxCreate) onBBoxCreate(bboxEl, detection);

        setBBoxPos(
            bboxEl,
            detection.bbox,
            videoContainer.offsetWidth,
            videoContainer.offsetHeight,
        );

        bboxData.push(detection.bbox);
    });

    // Remove extra boxes if we have fewer detections than before
    for (let i = detectionsWithBbox.length; i < existingBoxes.length; i++) {
        existingBoxes[i].remove();
    }

    return bboxData;
};

export const delay = (time) => {
    return new Promise((resolve) => setTimeout(resolve, time));
};

export const fetchStreamStatus = async () => {
    const response = await fetch("/api/status");
    return response.json();
};

// Waits several times for stream to start, if not assume stream failed
export const waitForStream = async ({ attempts = 10, delayMs = 250 } = {}) => {
    for (let i = 0; i < attempts; i += 1) {
        const status = await fetchStreamStatus();
        if (status.stream_state === "running") {
            return status;
        }
        if (status.stream_state === "failed") {
            return status;
        }
        await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    return { stream_state: "failed", last_error: null };
};

export const waitForEmbeddings = async ({
    attempts = 40,
    delayMs = 250,
} = {}) => {
    for (let i = 0; i < attempts; i += 1) {
        const status = await fetchStreamStatus();
        if (status.stream_state === "failed") {
            return status;
        }
        if (status.embeddings_loaded) {
            return status;
        }
        await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    return {
        stream_state: "failed",
        last_error: null,
        embeddings_loaded: false,
    };
};

// type = "info", "error", "success"
export const showToast = (toast, message, type="info", duration_ms=3000) => {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.remove("is-success", "is-error", "is-info");
    toast.classList.add(`is-${type}`);
    toast.classList.add("show");
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => {
        toast.classList.remove("show");
    }, duration_ms);
};