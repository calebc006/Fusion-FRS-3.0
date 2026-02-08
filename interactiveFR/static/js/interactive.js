import { updateBBoxes, clearBBoxes, waitForStream, fetchStreamStatus } from "./utils.js";

// ───────────────────────────── State ─────────────────────────────────────
let hasTarget = false;

const $ = (id) => document.getElementById(id);
const detectionList = $("table-detection-list");
const captureTarget = $("capture-target");
const captureHeader = document.querySelector(".capture-header");
const captureInput = $("capture-name");
const captureBtn = $("capture-button");
const referencesPanelBtn = $("references-panel-button");
const captureToast = $("capture-toast");

// ───────────────────────────── Init ──────────────────────────────────────

const tryRestartStream = async () => {
    const streamSrc = localStorage.getItem("streamSrc");
    if (!streamSrc) {
        return { ok: false, message: "Missing stream source." };
    }

    const formData = new FormData();
    formData.set("stream_src", streamSrc);

    const response = await fetch("/start", { method: "POST", body: formData });
    const data = await response.json();
    if (!data.stream) {
        return {
            ok: false,
            message: data.message || "Failed to start stream.",
        };
    }

    const status = await waitForStream();
    if (status.stream_state === "running") {
        return { ok: true };
    }
    return {
        ok: false,
        message: status.last_error
            ? `Stream failed (${status.stream_state}): ${status.last_error}`
            : `Stream failed (${status.stream_state}). Please try again.`,
    };
};

window.addEventListener("DOMContentLoaded", async () => {
    try {
        const status = await fetchStreamStatus();
        if (status.stream_state !== "running") {
            const restart = await tryRestartStream();
            if (!restart.ok) {
                alert(
                    restart.message ||
                        "Stream not started. Please start from Home.",
                );
                return (window.location.href = "/");
            }
        }

        // cache-buster to prevent getting stuck by browser caching
        $("video-feed").setAttribute("data", `/vidFeed?t=${Date.now()}`);

        const srcLabel = $("stream-source");
        if (srcLabel) {
            srcLabel.textContent = localStorage.getItem("streamSrc") || "";
        }

        clearBBoxes($("video-container"));
        fetchDetections();
    } catch {
        alert("Unable to connect to server.");
        window.location.href = "/";
    }
});

// ───────────────────────────── Main Loop ─────────────────────────────────
const fetchDetections = () => {
    let buffer = "";
    let data = [];

    fetch("/frResults")
        .then((response) => {
            if (!response.ok || !response.body) {
                console.error("Fetch failed, retrying...");
                setTimeout(() => fetchDetections(), 5000);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            const processStream = () => {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        console.log("Stream ended, reconnecting...");
                        setTimeout(() => fetchDetections(), 2000);
                        return;
                    }

                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk;

                    const parts = buffer.split("\n");

                    try {
                        if (parts.length > 1) {
                            data = JSON.parse(parts[parts.length - 2])?.data || [];
                        }
                    } catch (err) {
                        console.error("Error parsing JSON:", err);
                    }

                    buffer = parts[parts.length - 1] || "";

                    updateBBoxes($("video-container"), data, {
                        showLabels: true,
                        showUnknown: true,
                    });
                    updateDetectionList(data);
                    updateCapturePanel(data);

                    processStream(); // recursive call
                });
            };

            processStream();
        })
        .catch((error) => {
            console.error("Error fetching detections:", error);
            setTimeout(() => fetchDetections(), 5000);
        });
};

function updateDetectionList(data) {
    const seen = new Set();
    const detections = [];

    data.forEach((d) => {
        const label = d.label?.toUpperCase();
        if (!label || label === "UNKNOWN" || seen.has(label)) return;
        seen.add(label);

        const el = document.createElement("div");
        el.className = "table-detection-element";
        el.dataset.name = label;
        el.innerHTML = `<span class="detection-name">${label}</span>`;
        detections.push(el);
    });

    detections.sort((a, b) =>
        (a.dataset.name || "").localeCompare(b.dataset.name || ""),
    );
    detectionList.replaceChildren(...detections);
}

function updateCapturePanel(data) {
    hasTarget = data.some((d) => d.is_target);
    captureTarget.textContent = hasTarget ? "Ready" : "None";
    captureBtn.disabled = !hasTarget || !captureInput.value.trim();
    captureHeader?.classList.toggle("is-ready", hasTarget);
}

// ───────────────────────────── Capture ───────────────────────────────────
captureInput.addEventListener(
    "input",
    () => (captureBtn.disabled = !hasTarget || !captureInput.value.trim()),
);

const showCaptureToast = (message, type = "info") => {
    if (!captureToast) return;
    captureToast.textContent = message;
    captureToast.classList.remove("is-success", "is-error", "is-info");
    captureToast.classList.add(`is-${type}`);
    captureToast.classList.add("show");
    clearTimeout(showCaptureToast._timer);
    showCaptureToast._timer = setTimeout(() => {
        captureToast.classList.remove("show");
    }, 3000);
};

captureBtn.addEventListener("click", async () => {
    const name = captureInput.value.trim();
    if (!name) return showCaptureToast("Enter a name first.", "error");

    captureBtn.disabled = true;
    showCaptureToast("Capturing...", "info");

    try {
        const res = await fetch("/capture", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name }),
        });
        const result = await res.json();
        showCaptureToast(
            result.message || "Done.",
            result.ok ? "success" : "error",
        );
        if (result.ok) captureInput.value = "";
    } catch {
        showCaptureToast("Capture failed.", "error");
    } finally {
        captureBtn.disabled = !hasTarget || !captureInput.value.trim();
    }
});

// ───────────────────────────── References Modal ──────────────────────────
const modal = $("references-modal");
const openReferencesPage = $("open-references-page");
const nameSelect = $("name-select");
const gallery = $("image-gallery");
const imgCount = $("image-count");
const noImgs = $("no-images");
const lightbox = $("lightbox");
const lightboxImg = lightbox?.querySelector("img");

let refData = [];

referencesPanelBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    modal.classList.remove("hidden");
    loadReferences();
});

openReferencesPage?.addEventListener("click", (e) => {
    e.preventDefault();
    window.open("/references", "_blank");
});

$("close-references-modal")?.addEventListener("click", () =>
    modal.classList.add("hidden"),
);
modal
    ?.querySelector(".modal-overlay")
    ?.addEventListener("click", () => modal.classList.add("hidden"));

async function loadReferences() {
    try {
        refData = await (await fetch("/reference_images")).json();
        nameSelect.innerHTML = '<option value="">-- All --</option>';

        if (!refData.length) {
            noImgs.style.display = "block";
            gallery.innerHTML = "";
            imgCount.textContent = "";
            return;
        }

        noImgs.style.display = "none";
        refData.forEach((p) => {
            const opt = document.createElement("option");
            opt.value = p.name;
            opt.textContent = `${p.name} (${p.images.length})`;
            nameSelect.appendChild(opt);
        });
        showImages();
    } catch (e) {
        console.error(e);
        noImgs.textContent = "Error loading images.";
        noImgs.style.display = "block";
    }
}

function showImages(name = null) {
    gallery.innerHTML = "";
    const people = name ? refData.filter((p) => p.name === name) : refData;
    let total = 0;

    people.forEach((p) =>
        p.images.forEach((src) => {
            const card = document.createElement("div");
            card.className = "image-card";
            card.innerHTML = `<img src="${src}" alt="${p.name}" loading="lazy"><div class="image-name">${src.split("/").pop()}</div>`;
            card.addEventListener("click", () => {
                lightboxImg.src = src;
                lightbox.classList.add("active");
            });
            gallery.appendChild(card);
            total++;
        }),
    );

    imgCount.textContent = name
        ? `${total} images for "${name}"`
        : `${total} images from ${refData.length} people`;
}

nameSelect?.addEventListener("change", (e) =>
    showImages(e.target.value || null),
);

// Lightbox close
lightbox
    ?.querySelector(".lightbox-close")
    ?.addEventListener("click", () => lightbox.classList.remove("active"));
lightbox?.addEventListener(
    "click",
    (e) => e.target === lightbox && lightbox.classList.remove("active"),
);
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        lightbox.classList.remove("active");
        modal.classList.add("hidden");
    }
});
