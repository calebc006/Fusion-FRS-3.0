import { updateBBoxes, clearBBoxes, waitForStream, fetchStreamStatus } from "./utils.js";
import { initReferencesUI, refreshReferenceImages } from "./references.js";

// ───────────────────────────── State ─────────────────────────────────────
let hasTarget = false;

const $ = (id) => document.getElementById(id);
const captureTarget = $("capture-target");
const captureHeader = document.querySelector(".capture-header");
const captureInput = $("capture-name");
const captureBtn = $("capture-button");
const referencesPanelBtn = $("references-panel-button");
const captureToast = $("capture-toast");
const videoFeed = $("video-feed");
let isVideoReady = false;

// ───────────────────────────── Init ──────────────────────────────────────

const tryRestartStream = async () => {
    const streamSrc = localStorage.getItem("streamSrc");
    if (!streamSrc) {
        return { ok: false, message: "Missing stream source." };
    }

    const formData = new FormData();
    formData.set("stream_src", streamSrc);

    const response = await fetch("/api/start", { method: "POST", body: formData });
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
        videoFeed?.setAttribute("data", `/api/vidFeed?t=${Date.now()}`);

        videoFeed?.addEventListener("load", () => {
            isVideoReady = true;
            clearBBoxes($("video-container"));
        });

        const srcLabel = $("stream-source");
        if (srcLabel) {
            srcLabel.textContent = localStorage.getItem("streamSrc") || "";
        }

        clearBBoxes($("video-container"));
        fetchDetections();
        initReferencesUI({ autoLoad: true });
    } catch {
        alert("Unable to connect to server.");
        window.location.href = "/";
    }
});

// ───────────────────────────── Main Loop ─────────────────────────────────
const fetchDetections = () => {
    let buffer = "";
    let data = [];

    fetch("/api/frResults")
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

                    if (isVideoReady) {
                        updateBBoxes($("video-container"), data, {
                            showLabels: true,
                            showUnknown: true,
                        });
                    } else {
                        clearBBoxes($("video-container"));
                    }
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

    const safeName = name
    .replace(/[^a-zA-Z0-9\s_-]/g, "") // Remove non-legal (non-alnum, non-space, non-dash, non-underscore)
    .trim()                           // Remove leading/trailing spaces
    .replace(/\s+/g, "_")             // Replace internal spaces with underscores
    .toUpperCase();                   // Convert to uppercase
    
    captureBtn.disabled = true;    
    showCaptureToast("Capturing...", "info");

    try {
        const res = await fetch("/api/capture", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: safeName }),
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

    // Check if name is already in database, warn if so
    const response = await fetch("/api/get_reference_names");
    const namelist = await response.json();
    console.log(namelist);
    if (namelist.includes(safeName)) {
        alert(`Beware: ${safeName} was already in the database! If this was unintended, remove name and try again.`);
    }
});

// ───────────────────────────── References Modal ──────────────────────────
const modal = $("references-modal");
const openReferencesPage = $("open-references-page");

referencesPanelBtn?.addEventListener("click", (e) => {
    e.preventDefault();
    modal?.classList.remove("hidden");
    refreshReferenceImages({ preserveSelection: true });
});

openReferencesPage?.addEventListener("click", (e) => {
    e.preventDefault();
    window.open("/references", "_blank");
});

$("close-references-modal")?.addEventListener("click", () =>
    modal?.classList.add("hidden"),
);
modal
    ?.querySelector(".modal-overlay")
    ?.addEventListener("click", () => modal?.classList.add("hidden"));

document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        modal?.classList.add("hidden");
    }
});
