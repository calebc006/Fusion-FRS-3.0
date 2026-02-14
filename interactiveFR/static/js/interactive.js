import { updateBBoxes, clearBBoxes, waitForStream, fetchStreamStatus, showToast } from "./utils.js";
import { initReferencesUI, refreshReferenceImages } from "./references.js";

// ───────────────────────────── State ─────────────────────────────────────
let hasTarget = false;

const $ = (id) => document.getElementById(id);
const captureTarget = $("capture-target");
const captureHeader = document.querySelector(".capture-header");
const captureInput = $("capture-name");
const captureBtn = $("capture-button");
const referencesPanelBtn = $("references-panel-button");
const toast = $("toast");
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



captureBtn.addEventListener("click", async () => {
    const name = captureInput.value.trim();
    if (!name) return showToast(toast, "Enter a name first.", "error");

    const safeName = name
    .replace(/[^a-zA-Z0-9\s_-]/g, "") // Remove all non-legal (non-alnum, non-space, non-dash, non-underscore)
    .trim()                           // Remove leading/trailing spaces
    .replace(/\s+/g, "_")             // Replace internal spaces with underscores
    .toUpperCase();                   // Convert to uppercase
    
    captureBtn.disabled = true;    
    showToast(toast, "Capturing...", "info");

    try {
        // Capture first WITHOUT allowing duplicate
        const res = await fetch("/api/capture", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: safeName, allow_duplicate: false }),
        });
        let result = await res.json();
        
        // Check if we are blocking because of duplicate
        if (res.status === 202) {
            let allow = confirm(`Warning: ${safeName} is already in the database! Proceed?`);
            if (allow) {
                // Give confirmation
                const new_res = await fetch("/api/capture/confirm", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ name: safeName, allow_duplicate: true }),
                });
                result = await new_res.json();
            }
            else {
                // add back capture btn and exit
                showToast(toast, "Canceled capture", "success");
                captureBtn.disabled = !hasTarget || !captureInput.value.trim(); 
                return;
            }
        }
        
        showToast(
            toast, 
            result.message || "Done.",
            result.ok ? "success" : "error",
        );

        if (result.ok) 
            captureInput.value = "";
        
    } catch {
        showToast(toast, "Capture failed unexpectedly.", "error");
    } finally {
        captureBtn.disabled = !hasTarget || !captureInput.value.trim();
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
