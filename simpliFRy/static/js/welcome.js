import {
    getCountryFlag,
    getDescription,
    setBBoxPos,
    clearBBoxes,
    updateBBoxes,
    loadNamelistJSON,
    sortDetectionsByPriority,
    fetchSettings
} from "./utils.js";

const N = 3; // number of detections shown (last N)

const detectionList = document.getElementById("detections-list");
const countryFlagImg = document.getElementById("country-flag-img");
const videoModal = document.getElementById("video-modal");
const videoContainer = document.getElementById("video-container");
let namelistPath = null;
let namelistJSON = undefined;

let HOLD_TIME = 100;
fetchSettings().then(settings => {
    HOLD_TIME = settings.holding_time * 1000; 
});
const activeDetections = new Map(); // name -> { lastSeen, detection }
let currData = [];

window.addEventListener("DOMContentLoaded", async () => {
    videoModal.classList.add("hidden");

    namelistPath = localStorage.getItem("namelistPath");
    if (namelistPath != null) {
        loadNamelistJSON(namelistPath).then((data) => {
            namelistJSON = data;
            console.log("loaded namelist")
            console.log(namelistJSON)
        });
    }

    startStream(() =>
        alert(`FFMPEG unable to stream from provided source!`)
    );
});

// ENTRY POINT: Check if stream is started and immediately loads video feed if it has
const startStream = (no_stream_callback = () => {}) => {
    fetch("/api/status")
        .then((response) => response.json())
        .then((data) => {
            if (data.stream_state === "running") {
                // Start detection overlays
                fetchDetections();
            } else {
                no_stream_callback();
            }
        })
        .catch((error) => console.log(error));
};


// ----------- Welcome page detections ------------

// Update country flag display
const updateCountryFlag = (detectionName) => {
    if (!detectionName || detectionName === "Unknown") {
        countryFlagImg.style.display = "none";
        return;
    }

    const flagPath = getCountryFlag(detectionName, namelistJSON);
    if (flagPath) {
        countryFlagImg.src = flagPath;
        countryFlagImg.style.display = "block";
    } else {
        countryFlagImg.style.display = "none";
    }
};

// MAIN LOOP
export const fetchDetections = () => {
    console.log("FETCHING...");
    let buffer = "";
    let data = [];

    fetch(`/api/frResults`).then((response) => {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        const processStream = () => {
            reader.read().then(({ done, value }) => {
                if (done) {
                    clearBBoxes(videoContainer);
                    detectionList.innerHTML = "";
                }

                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;

                const parts = buffer.split("\n");

                try {
                    if (parts.length > 1) {
                        data = JSON.parse(parts[parts.length - 2])?.data;
                        // console.log(data[0].label + "\r")
                    }
                } catch (err) {
                    console.log(buffer);
                    console.error("Error parsing JSON:", err);
                }

                buffer = parts[parts.length - 1];
                
                const videoContainer = document.getElementById("video-container");
                currData = updateBBoxes(videoContainer, data, { showLabels: true, showUnknown: true });
                updateDetections(data);

                // Recursive call
                processStream(); 
            });
        };

        processStream();
    });
};

export const endDetections = () => {
    currData = [];
    const detectionList = document.getElementById("detections-list");
    detectionList.innerHTML = "";
    clearBBoxes(videoContainer);

    // Clear the country flag
    countryFlagImg.style.display = "none";
};

const updateDetections = (data) => {
    const now = Date.now();

    // Update / refresh detections from stream
    data.forEach((detection) => {
        const name = detection.label.toUpperCase();
        if (name === "UNKNOWN") return;

        activeDetections.set(name, {
            lastSeen: now,
            detection
        });
    });

    // Remove expired detections
    for (const [name, entry] of activeDetections.entries()) {
        if (now - entry.lastSeen > HOLD_TIME) {
            activeDetections.delete(name);
        }
    }

    // Render from activeDetections
    let detections = [];

    for (const [name, entry] of activeDetections.entries()) {
        let description = getDescription(name, namelistJSON);

        const detectionEl = document.createElement("div");
        detectionEl.innerHTML = `<p class="detectionName">${name}</p> ${
            description === null
                ? ""
                : `<p class="detectionDesc">${description}</p>`
        }`;
        detectionEl.classList.add("detectionEntry");
        detectionEl.dataset.name = name; // For priority sorting

        detections.push(detectionEl);
    }

    // Sort detections and take top N
    detections = sortDetectionsByPriority(detections, namelistJSON);
    if (detections.length > N) 
        detections = detections.slice(-N);
    
    detectionList.replaceChildren(...detections);

    // Update country flag based on top detection
    const topDetection = detections[0]?.label;
    if (topDetection)
        updateCountryFlag(topDetection);
};

// -------- VIDEO MODAL STUFF ----------

// Handle resizing of modal
window.addEventListener("resize", () => {
    const videoContainer = document.getElementById("video-container");
    const bboxesEl = videoContainer.querySelectorAll(".bbox");
    bboxesEl.forEach((element, idx) => {
        setBBoxPos(
            element,
            currData[idx],
            videoContainer.offsetWidth,
            videoContainer.offsetHeight
        );
    });
});

const showVideoModal = () => {
    videoModal.classList.remove("hidden");
};

const hideVideoModal = () => {
    videoModal.classList.add("hidden");
};

// Handles taskbar button to open video modal
const openVideoModalButton = document.getElementById("open-video-modal-button");
if (openVideoModalButton) {
    openVideoModalButton.addEventListener("click", () => {
        const videoFeed = document.getElementById("video-feed");
        videoFeed.setAttribute("data", `/api/vidFeed?t=${Date.now()}`);

        showVideoModal();
    });
}

// Close video model button
document.getElementById("close-video-modal").addEventListener("click", (e) => {
    hideVideoModal();
    const videoFeed = document.getElementById("video-feed");
    videoFeed.removeAttribute("data");
});

// Close video modal on Escape key
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !videoModal.classList.contains("hidden")) {
        hideVideoModal();
        const videoFeed = document.getElementById("video-feed");
        videoFeed.removeAttribute("data");
    }
});
