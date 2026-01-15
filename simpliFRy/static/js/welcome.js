import {
    getCountryFlag,
    getDescription,
    setBBoxPos,
    clearBBoxes,
    updateBBoxes,
    loadNamelistJSON,
    sortDetectionsByPriority,
} from "./utils.js";

const N = 3; // number of detections shown (last N)

const detectionList = document.getElementById("detections-list");
const countryFlagImg = document.getElementById("country-flag-img");
const videoModal = document.getElementById("video-modal");
const videoContainer = document.getElementById("video-container");
let namelistPath = null;
let namelistJSON = undefined;
let currData = [];

window.addEventListener("DOMContentLoaded", async () => {
    videoModal.classList.add("hidden");

    namelistPath = localStorage.getItem("namelistPath");
    if (namelistPath != null) {
        loadNamelistJSON(namelistPath).then((data) => {
            namelistJSON = data;
            console.log("loaded namelist")
        });
    }

    startStream(() =>
        alert(`FFMPEG unable to stream from provided source!`)
    );
});

// ENTRY POINT: Check if stream is started and immediately loads video feed if it has
const startStream = (no_stream_callback = () => {}) => {
    fetch("/checkAlive")
        .then((response) => response.text())
        .then((data) => {
            if (data === "Yes") {
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

// Update detection list with new element
const addDetectionEl = (name, description) => {
    const detectionEl = document.createElement("div");
    detectionEl.innerHTML = `<p class="detectionName">${name}</p> ${
        description === null
            ? ""
            : `<p class="detectionDesc">${description}</p>`
    }`;
    detectionEl.classList.add("detectionEntry");
    detectionEl.dataset.name = name; // For priority sorting

    detectionList.appendChild(detectionEl);

    // show last N detections, sorted by priority
    if (detectionList.children.length > N) {
        detectionList.replaceChildren(
            ...sortDetectionsByPriority([...detectionList.children], namelistJSON).slice(-N)
        );
    } else {
        detectionList.replaceChildren(
            ...sortDetectionsByPriority([...detectionList.children], namelistJSON)
        );
    }
};

// MAIN LOOP
export const fetchDetections = () => {
    console.log("FETCHING...");
    let buffer = "";
    let data = [];

    fetch(`/frResults`).then((response) => {
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
                updateDetections(data);

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
    detectionList.innerHTML = "";
    const uniqueLabels = new Set();
    let mostRecentDetection = null;

    // Process detections for the list (non-bbox related)
    data.forEach((detection) => {
        const unknown = detection.label === "Unknown";

        if (!unknown && !uniqueLabels.has(detection.label)) {
            const description = getDescription(detection.label, namelistJSON);
            addDetectionEl(detection.label, description);
            uniqueLabels.add(detection.label);
        }

        // Track the last non-unknown detection as the most recent
        if (!unknown) {
            mostRecentDetection = detection.label;
        }
    });

    // Use optimized bbox update utility with labels shown
    currData = updateBBoxes(videoContainer, data, { showLabels: true, showUnknown: true });

    // Update country flag for the latest detection
    if (mostRecentDetection) {
        updateCountryFlag(mostRecentDetection);
    } else {
        // No identified detections in current list, hide flag
        countryFlagImg.style.display = "none";
    }
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
        videoFeed.setAttribute("data", `/vidFeed?t=${Date.now()}`);

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
