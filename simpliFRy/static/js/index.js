import {
    getCountryFlag,
    getDescription,
    setBBoxPos,
    clearBBoxes,
    loadNamelistJSON,
    delay,
} from "./utils.js";

const N = 3; // number of detections shown (last N)

const customInput = document.getElementById("stream_src_custom");
const detectionList = document.getElementById("detections-list");
const countryFlagImg = document.getElementById("country-flag-img");
const videoModal = document.getElementById("video-modal");
const videoContainer = document.getElementById("video-container");
const form = document.getElementById("init");
let namelistPath = null;
let namelistJSON = undefined;
let currData = [];

window.addEventListener("DOMContentLoaded", async () => {
    document.getElementById("main-container").style.display = "none";
    videoModal.classList.add("hidden");

    namelistPath = localStorage.getItem("namelistPath");
    if (namelistPath != null) {
        loadNamelistJSON(namelistPath).then((data) => {
            namelistJSON = data;
        });
    }

    startStream();
});

// ENTRY POINT: Check if stream is started and immediately loads video feed if it has
const startStream = (no_stream_callback = () => {}) => {
    fetch("/checkAlive")
        .then((response) => response.text())
        .then((data) => {
            if (data === "Yes") {
                const form = document.getElementById("init");

                // Hide the form
                form.style.display = "none";

                // Show the main UI
                const mainContainer = document.getElementById("main-container");
                mainContainer.style.display = "flex";

                // Start detection overlays
                console.log(namelistJSON);
                fetchDetections();
            } else {
                no_stream_callback();
            }
        })
        .catch((error) => console.log(error));
};

// ------------ Init form ---------------

// Handles form submission (stream url and data file)
document.getElementById("init").onsubmit = async (event) => {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    // Remove submit button and create loading indicator
    const submitButton = document.getElementById("submit-button");
    submitButton.remove();

    const loader = document.createElement("h4");
    loader.classList.add("loading-indicator");
    let intervalId = createLoadingAnimation("Loading embeddings", loader);

    form.appendChild(loader);

    // Remove loader and put back submit button
    const reset_button = (loading_intervalId) => {
        clearInterval(loading_intervalId);
        form.appendChild(submitButton);
        loader.remove();
    };

    // Load namelist
    namelistPath = `./data/${formData.get("data_file")}`;
    localStorage.setItem("namelistPath", namelistPath);
    loadNamelistJSON(namelistPath).then((data) => {
        namelistJSON = data;
    });

    // Starts stream
    fetch(`/start`, {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            clearInterval(intervalId);
            intervalId = createLoadingAnimation("Starting stream", loader);

            if (data.stream)
                delay(5000).then(() => {
                    startStream(() =>
                        alert(`FFMPEG unable to stream from provided source!`)
                    );
                    reset_button(intervalId);
                });
            else {
                alert(data.message);
                reset_button(intervalId);
            }
        })
        .catch((error) => {
            console.log(error);
            reset_button(intervalId);
        });
};

// Handles loading animation (for dots)
const createLoadingAnimation = (text, loaderEl) => {
    let dotCount = 0;
    const updateLoadingText = () => {
        dotCount = (dotCount % 3) + 1;
        loaderEl.innerText = text + ".".repeat(dotCount);
    };

    return setInterval(updateLoadingText, 500);
};

// Handles stream selection
const streamSelectElem = document.getElementById("stream_src_select");
streamSelectElem.addEventListener("change", function () {
    if (this.value === "custom") {
        // Hide the select's name to prevent duplicate keys
        streamSelectElem.removeAttribute("name");

        // Show and enable the custom input
        customInput.style.display = "block";
        customInput.setAttribute("required", "required");
        customInput.setAttribute("name", "stream_src");
    } else {
        // Restore name to select
        streamSelectElem.setAttribute("name", "stream_src");

        // Hide and disable the custom input
        customInput.style.display = "none";
        customInput.removeAttribute("required");
        customInput.removeAttribute("name");
        customInput.value = ""; // Clear out stale value
    }
});

// Optional: Form validation reminder
form.addEventListener("submit", function (e) {
    if (streamSelectElem.value === "custom" && !customInput.value.trim()) {
        e.preventDefault();
        alert("Please enter a valid custom RTSP URL.");
    }
});

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
    // detectionEl.innerHTML = `<p class="detectionName">${name}</p>`;
    detectionEl.classList.add("detectionEntry");

    detectionList.appendChild(detectionEl);

    // show last N detections
    if (detectionList.children.length > N) {
        detectionList.replaceChildren(
            ...sortDetections([...detectionList.children]).slice(-N)
        );
    } else {
        detectionList.replaceChildren(
            ...sortDetections([...detectionList.children])
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

// takes in an array of HTML detection elements and returns a sorted list
const sortDetections = (detectionList) => {
    return detectionList.sort((a, b) => a.innerText.localeCompare(b.innerText));
};

const updateDetections = (data) => {
    currData = [];
    detectionList.innerHTML = "";
    clearBBoxes(videoContainer);
    const uniqueLabels = new Set();
    let mostRecentDetection = null;

    // Process detections in order of detection (no sorting)
    data.forEach((detection) => {
        const unknown = detection.label === "Unknown";

        // if you want to hide unknown bboxes
        // if (unknown) {
        //   return;
        // }

        if (!unknown && !uniqueLabels.has(detection.label)) {
            const description = getDescription(detection.label, namelistJSON);
            addDetectionEl(detection.label, description);
            uniqueLabels.add(detection.label);
        }

        // Track the last non-unknown detection as the most recent
        if (!unknown) {
            mostRecentDetection = detection.label;
        }

        if (!detection.bbox) return;

        const bboxEl = document.createElement("div");
        bboxEl.classList.add("bbox");
        if (!unknown) {
            bboxEl.classList.add("bbox-identified");
        }

        // old UI for blue and red boxes
        bboxEl.innerHTML = `<p class="bbox-label${
            unknown ? "" : " bbox-label-identified"
        }">${
            detection.label
        } <span class="bbox-score">${detection.score.toFixed(2)}</span></p>`;

        // new UI with all blue boxes
        // bboxEl.innerHTML = `<p class="bbox-label${" bbox-label-identified"}"><span class="bbox-score"></span></p>`;

        currData.push(detection.bbox);
        setBBoxPos(
            bboxEl,
            detection.bbox,
            videoContainer.offsetWidth,
            videoContainer.offsetHeight
        );
        videoContainer.appendChild(bboxEl);
    });

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

// Handles taskbar button to end stream
document
    .getElementById("end_stream_button")
    .addEventListener("click", async (event) => {
        event.preventDefault();

        endDetections();
        console.log("Detections Ended")
        
        document.getElementById("video-feed").removeAttribute("data");
        console.log("Video Feed Down")
        
        fetch("/end", {
            method: "POST",
        })
        .then((response) => response.json())
        .then((_data) => {
            document.getElementById("main-container").style.display =
            "none";
            document.getElementById("init").style.display = "flex";
            console.log("Reset to init form")
        })
        .catch((error) => {
            console.log(error);
        });

        location.reload();
    });

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
