import {
    updateBBoxes,
    loadNamelistJSON,
    getTable,
    getDescription,
    sortDetectionsByPriority,
    fetchSettings 
} from "./utils.js";

const detectionList = document.getElementById("table-detection-list");
let namelistJSON = undefined;

let HOLD_TIME = 100;
fetchSettings().then(settings => {
    HOLD_TIME = settings.holding_time * 1000; 
});
const activeDetections = new Map(); // name -> { lastSeen, detection }

window.addEventListener("DOMContentLoaded", () => {
    document.getElementById("video-feed").setAttribute("data", `/api/vidFeed?t=${Date.now()}`);
    let namelistPath = localStorage.getItem("namelistPath");

    loadNamelistJSON(namelistPath).then((data) => {
        namelistJSON = data;
        fetchDetections();
    });
});

// MAIN LOOP
const fetchDetections = () => {
    console.log("FETCHING...");
    let buffer = "";
    let data = [];

    fetch(`/api/frResults`)
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
                            data =
                                JSON.parse(parts[parts.length - 2])?.data || [];
                        }
                    } catch (err) {
                        console.error("Error parsing JSON:", err);
                    }

                    buffer = parts[parts.length - 1] || "";

                    const videoContainer = document.getElementById("video-container");
                    updateBBoxes(videoContainer, data, { showLabels: false, showUnknown: true });
                    updateDetectionList(data);

                    // Recursive call
                    processStream();
                });
            };

            processStream();
        })
        .catch((error) => {
            console.error("Error fetching detections:", error);
            setTimeout(() => fetchDetections(), 5000);
        });
};

const updateDetectionList = (data) => {
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
        let table = getTable(name, namelistJSON);
        table = table ? `(${table})` : "";

        let description = getDescription(name, namelistJSON);

        let detectionEl = document.createElement("div");
        detectionEl.classList.add("table-detection-element");
        detectionEl.dataset.name = name;

        detectionEl.innerHTML = `
            <span class="detection-name">${name} ${table}</span>
            ${description ? `<span class="detection-desc">${description}</span>` : ""}
        `;

        detections.push(detectionEl);
    }

    detections = sortDetectionsByPriority(detections, namelistJSON);
    detectionList.replaceChildren(...detections);
};