import {
    updateBBoxes,
    loadNamelistJSON,
    getTable,
} from "./utils.js";

const detectionList = document.getElementById("table-detection-list");
let namelistJSON = undefined;
let currData = [];

window.addEventListener("DOMContentLoaded", () => {
    document.getElementById("video-feed").setAttribute("data", `/vidFeed?t=${Date.now()}`);
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

    fetch(`/frResults`)
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
                    currData = updateBBoxes(videoContainer, data, { showLabels: false, showUnknown: true });
                    
                    updateDetectionList(data);
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
    let detections = [];

    data.forEach((detection) => {
        const name = detection.label.toUpperCase();
        if (name == "UNKNOWN") {
            return;
        }

        let table = getTable(name, namelistJSON);
        if (table == null) {
            table = "";
        } else {
            table = "(" + table + ")"; 
        }

        let detectionEl = document.createElement("div");
        detectionEl.classList.add("table-detection-element");
        detectionEl.innerHTML = `${name} ${table}`;

        detections.push(detectionEl);
    });

    detections = sortDetections(detections);
    detectionList.replaceChildren(...detections);
};

const sortDetections = (detectionList) => {
    return detectionList.sort((a, b) => a.innerText.localeCompare(b.innerText));
};
