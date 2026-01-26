const customInput = document.getElementById("stream_src_custom");
const webcamInput = document.getElementById("webcam_device");
const form = document.getElementById("init");
const infoMenu = document.getElementById("info-menu");
let namelistPath = null;

window.addEventListener("DOMContentLoaded", async () => {
    fetch("/checkAlive")
        .then((response) => response.text())
        .then((data) => {
            if (data === "Yes") {
                // Hide form and show post-init menu
                form.style.display = "none";
                infoMenu.style.display = "flex";
                document.getElementById("stream-url").textContent =
                    localStorage.getItem("streamSrc") || "N/A";
                document.getElementById("namelist-path").textContent =
                    localStorage.getItem("namelistPath") || "N/A";
            } else {
                // Hide post-init menu and show form
                infoMenu.style.display = "none";
                form.style.display = "flex";
            }
        })
        .catch((error) => console.log(error));
});

// ------------ Init form ---------------

// Handles form submission (stream url and data file)
document.getElementById("init").onsubmit = async (event) => {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    // Set namelist path if provided
    const dataFile = formData.get("data_file");
    if (dataFile) {
        namelistPath = `./data/${dataFile}`;
        localStorage.setItem("namelistPath", namelistPath);
    } else {
        localStorage.removeItem("namelistPath");
    }

    // Store stream source
    const streamSrc = formData.get("stream_src");
    localStorage.setItem("streamSrc", streamSrc);

    // Remove submit button and create loading indicator
    const submitButton = document.getElementById("submit-button");
    submitButton.remove();

    const loader = document.createElement("h4");
    loader.classList.add("loading-indicator");
    let intervalId = createLoadingAnimation("Loading embeddings", loader);

    form.appendChild(loader);

    // Load embeddings then start stream
    fetch(`/start`, {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            // Replace loading animation with "Starting stream..."
            clearInterval(intervalId);
            intervalId = createLoadingAnimation("Starting stream", loader);

            if (data.stream) {
                console.log("Stream started!");

                // Hide form, loader and show post-init menu
                clearInterval(intervalId);
                loader.remove();
                form.style.display = "none";
                infoMenu.style.display = "flex";
                document.getElementById("stream-url").textContent =
                    localStorage.getItem("streamSrc") || "N/A";
                document.getElementById("namelist-path").textContent =
                    localStorage.getItem("namelistPath") || "N/A";
            } else {
                alert(data.message);
            }
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

        // Hide webcam input
        webcamInput.style.display = "none";
        webcamInput.removeAttribute("required");
        webcamInput.removeAttribute("name");
        webcamInput.value = "";
    } else if (this.value === "webcam") {
        // Hide the select's name to prevent duplicate keys
        streamSelectElem.removeAttribute("name");

        // Show webcam input (optional)
        webcamInput.style.display = "block";
        webcamInput.setAttribute("name", "stream_src");

        // Hide and disable the custom input
        customInput.style.display = "none";
        customInput.removeAttribute("required");
        customInput.removeAttribute("name");
        customInput.value = "";
    } else {
        // Restore name to select
        streamSelectElem.setAttribute("name", "stream_src");

        // Hide and disable the custom input
        customInput.style.display = "none";
        customInput.removeAttribute("required");
        customInput.removeAttribute("name");
        customInput.value = ""; // Clear out stale value

        // Hide and disable webcam input
        webcamInput.style.display = "none";
        webcamInput.removeAttribute("required");
        webcamInput.removeAttribute("name");
        webcamInput.value = "";
    }
});

// ------------ Drag/Drop File Handling ---------------

const dropZone = document.getElementById("drop-zone");
const fileButton = document.getElementById("file-button");
const dataFileInput = document.getElementById("data_file");
const fileNameDisplay = document.getElementById("file-name");
let selectedFileName = "";

// Handle file button click
fileButton.addEventListener("click", (e) => {
    e.preventDefault();
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFileSelection(file.name);
        }
    };
    input.click();
});

// Handle drag over
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.add("dragover");
});

// Handle drag leave
dropZone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");
});

// Handle drop
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropZone.classList.remove("dragover");

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.name.endsWith(".json")) {
            handleFileSelection(file.name);
        } else {
            alert("Please drop a .json file");
        }
    }
});

// Handle file selection
const handleFileSelection = (fileName) => {
    selectedFileName = fileName;
    dataFileInput.value = fileName;
    fileNameDisplay.textContent = `Selected: ${fileName}`;
};

// Optional: Form validation reminder
form.addEventListener("submit", function (e) {
    if (streamSelectElem.value === "custom" && !customInput.value.trim()) {
        e.preventDefault();
        alert("Please enter a valid custom RTSP URL.");
    }

    if (streamSelectElem.value === "webcam") {
        const deviceName = webcamInput.value.trim();
        webcamInput.value = deviceName ? `webcam:${deviceName}` : "webcam";
    }
});

// Handle taskbar button to end stream
document
    .getElementById("reset-button")
    .addEventListener("click", async (event) => {
        event.preventDefault();

        fetch("/end", {
            method: "POST",
        })
            .then((response) => response.json())
            .then((_data) => {
                localStorage.removeItem("namelistPath");
                localStorage.removeItem("streamSrc");
                location.reload();
            });
    });

// -------- VIDEO MODAL STUFF ----------
const videoModal = document.getElementById("video-modal");
const videoContainer = document.getElementById("video-container");

// Handle resizing of modal
window.addEventListener("resize", () => {
    const videoContainer = document.getElementById("video-container");
    const bboxesEl = videoContainer.querySelectorAll(".bbox");
    bboxesEl.forEach((element, idx) => {
        setBBoxPos(
            element,
            currData[idx],
            videoContainer.offsetWidth,
            videoContainer.offsetHeight,
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
