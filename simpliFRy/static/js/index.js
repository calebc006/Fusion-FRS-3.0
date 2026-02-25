const customInput = document.getElementById("stream_src_custom");
const form = document.getElementById("init");
const postInitMenu = document.getElementById("info-menu");
let namelistPath = null;

window.addEventListener("DOMContentLoaded", async () => {
    // Show form and hide post-init menu
    postInitMenu.style.display = "none";
    form.style.display = "flex";

    // Check if initialized
    if (localStorage.getItem("initialized") === "true") {
        // Hide form and show post-init menu 
        form.style.display = "none";
        postInitMenu.style.display = "flex";
        document.getElementById("stream-url").textContent = localStorage.getItem("streamSrc") || "N/A";
        document.getElementById("namelist-path").textContent = localStorage.getItem("namelistPath") || "N/A";
    }
});


// ------------ Init form ---------------

// Handles form submission (stream url and data file)
document.getElementById("init").onsubmit = async (event) => {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    // Validate data file is provided
    const dataFile = formData.get("data_file");
    if (!dataFile || !dataFile.trim()) {
        alert("Please select a JSON namelist file.");
        return;
    }

    // Store namelist path as-is
    namelistPath = dataFile;
    localStorage.setItem("namelistPath", namelistPath);

    // Store stream source
    const streamSrc = formData.get("stream_src");
    localStorage.setItem("streamSrc", streamSrc);

    // Remove submit button and create loading indicator
    const submitButton = document.getElementById("submit-button");
    submitButton.style.display = "none";

    const loading = Loading(form);
    loading.start("Starting stream");

    try {
        // Start stream
        let response = await fetch(`/api/start_stream`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({"stream_src": streamSrc}),
        });
        let data = await response.json();
        loading.stop();

        if (!data.stream) {
            loading.remove();
            submitButton.style.display = "block";
            alert(data.message || "Failed to start stream");
            return;
        }

        // Start FR
        loading.start("Loading embeddings");
        response = await fetch("/api/start_fr", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({"data_file": namelistPath}),
        });
        data = await response.json();
        loading.stop();
        
        if (!data.inference) {
            loading.remove();
            submitButton.style.display = "block";

            alert(data.message || "Failed to start FR");
            return;
        }

        // Success! 
        // Hide form, loader and show post-init menu 
        submitButton.style.display = "block";
        form.style.display = "none";
        postInitMenu.style.display = "flex";
        document.getElementById("stream-url").textContent = localStorage.getItem("streamSrc") || "N/A";
        document.getElementById("namelist-path").textContent = localStorage.getItem("namelistPath") || "N/A";

        localStorage.setItem("initialized", true);

    } catch (error) {
        console.log(error)
        loading.remove();
        submitButton.style.display = "block"
        localStorage.setItem("initialized", false);

        alert(`Error loading stream from ${streamSrc}. Please reset and try again.`);
    }
};

const Loading = (formEl) => {
    let loader = formEl.querySelector(".loading-indicator");
    if (!loader) {
        loader = document.createElement("h4");
        loader.classList.add("loading-indicator");
        formEl.appendChild(loader);
    }

    let intervalId = null;

    const stop = () => {
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
        }
    };

    const start = (text) => {
        stop();

        let dotCount = 0;
        const updateLoadingText = () => {
            dotCount = (dotCount % 3) + 1;
            loader.innerText = text + ".".repeat(dotCount);
        };

        intervalId =  setInterval(updateLoadingText, 500);
    }

    const remove = () => {
        stop();
        loader?.remove();
        loader = null;
    };

    return {
        start,
        stop,
        remove,
    };
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

// Handle file selection - stores full relative path
const handleFileSelection = (fileName) => {
    const fullPath = `data/${fileName}`;
    selectedFileName = fullPath;
    dataFileInput.value = fullPath;
    fileNameDisplay.textContent = `Selected: ${fileName}`;
};

// Form validation
form.addEventListener("submit", function (e) {
    if (streamSelectElem.value === "custom" && !customInput.value.trim()) {
        e.preventDefault();
        alert("Please enter a valid custom RTSP URL.");
        return;
    }
    if (!dataFileInput.value.trim()) {
        e.preventDefault();
        alert("Please select a JSON namelist file.");
        return;
    }
});

// Handle taskbar button to end stream
document
    .getElementById("reset-button")
    .addEventListener("click", async (event) => {
        event.preventDefault();
        localStorage.setItem("initialized", false);

        fetch("/api/end", {
            method: "POST",
        })
            .then((response) => response.json())
            .then((_data) => {
                localStorage.removeItem("namelistPath");
                localStorage.removeItem("streamSrc");
                location.reload();
            })
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
