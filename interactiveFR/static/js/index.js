import {
    waitForEmbeddings,
    waitForStream,
} from "./utils.js";

const customRTSP = document.getElementById("stream_src_custom");
const cameraSelect = document.getElementById("camera_device_select");

window.addEventListener("DOMContentLoaded", async () => {
    try {
        // if (localStorage.getItem("streamSrc") != null) {
        //     window.location.href = "/interactive";
        // }
    } catch (error) {
        console.log(error);
    }
});

const endStreamAndReload = async () => {
    try {
        await fetch("/api/end", { method: "POST" });
    } catch {}

    localStorage.removeItem("streamSrc");
    location.reload();
};

const resolveStreamSource = () => {
    const selection = streamSelectElem.value;

    if (selection === "custom") {
        const rtspValue = customRTSP.value.trim();
        if (!rtspValue || !rtspValue.toUpperCase().startsWith("RTSP://")) {
            alert("Please enter a valid custom RTSP URL.");
            return null;
        }
        return rtspValue;
    }

    if (selection === "camera") {
        const device = cameraSelect.value;
        if (!device) {
            alert("Please select a camera device.");
            return null;
        }
        return `camera:${device}`;
    }

    return streamSelectElem.value;
};

// ------------ Init form ---------------


// Handles form submission (stream url and data file)
document.getElementById("init").onsubmit = async (event) => {
    event.preventDefault();

    const form = event.target;
    const submitButton = document.getElementById("submit-button");

    const streamSrc = resolveStreamSource();
    if (!streamSrc) {
        return;
    }
    localStorage.setItem("streamSrc", streamSrc);

    submitButton.style.display = "none";
    const loading = Loading(form);
    loading.start("Starting stream");

    try {
        // Start stream
        let response = await fetch("/api/start_stream", {
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
        });
        data = await response.json();
        loading.stop();
        
        if (!data.inference) {
            loading.remove();
            submitButton.style.display = "block"

            alert(data.message || "Failed to start FR");
            return;
        }

        // Check that stream has started
        loading.start("Verifying stream");
        const status = await waitForStream();
        
        loading.remove();
        submitButton.style.display = "block"
        if (status.stream_state === "running") {
            window.location.href = "/interactive";
        } else {
            alert(
                status.last_error
                    ? `Stream failed (${status.stream_state}): ${status.last_error}`
                    : `Stream failed (${status.stream_state}). Please check your source and try again.`,
            );
        }

    } catch {
        loading.remove();
        submitButton.style.display = "block"

        alert(`Error loading stream from ${streamSrc}. Please reset and try again.`);
    }
};

// Handles loading animation (for dots)
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
            loaderEl.innerText = text + ".".repeat(dotCount);
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

const hideInput = (inputEl, { clear = true } = {}) => {
    inputEl.style.display = "none";
    inputEl.removeAttribute("name");
    inputEl.removeAttribute("required");
    if (clear) {
        inputEl.value = "";
    }
};

const showInput = (inputEl, { required = false } = {}) => {
    inputEl.style.display = "block";
    inputEl.setAttribute("name", "stream_src");
    if (required) {
        inputEl.setAttribute("required", "required");
    }
};

const inputConfigBySelection = {
    custom: { input: customRTSP, required: true },
    camera: { input: cameraSelect, required: true },
};

const fetchCameras = async () => {
    try {
        const response = await fetch("/api/listCameras");
        const cameras = await response.json();

        cameraSelect.innerHTML = "";

        if (cameras.length === 0) {
            cameraSelect.innerHTML =
                '<option value="" disabled selected>No cameras detected</option>';
            return false;
        }

        cameras.forEach((name) => {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            cameraSelect.appendChild(option);
        });
        return true;
    } catch (error) {
        console.log(error);
        cameraSelect.innerHTML =
            '<option value="" disabled selected>Error detecting cameras</option>';
        return false;
    }
};

let cameraPollTimer = null;

const stopCameraPolling = () => {
    if (cameraPollTimer) {
        clearTimeout(cameraPollTimer);
        cameraPollTimer = null;
    }
};

const startCameraPolling = async () => {
    stopCameraPolling();

    const poll = async () => {
        if (streamSelectElem.value !== "camera") {
            stopCameraPolling();
            return;
        }
        await fetchCameras();
        cameraPollTimer = setTimeout(poll, 500);
    };

    poll();
};

const updateStreamSourceUI = () => {
    const selection = streamSelectElem.value;

    hideInput(customRTSP, { required: true });
    hideInput(cameraSelect, { clear: false });

    const config = inputConfigBySelection[selection];
    if (config) {
        streamSelectElem.removeAttribute("name");
        showInput(config.input, { required: config.required });

        if (selection === "camera") {
            startCameraPolling();
        } else {
            stopCameraPolling();
        }
        return;
    }

    stopCameraPolling();

    streamSelectElem.setAttribute("name", "stream_src");
};

streamSelectElem.addEventListener("change", updateStreamSourceUI);
updateStreamSourceUI();

// Handle taskbar button to end stream
document.getElementById("reset-button").addEventListener("click", async (e) => {
    e.preventDefault();
    endStreamAndReload();
})
