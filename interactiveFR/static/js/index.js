import {
    fetchStreamStatus,
    waitForEmbeddings,
    waitForStream,
} from "./utils.js";

const customRTSP = document.getElementById("stream_src_custom");
const cameraSelect = document.getElementById("camera_device_select");
let isEndingStream = false;

window.addEventListener("DOMContentLoaded", async () => {
    try {
        const status = await fetchStreamStatus();
        if (status.stream_state === "running" && status.embeddings_loaded) {
            window.location.href = "/interactive";
        }
    } catch (error) {
        console.log(error);
    }
});

const endStreamAndReload = async () => {
    if (isEndingStream) {
        return;
    }
    isEndingStream = true;
    try {
        await fetch("/end", { method: "POST" });
    } catch {}
    localStorage.removeItem("namelistPath");
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
    const streamSrc = resolveStreamSource();
    if (!streamSrc) {
        return;
    }

    const formData = new FormData(form);
    formData.set("stream_src", streamSrc);

    // Store stream source
    localStorage.setItem("streamSrc", streamSrc);

    const removeSubmitButton = () => {
        const submitButton = document.getElementById("submit-button");
        submitButton?.remove();
    };

    const addSubmitButton = () => {
        if (document.getElementById("submit-button")) {
            return;
        }
        const newSubmit = document.createElement("input");
        newSubmit.type = "submit";
        newSubmit.id = "submit-button";
        newSubmit.className = "submit-button";
        newSubmit.value = "Submit";
        form.appendChild(newSubmit);
    };

    removeSubmitButton();
    const loading = createLoadingManager(form);
    loading.start("Starting stream");
    let startRequestPending = true;
    loading.schedule(() => {
        if (startRequestPending) {
            loading.start("Loading embeddings");
        }
    }, 600);

    try {
        const response = await fetch("/start", {
            method: "POST",
            body: formData,
        });
        const data = await response.json();
        startRequestPending = false;
        loading.stop();

        if (!data.stream) {
            loading.remove();
            addSubmitButton();

            alert(data.message || "Failed to start stream.");
            return;
        }

        loading.start("Loading embeddings");
        const embeddingsStatus = await waitForEmbeddings();

        if (!embeddingsStatus.embeddings_loaded) {
            loading.remove();
            addSubmitButton();
            alert("Embedding load timed out. Please try again.");
            return;
        }

        loading.start("Verifying stream");
        const status = await waitForStream();

        loading.remove();

        if (status.stream_state === "running") {
            window.location.href = "/interactive";
            return;
        }

        addSubmitButton();
        alert(
            status.last_error
                ? `Stream failed (${status.stream_state}): ${status.last_error}`
                : `Stream failed (${status.stream_state}). Please check your source and try again.`,
        );
    } catch {
        startRequestPending = false;
        loading.remove();
        addSubmitButton();

        alert(`Error loading stream from ${streamSrc}. Please try again.`);
    }
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

const createLoadingManager = (formEl) => {
    let loader = formEl.querySelector(".loading-indicator");
    if (!loader) {
        loader = document.createElement("h4");
        loader.classList.add("loading-indicator");
        formEl.appendChild(loader);
    }

    let intervalId = null;
    let timerId = null;

    const stop = () => {
        if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
        }
        if (timerId) {
            clearTimeout(timerId);
            timerId = null;
        }
    };

    const start = (text) => {
        stop();
        intervalId = createLoadingAnimation(text, loader);
    };

    const schedule = (fn, delayMs) => {
        if (timerId) {
            clearTimeout(timerId);
        }
        timerId = setTimeout(() => {
            timerId = null;
            fn();
        }, delayMs);
    };

    const remove = () => {
        stop();
        loader?.remove();
        loader = null;
    };

    return {
        start,
        stop,
        schedule,
        remove,
    };
};

// Handles stream selection
const streamSelectElem = document.getElementById("stream_src_select");

const hideInput = (inputEl, { clear = true, required = false } = {}) => {
    inputEl.style.display = "none";
    inputEl.removeAttribute("name");
    if (required) {
        inputEl.removeAttribute("required");
    }
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
        const response = await fetch("/listCameras");
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
document
    .getElementById("reset-button")
    .addEventListener("click", async (event) => {
        event.preventDefault();
        endStreamAndReload();
    });
