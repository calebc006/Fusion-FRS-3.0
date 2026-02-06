const customRTSP = document.getElementById("stream_src_custom");
const cameraSelect = document.getElementById("camera_device_select");
const form = document.getElementById("init");
let isEndingStream = false;

window.addEventListener("DOMContentLoaded", async () => {
    fetch("/checkAlive")
        .then((response) => response.text())
        .then((data) => {
            if (data === "Yes") {
                window.location.href = "/interactive";
            }
        })
        .catch((error) => console.log(error));
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

    // Remove submit button and create loading indicator
    const submitButton = document.getElementById("submit-button");
    submitButton.remove();

    const loader = document.createElement("h4");
    loader.classList.add("loading-indicator");

    form.appendChild(loader);

    let addSubmitButton = () => {
        const newSubmit = document.createElement("input");
        newSubmit.type = "submit";
        newSubmit.id = "submit-button";
        newSubmit.className = "submit-button";
        newSubmit.value = "Submit";
        form.appendChild(newSubmit);
    };

    // Load embeddings then start stream
    fetch(`/start`, {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            // Create loading animation with "Starting stream..."
            let intervalId = createLoadingAnimation("Starting stream", loader);

            if (data.stream) {
                console.log("Stream started!");

                // Brief delay then verify stream is still alive before redirecting
                clearInterval(intervalId);
                intervalId = createLoadingAnimation("Verifying stream", loader);

                setTimeout(async () => {
                    try {
                        const aliveRes = await fetch("/checkAlive");
                        const alive = await aliveRes.text();
                        clearInterval(intervalId);
                        loader.remove();

                        if (alive === "Yes") {
                            form.style.display = "none";
                            window.location.href = "/interactive";
                        } else {
                            alert(
                                "Stream failed shortly after starting. Please check your source and try again.",
                            );
                            addSubmitButton();
                        }
                    } catch {
                        clearInterval(intervalId);
                        loader.remove();
                        addSubmitButton();
                    }
                }, 1500);
            } else {
                clearInterval(intervalId);
                loader.remove();
                alert(data.message);

                // Re-add the submit button so user can retry
                addSubmitButton();
            }
        })
        .catch(() => {
            alert(`Error loading stream from ${streamSrc}. Please try again.`);

            // Re-add the submit button so user can retry
            addSubmitButton();
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
    cameraSelect.innerHTML =
        '<option value="" disabled selected>Detecting cameras...</option>';
    try {
        const response = await fetch("/listCameras");
        const cameras = await response.json();

        cameraSelect.innerHTML = "";

        if (cameras.length === 0) {
            cameraSelect.innerHTML =
                '<option value="" disabled selected>No cameras detected</option>';
            return;
        }

        cameras.forEach((name) => {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            cameraSelect.appendChild(option);
        });
    } catch (error) {
        console.log(error);
        cameraSelect.innerHTML =
            '<option value="" disabled selected>Error detecting cameras</option>';
    }
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
            fetchCameras();
        }
        return;
    }

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
