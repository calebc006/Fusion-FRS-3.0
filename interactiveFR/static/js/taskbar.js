const taskbarPlaceholder = document.querySelector(".taskbar-placeholder");
const taskbar = document.querySelector(".taskbar");
const home_button = document.getElementById("home-button");
const settings_button = document.getElementById("settings-button");
const reset_button = document.getElementById("reset-button");

const fetchStreamStatus = async () => {
    const response = await fetch("/api/streamStatus");
    return response.json();
};

taskbarPlaceholder.addEventListener("mouseenter", () => {
    taskbar.classList.add("show");
});

taskbarPlaceholder.addEventListener("mouseleave", () => {
    taskbar.classList.remove("show");
});

home_button?.addEventListener("click", async (event) => {
    window.location.href = "/";
});

settings_button?.addEventListener("click", async (event) => {
    window.location.href = "/settings";
});

reset_button?.addEventListener("click", async (event) => {
    event.preventDefault();
    try {
        await fetch("/api/end", { method: "POST" });
    } catch {}

    try {
        await waitForStreamStop();
    } catch {}

    localStorage.removeItem("namelistPath");
    localStorage.removeItem("streamSrc");
    window.location.href = "/";
});

const waitForStreamStop = async ({ attempts = 6, delayMs = 500 } = {}) => {
    let lastStatus = null;
    for (let i = 0; i < attempts; i += 1) {
        lastStatus = await fetchStreamStatus();
        if (lastStatus.stream_state !== "running") {
            return lastStatus;
        }
        await new Promise((resolve) => setTimeout(resolve, delayMs));
    }
    return lastStatus;
};
