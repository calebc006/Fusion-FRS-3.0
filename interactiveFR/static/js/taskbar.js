const taskbarPlaceholder = document.querySelector(".taskbar-placeholder");
const taskbar = document.querySelector(".taskbar");
const home_button = document.getElementById("home-button");
const settings_button = document.getElementById("settings-button");
const reset_button = document.getElementById("reset-button");

taskbarPlaceholder.addEventListener("mouseenter", () => {
    taskbar.classList.add("show");
});

taskbarPlaceholder.addEventListener("mouseleave", () => {
    taskbar.classList.remove("show");
});

home_button?.addEventListener("click", async (event) => {
    window.location.href = "/interactive";
});

settings_button?.addEventListener("click", async (event) => {
    window.location.href = "/settings";
});

reset_button?.addEventListener("click", async (event) => {
    event.preventDefault();
    try {
        await fetch("/end", { method: "POST" });
    } finally {
        window.location.href = "/";
    }
});
