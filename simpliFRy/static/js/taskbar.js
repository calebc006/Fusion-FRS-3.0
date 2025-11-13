const taskbarPlaceholder = document.querySelector(".taskbar-placeholder")
const taskbar = document.querySelector(".taskbar")

taskbarPlaceholder.addEventListener("mouseenter", () => {
    taskbar.classList.add("show")
})

taskbarPlaceholder.addEventListener("mouseleave", () => {
    taskbar.classList.remove("show")
})