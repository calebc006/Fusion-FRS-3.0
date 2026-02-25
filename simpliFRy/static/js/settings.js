import { showToast } from "./utils.js";

const toast = document.getElementById("toast");
const form = document.getElementById("settings-form");

// Submit handler
form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(form);

    const response = await fetch("/api/submit_settings", {
        method: 'POST', 
        body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
        showToast(toast, data.message, "error", 1500);
    } else {
        showToast(toast, "Settings Updated", "success", 1500);
    }
});

const formSections = ["differentiator", "persistor"];

const addToggler = (form_section) => {
    // Disables other inputs in a form section when user indicates that he/she isn't making use of the mechanic

    const checkbox = document.getElementById(`use_${form_section}`);
    const inputs = document.querySelectorAll(`.${form_section}-inputs`);

    const toggleDisable = () => {
        inputs.forEach((input) => {
            input.disabled = !checkbox.checked;
        });
    };

    checkbox.addEventListener("change", toggleDisable);

    toggleDisable();
};

formSections.forEach((form_section) => {
    addToggler(form_section);
});