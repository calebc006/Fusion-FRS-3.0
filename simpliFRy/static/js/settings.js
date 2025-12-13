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
