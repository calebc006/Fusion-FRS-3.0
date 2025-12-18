import {
    setBBoxPos,
    clearBBoxes,
    loadNamelistJSON,
    getTable,
} from "./utils.js";

let namelistJSON = null;


window.addEventListener("DOMContentLoaded", () => {
    makeMenuDraggable("table-menu", "table-menu-header");
    loadTablesFromStorage();

    let namelistPath = localStorage.getItem("namelistPath");
    loadNamelistJSON(namelistPath).then((data) => {
        namelistJSON = data;
        fetchDetections();
    });

    // restore state of lock toggle
    const lockToggle = document.getElementById("lock-tables");
    lockToggle.checked = localStorage.getItem("locked") === "true";
});

const toggleSeatingsButton = document.getElementById("toggle-seating-button");

toggleSeatingsButton.addEventListener("click", (e) => {
    const menu = document.getElementById("table-menu");
    menu.style.display = menu.style.display === "none" ? "block" : "none";

    // Undo on Ctrl+Z or Cmd+Z
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();

        if (historyIndex > 0) {
            historyIndex--;
            const prevState = historyStack[historyIndex];
            restoreState(prevState);
        }
    }

    // Redo on Ctrl+Y or Cmd+Y
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "y") {
        e.preventDefault();

        if (historyIndex < historyStack.length - 1) {
            historyIndex++;
            const nextState = historyStack[historyIndex];
            restoreState(nextState);
        }
    }
});

let historyStack = [];
let historyIndex = -1; // points to current state in history

const pushHistory = () => {
    // Get current state snapshot
    const boxes = document.querySelectorAll("#seatings-container .box");
    const currentState = Array.from(boxes).map((box) => ({
        id: box.classList[1],
        label: box.querySelector(".box-label")?.innerText || "",
        x: box.offsetLeft,
        y: box.offsetTop,
        color: box.style.backgroundColor,
        width: box.offsetWidth,
        height: box.offsetHeight,
    }));

    // If we have undone some steps and then make a new change,
    // discard redo history
    historyStack = historyStack.slice(0, historyIndex + 1);

    historyStack.push(currentState);
    historyIndex++;

    // Optional: limit history size to last 50 states
    if (historyStack.length > 50) {
        historyStack.shift();
        historyIndex--;
    }

    console.log("Current Tables: ", historyStack[historyStack.length - 1]);
};

const restoreState = (state) => {
    const container = document.getElementById("seatings-container");
    if (!container) return;
    // Remove all existing boxes
    container.querySelectorAll(".box").forEach((el) => el.remove());

    boxCount = state.length;

    // Recreate boxes from the state snapshot
    state.forEach(({ id, x, y, label, color, width, height }) => {
        const newBox = document.createElement("div");
        newBox.className = `box ${id}`;
        newBox.id = id;
        newBox.innerHTML = `<div class="box-label">${label}</div>`;
        newBox.style.left = `${x}px`;
        newBox.style.top = `${y}px`;
        newBox.style.backgroundColor = color;
        newBox.style.width = `${width}px`;
        newBox.style.height = `${height}px`;

        container.appendChild(newBox);
        makeDraggableBox(newBox);
    });

    saveTablesToStorage();
};

// TABLE MANAGEMENT
let boxCount = 0;

const loadTablesFromStorage = () => {
    try {
        const savedTables = JSON.parse(localStorage.getItem("tables") || "[]");
        boxCount = savedTables.length;
        savedTables.forEach(({ id, x, y, label, color, width, height }) => {
            const newBox = document.createElement("div");
            newBox.className = `box ${id}`;
            newBox.id = id;
            newBox.innerHTML = `<div class="box-label">${label}</div>`;
            newBox.style.left = `${x}px`;
            newBox.style.top = `${y}px`;
            newBox.style.backgroundColor = color;
            newBox.style.width = `${width}px`;
            newBox.style.height = `${height}px`;
            const container = document.getElementById("seatings-container");
            if (container) {
                container.appendChild(newBox);
            }
            makeDraggableBox(newBox);
        });
    } catch (error) {
        console.error("Error loading tables from storage:", error);
    }
};

const saveTablesToStorage = () => {
    try {
        const boxes = document.querySelectorAll("#seatings-container .box");
        const tableData = Array.from(boxes).map((box) => {
            return {
                id: box.classList[1],
                label: box.querySelector(".box-label")?.innerText || "",
                x: box.offsetLeft,
                y: box.offsetTop,
                color: box.style.backgroundColor,
                width: box.offsetWidth,
                height: box.offsetHeight,
            };
        });
        localStorage.setItem("tables", JSON.stringify(tableData));
    } catch (error) {
        console.error("Error saving tables to storage:", error);
    }
};

const randomColor = () => {
    const colors = [
        "#e6194b",
        "#3cb44b",
        "#ffe119",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#46f0f0",
    ];
    // const colors = ["#e6194b"];
    return colors[Math.floor(Math.random() * colors.length)];
};

const updateBoxAnimations = (detectedLabels) => {
    const boxes = document.querySelectorAll("#seatings-container .box");

    boxes.forEach((box) => {
        const boxName = box.innerText.trim().toLowerCase();

        if (!boxName) {
            box.classList.remove("animate-pulse");
            return;
        }

        const isMatch = detectedLabels.some((label) =>
            label.toLowerCase().includes(boxName)
        );

        if (isMatch) {
            box.classList.add("animate-pulse");
        } else {
            box.classList.remove("animate-pulse");
        }
    });
};

const createDeleteBtn = (box) => {
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "delete-btn";
    deleteBtn.innerHTML = "×";
    deleteBtn.title = "Delete";

    deleteBtn.addEventListener("click", (e) => {
        e.stopPropagation(); // Prevent triggering drag
        box.remove();
        saveTablesToStorage();
        pushHistory();
    });

    box.appendChild(deleteBtn);
};

const createResizeSlider = (box) => {
    const sliderWrapper = document.createElement("div");
    sliderWrapper.className = "resize-slider-wrapper";
    sliderWrapper.style.display = "flex";

    const label = document.createElement("span");
    label.className = "slider-label";
    label.textContent = `${box.offsetWidth}px`;

    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = 80;
    slider.max = 260;
    slider.value = box.offsetWidth;
    slider.step = 10;
    slider.className = "resize-slider";

    // Prevent drag when interacting with the slider
    slider.addEventListener("mousedown", (e) => e.stopPropagation());

    slider.addEventListener("input", () => {
        const size = `${slider.value}px`;
        box.style.width = size;
        box.style.height = size;
        label.textContent = size;
        saveTablesToStorage();
        pushHistory();
    });

    sliderWrapper.appendChild(slider);
    sliderWrapper.appendChild(label);
    box.appendChild(sliderWrapper);
};

const makeDraggableBox = (box) => {
    box.style.position = "absolute";
    createResizeSlider(box);
    createDeleteBtn(box);

    // Respect current lock state
    box.style.pointerEvents =
        localStorage.getItem("locked") === "true" ? "none" : "auto";

    // Enable renaming, show slider and delete-btn when clicking on the table name
    box.addEventListener("click", (e) => {
        const resizeSlider = box
            .getElementsByClassName("resize-slider-wrapper")[0]
            .querySelector("input");
        const deleteBtn = box.getElementsByClassName("delete-btn")[0];
        resizeSlider.style.display = "block";
        deleteBtn.style.display = "block";

        if (localStorage.getItem("locked") === "true") return; // Prevent renaming if locked

        const labelEl = box.querySelector(".box-label");
        const label = labelEl?.innerText.trim() || "";
        const inputField = document.createElement("input");
        inputField.value = label;
        inputField.classList.add("rename-input");

        // Replace the box label with input field for renaming
        labelEl.innerHTML = "";
        labelEl.appendChild(inputField);
        inputField.focus();

        // On blur (click outside)
        inputField.addEventListener("blur", (e) => {
            const newName = inputField.value.trim();
            if (newName !== label && newName !== "") {
                labelEl.innerText = newName;
                saveTablesToStorage(); // Save the updated name to localStorage
                pushHistory();
            } else {
                labelEl.innerText = label;
            }
        });

        // Allow renaming via Enter key
        inputField.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                const newName = inputField.value.trim();
                if (newName !== label && newName !== "") {
                    labelEl.innerText = newName;
                    saveTablesToStorage(); // Save the updated name to localStorage
                    pushHistory();
                } else {
                    labelEl.innerText = label;
                }
            }
        });
    });

    // Remove deletebtn and resizeSlider when clicking out
    document.addEventListener("click", (e) => {
        const resizeSlider = box
            .getElementsByClassName("resize-slider-wrapper")[0]
            .querySelector("input");
        const deleteBtn = box.getElementsByClassName("delete-btn")[0];
        if (!resizeSlider.contains(e.target) && !box.contains(e.target)) {
            resizeSlider.style.display = "none";
            deleteBtn.style.display = "none";
        }
    });

    // dragging logic
    box.addEventListener("mousedown", (e) => {
        if (localStorage.getItem("locked") === "true") return; // Prevent drag if locked

        let offsetX = e.clientX - box.offsetLeft;
        let offsetY = e.clientY - box.offsetTop;
        box.style.zIndex = 1000;

        const onMouseMove = (e) => {
            box.style.left = `${e.clientX - offsetX}px`;
            box.style.top = `${e.clientY - offsetY}px`;
        };

        const onMouseUp = () => {
            document.removeEventListener("mousemove", onMouseMove);
            document.removeEventListener("mouseup", onMouseUp);
            saveTablesToStorage();
            pushHistory();
        };

        document.addEventListener("mousemove", onMouseMove);
        document.addEventListener("mouseup", onMouseUp);
    });
};

document.getElementById("lock-tables").addEventListener("change", () => {
    const boxes = document.querySelectorAll("#seatings-container .box");
    localStorage.setItem(
        "locked",
        localStorage.getItem("locked") === "true" ? "false" : "true"
    );
    const locked = localStorage.getItem("locked") === "true";

    boxes.forEach((box) => {
        box.style.pointerEvents = locked ? "none" : "auto";

        const deleteBtn = box.querySelector(".delete-btn");
        if (deleteBtn) {
            deleteBtn.style.display = locked ? "none" : "block";
        }

        const resizeSlider = box
            .getElementsByClassName("resize-slider-wrapper")[0]
            .querySelector("input");
        if (resizeSlider) {
            resizeSlider.style.display = locked ? "none" : "block";
        }
    });
});

document.getElementById("close-menu").addEventListener("click", () => {
    document.getElementById("table-menu").style.display = "none";
});

document.getElementById("add-table").addEventListener("click", () => {
    boxCount += 1;
    const newBox = document.createElement("div");
    newBox.className = `box box${boxCount}`;
    newBox.id = `box${boxCount}`;
    newBox.innerHTML = `<div class="box-label">T${boxCount}</div>`;
    newBox.style.backgroundColor = randomColor();
    const container = document.getElementById("seatings-container");
    if (container) {
        container.appendChild(newBox);
    }
    makeDraggableBox(newBox);
    saveTablesToStorage();
    pushHistory();
});

document.getElementById("remove-table").addEventListener("click", () => {
    if (boxCount > 0) {
        const lastBox = document.querySelector(`.box${boxCount}`);
        if (lastBox) lastBox.remove();
        boxCount -= 1;
        saveTablesToStorage();
        pushHistory();
    }
});

document.getElementById("reset-tables")?.addEventListener("click", () => {
    document
        .querySelectorAll("#seatings-container .box")
        .forEach((el) => el.remove());
    boxCount = 0;
    saveTablesToStorage();
    pushHistory();
});

const colorPicker = document.getElementById("box-color-picker");

function rgbToHex(rgb) {
    const result = rgb.match(/\d+/g).map(Number);
    return (
        "#" +
        result
            .slice(0, 3)
            .map((x) => x.toString(16).padStart(2, "0"))
            .join("")
    );
}

function makeMenuDraggable(menuId, handleId) {
    const menu = document.getElementById(menuId);
    const handle = document.getElementById(handleId);

    let offsetX = 0,
        offsetY = 0,
        isDragging = false;

    handle.addEventListener("mousedown", (e) => {
        isDragging = true;
        offsetX = e.clientX - menu.offsetLeft;
        offsetY = e.clientY - menu.offsetTop;
        document.addEventListener("mousemove", moveMenu);
        document.addEventListener("mouseup", stopDragging);
    });

    function moveMenu(e) {
        if (!isDragging) return;
        menu.style.left = `${e.clientX - offsetX}px`;
        menu.style.top = `${e.clientY - offsetY}px`;
    }

    function stopDragging() {
        isDragging = false;
        document.removeEventListener("mousemove", moveMenu);
        document.removeEventListener("mouseup", stopDragging);
    }
}

const seatingsContainer = document.getElementById("seatings-container");
if (seatingsContainer !== null) {
    seatingsContainer.addEventListener("contextmenu", (e) => {
        const targetBox = e.target.closest(".box");
        if (!targetBox) return;

        e.preventDefault();

        colorPicker.style.left = `${e.pageX}px`;
        colorPicker.style.top = `${e.pageY}px`;
        colorPicker.style.display = "block";

        const currentColor = rgbToHex(
            getComputedStyle(targetBox).backgroundColor
        );
        colorPicker.value = currentColor;

        const applyColor = (event) => {
            targetBox.style.backgroundColor = event.target.value;
            saveTablesToStorage();
            pushHistory();
            colorPicker.style.display = "none";
            colorPicker.removeEventListener("input", applyColor);
        };

        colorPicker.addEventListener("input", applyColor);
    });
}

document.addEventListener("click", (e) => {
    if (e.target !== colorPicker) {
        colorPicker.style.display = "none";
    }
});

// ------------ LIGHTING FUNCTIONALITY --------------

// MAIN LOOP
const fetchDetections = () => {
    console.log("FETCHING...");
    let buffer = "";
    let data = [];

    fetch(`/frResults`)
        .then((response) => {
            if (!response.ok || !response.body) {
                console.error("Fetch failed, retrying...");
                setTimeout(() => fetchDetections(), 5000);
                return;
            }

            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            const processStream = () => {
                reader.read().then(({ done, value }) => {
                    if (done) {
                        console.log("Stream ended, reconnecting...");
                        setTimeout(() => fetchDetections(), 2000);
                        return;
                    }

                    const chunk = decoder.decode(value, { stream: true });
                    buffer += chunk;

                    const parts = buffer.split("\n");

                    try {
                        if (parts.length > 1) {
                            data =
                                JSON.parse(parts[parts.length - 2])?.data || [];
                        }
                    } catch (err) {
                        console.error("Error parsing JSON:", err);
                        data = [];
                    }

                    buffer = parts[parts.length - 1] || "";

                    if (Array.isArray(data)) {
                        updateTables(data);
                        updateTableDetections(data);
                        if (document.URL.includes("old_layout")) {
                            // hacky and not good practice. will refactor in the future.
                            updateBBoxes(data);
                        }
                    }

                    processStream();
                });
            };

            processStream();
        })
        .catch((error) => {
            console.error("Error fetching detections:", error);
            setTimeout(() => fetchDetections(), 5000);
        });
};

const updateTable = (tableName) => {
    if (tableName == null) {
        return;
    }
    const tables = JSON.parse(localStorage.getItem("tables"));
    const id = tables.find((table) => {
        return table.label === tableName;
    }).id;

    const tableEl = document.getElementById(id);
    if (tableEl !== null) {
        tableEl.classList.add("highlighted");
    }
};

const resetTables = () => {
    const tables = JSON.parse(localStorage.getItem("tables"));
    tables.forEach((table) => {
        const tableEl = document.getElementById(table.id);
        if (tableEl !== null && tableEl.classList.contains("highlighted")) {
            tableEl.classList.remove("highlighted");
        }
    });
};

const updateTables = (data) => {
    const uniqueLabels = new Set();
    resetTables();

    // Process detections in order of detection (no sorting)
    data.forEach((detection) => {
        const unknown = detection.label === "UNKNOWN";
        let table = null;

        if (!unknown && !uniqueLabels.has(detection.label)) {
            table = getTable(detection.label, namelistJSON); // e.g. "T4"
            if (table !== null) {
                uniqueLabels.add(detection.label);

                // light up the table
                updateTable(table);
            }
        }

        if (!detection.bbox) return;
    });
};

// table detection list functionality

const detectionList = document.getElementById("table-detection-list");

const updateTableDetections = (data) => {
    let detections = [];

    data.forEach((detection) => {
        const name = detection.label.toUpperCase();
        if (name == "UNKNOWN") {
            return;
        }
        const table = getTable(name, namelistJSON);

        let detectionEl = document.createElement("div");
        detectionEl.classList.add("table-detection-element");
        detectionEl.innerHTML = `${name} (${table})`;

        detections.push(detectionEl);
    });

    detections = sortTableDetections(detections);
    detectionList.replaceChildren(...detections);
};

const sortTableDetections = (detectionList) => {
    return detectionList.sort((a, b) => a.innerText.localeCompare(b.innerText));
};
