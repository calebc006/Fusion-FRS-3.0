let historyStack = [];
let historyIndex = -1; // points to current state in history

const pushHistory = () => {
  // Get current state snapshot
  const boxes = document.querySelectorAll("#seatings-container .box");
  const currentState = Array.from(boxes).map(box => ({
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
};

const restoreState = (state) => {
  const container = document.getElementById("seatings-container");
  if (!container) return;

  // Remove all existing boxes
  container.querySelectorAll(".box").forEach(el => el.remove());

  boxCount = state.length;

  // Recreate boxes from the state snapshot
  state.forEach(({ id, x, y, label, color, width, height }) => {
    const newBox = document.createElement("div");
    newBox.className = `box ${id}`;
    newBox.innerHTML = `<div class="box-label">${label}</div>`;
    newBox.style.left = `${x}px`;
    newBox.style.top = `${y}px`;
    newBox.style.backgroundColor = color;
    newBox.style.width = `${width}px`;
    newBox.style.height = `${height}px`;

    container.appendChild(newBox);
    makeDraggable(newBox);
  });

  saveTablesToStorage();
};

const setBBoxPos = (bboxEl, bbox, width, height) => {
  let ratiod_height = height, ratiod_width = width;
  if ((height / width) > (9 / 16)) {
    ratiod_height = width * 9 / 16;
  } else {
    ratiod_width = height * 16 / 9;
  }

  const left_offset = (width - ratiod_width) / 2;
  const top_offset = (height - ratiod_height) / 2;

  const org_left = bbox[0] * ratiod_width;
  const org_top = bbox[1] * ratiod_height;
  const org_width = (bbox[2] - bbox[0]) * ratiod_width;
  const org_height = (bbox[3] - bbox[1]) * ratiod_height;

  const width_truncate = Math.max(0, -org_left);
  const height_truncate = Math.max(0, -org_top);

  bboxEl.style.left = `${Math.max(left_offset, org_left + left_offset).toFixed(0) - 5}px`;
  bboxEl.style.top = `${Math.max(top_offset, org_top + top_offset).toFixed(0) - 5}px`;
  bboxEl.style.width = `${Math.min(org_width - width_truncate, ratiod_width - org_left).toFixed(0)}px`;
  bboxEl.style.height = `${Math.min(org_height - height_truncate, ratiod_height - org_top).toFixed(0)}px`;
};

let currData = [];
let streamCheck = false;
const detectionList = document.getElementById("detections-list");
let fusionData = null;
let latestDetection = null;
const countryFlagImg = document.getElementById("country-flag-img");

// Load fusion.json data
const loadFusionData = async () => {
  try {
    const response = await fetch('/data/fusion.json');
    if (response.ok) {
      fusionData = await response.json();
    } else {
      console.warn('Could not load fusion.json');
    }
  } catch (error) {
    console.error('Error loading fusion.json:', error);
  }
};

// Get country flag path for a given name
const getCountryFlag = (name) => {
  if (!fusionData || !fusionData.details) return null;
  
  const person = fusionData.details.find(detail => {
    // Match by name (case-insensitive, partial match)
    return detail.name.toLowerCase().includes(name.toLowerCase()) || 
           name.toLowerCase().includes(detail.name.toLowerCase());
  });
  
  if (person && person.country_flag) {
    // Construct the full path relative to data directory
    const imgFolderPath = fusionData.img_folder_path || '';
    return `/data/${imgFolderPath}/${person.country_flag}`;
  }
  
  return null;
};

// Update country flag display
const updateCountryFlag = (detectionName) => {
  if (!detectionName || detectionName === "Unknown") {
    countryFlagImg.style.display = 'none';
    return;
  }
  
  const flagPath = getCountryFlag(detectionName);
  if (flagPath) {
    countryFlagImg.src = flagPath;
    countryFlagImg.style.display = 'block';
    latestDetection = detectionName;
  } else {
    countryFlagImg.style.display = 'none';
  }
};

const updateBoxAnimations = (detectedLabels) => {
  const boxes = document.querySelectorAll("#seatings-container .box");

  boxes.forEach(box => {
    const boxName = box.innerText.trim().toLowerCase();

    if (!boxName) {
      box.classList.remove("animate-pulse");
      return;
    }

    const isMatch = detectedLabels.some(label =>
      label.toLowerCase().includes(boxName)
    );

    if (isMatch) {
      box.classList.add("animate-pulse");
    } else {
      box.classList.remove("animate-pulse");
    }
  });
};

const endDetections = () => {
  streamCheck = false;
  currData = [];
  clearBBoxes();
  // Clear the country flag
  countryFlagImg.style.display = 'none';
  latestDetection = null;
};

const fetchDetections = () => {
  streamCheck = true;
  console.log("FETCHING...");
  let buffer = '';
  let data = [];

  fetch(`/frResults`).then(response => {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    const processStream = () => {
      reader.read().then(({ done, value }) => {
        if (done) {
          clearBBoxes();
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const parts = buffer.split('\n');

        try {
          if (parts.length > 1) {
            data = JSON.parse(parts[parts.length - 2])?.data;
          }
        } catch (err) {
          console.log(buffer);
          console.error('Error parsing JSON:', err);
        }

        buffer = parts[parts.length - 1];

        updateDetections(data);
        if (streamCheck) processStream();
      });
    };
    processStream();
  });
};

const createDetectionEl = (name) => {
  const detectionEl = document.createElement("p");
  detectionEl.innerHTML = name;
  detectionEl.classList.add("detectionEntry");
  detectionList.appendChild(detectionEl);
};

const clearBBoxes = () => {
  const videoContainer = document.getElementById("video-container");
  detectionList.innerHTML = "";
  const prevBBoxes = videoContainer.querySelectorAll(".bbox");
  prevBBoxes.forEach((element) => {
    element.remove();
  });
  return videoContainer;
};

const updateDetections = (data) => {
  currData = [];
  const videoContainer = clearBBoxes();
  const uniqueLabels = new Set();
  let mostRecentDetection = null;

  // Process detections in order of detection (no sorting)
  data.forEach((detection) => {
    const unknown = detection.label === "Unknown";

    if (!unknown && !uniqueLabels.has(detection.label)) {
      createDetectionEl(detection.label);
      uniqueLabels.add(detection.label);
    }

    // Track the last non-unknown detection as the most recent
    if (!unknown) {
      mostRecentDetection = detection.label;
    }

    if (!detection.bbox) return;

    const bboxEl = document.createElement("div");
    bboxEl.classList.add("bbox");
    if (!unknown) {
      bboxEl.classList.add("bbox-identified");
    }

    bboxEl.innerHTML = `<p class="bbox-label${unknown ? "" : " bbox-label-identified"}">${detection.label} <span class="bbox-score">${detection.score.toFixed(2)}</span></p>`;

    currData.push(detection.bbox);
    setBBoxPos(bboxEl, detection.bbox, videoContainer.offsetWidth, videoContainer.offsetHeight);
    videoContainer.appendChild(bboxEl);
  });

  updateBoxAnimations(Array.from(uniqueLabels));
  
  // Update country flag for the latest detection
  if (mostRecentDetection) {
    updateCountryFlag(mostRecentDetection);
  } else {
    // No identified detections in current list, hide flag
    countryFlagImg.style.display = 'none';
    latestDetection = null;
  }
};

window.addEventListener("resize", () => {
  const videoContainer = document.getElementById("video-container");
  const bboxesEl = videoContainer.querySelectorAll(".bbox");
  bboxesEl.forEach((element, idx) => {
    setBBoxPos(element, currData[idx], videoContainer.offsetWidth, videoContainer.offsetHeight);
  });
});

// TABLE MANAGEMENT
let boxCount = 0;

const loadTablesFromStorage = () => {
  const savedTables = JSON.parse(localStorage.getItem("tables") || "[]");
  boxCount = savedTables.length;
  savedTables.forEach(({ id, x, y, label, color, width, height }) => {
    const newBox = document.createElement("div");
    newBox.className = `box ${id}`;
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
    makeDraggable(newBox);
  });
};

const saveTablesToStorage = () => {
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
};

const randomColor = () => {
  const colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#46f0f0"];
  return colors[Math.floor(Math.random() * colors.length)];
};

const makeDraggable = (box) => {
  box.style.position = "absolute";

  // Respect current lock state
  box.style.pointerEvents = document.getElementById("lock-tables").checked ? "none" : "auto";

  // Enable renaming when clicking on the table name
  box.addEventListener("click", (e) => {
    if (document.getElementById("lock-tables").checked) return; // Prevent renaming if locked

    const labelEl = box.querySelector(".box-label");
    const label = labelEl?.innerText.trim() || "";
    const inputField = document.createElement("input");
    inputField.value = label;
    inputField.classList.add("rename-input");

    // Replace the box label with input field for renaming
    labelEl.innerHTML = "";
    labelEl.appendChild(inputField);
    inputField.focus();

    inputField.addEventListener("blur", () => {  // On blur (click outside)
      const newName = inputField.value.trim();
      if (newName !== label && newName !== "") {
        labelEl.innerText = newName;
        saveTablesToStorage();  // Save the updated name to localStorage
        pushHistory(); 
      } else {
        labelEl.innerText = label;
      }
    });

    inputField.addEventListener("keydown", (e) => {  // Allow renaming via Enter key
      if (e.key === "Enter") {
        const newName = inputField.value.trim();
        if (newName !== label && newName !== "") {
          labelEl.innerText = newName;
          saveTablesToStorage();  // Save the updated name to localStorage
          pushHistory(); 
        } else {
          labelEl.innerText = label;
        }
      }
    });
  });

const createResizeSlider = (box) => {
  const sliderWrapper = document.createElement("div");
  sliderWrapper.className = "resize-slider-wrapper";
  sliderWrapper.style.display = document.getElementById("lock-tables").checked ? "none" : "flex";

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

createResizeSlider(box); // Call this after the drag/rename logic

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


  box.addEventListener("mousedown", (e) => {
    if (document.getElementById("lock-tables").checked) return; // Prevent drag if locked

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
  const locked = document.getElementById("lock-tables").checked;

  boxes.forEach((box) => {  
    box.style.pointerEvents = locked ? "none" : "auto";

    const deleteBtn = box.querySelector(".delete-btn");
    if (deleteBtn) {
      deleteBtn.style.display = locked ? "none" : "block";
    }

    const sliderWrapper = box.querySelector(".resize-slider-wrapper");
    if (sliderWrapper) {
      sliderWrapper.style.display = locked ? "none" : "flex";
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
  newBox.innerHTML = `<div class="box-label">T${boxCount}</div>`;
  newBox.style.backgroundColor = randomColor();
  const container = document.getElementById("seatings-container");
  if (container) {
    container.appendChild(newBox);
  }
  makeDraggable(newBox);
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
  document.querySelectorAll("#seatings-container .box").forEach(el => el.remove());
  boxCount = 0;
  saveTablesToStorage();
  pushHistory(); 
});

const videoModal = document.getElementById("video-modal");
const videoContainer = document.getElementById("video-container");
const toggleVideoCheckbox = document.getElementById("toggle-video");
const closeVideoModal = document.getElementById("close-video-modal");
const modalOverlay = document.querySelector(".modal-overlay");

const isVideoVisible = localStorage.getItem("videoVisible") === "true";
toggleVideoCheckbox.checked = isVideoVisible;
if (isVideoVisible) {
  videoModal.classList.remove("hidden");
} else {
  videoModal.classList.add("hidden");
}

const showVideoModal = () => {
  videoModal.classList.remove("hidden");
  localStorage.setItem("videoVisible", "true");
};

const hideVideoModal = () => {
  videoModal.classList.add("hidden");
  localStorage.setItem("videoVisible", "false");
};

toggleVideoCheckbox.addEventListener("change", () => {
  if (toggleVideoCheckbox.checked) {
    showVideoModal();
  } else {
    hideVideoModal();
  }
});

closeVideoModal.addEventListener("click", () => {
  hideVideoModal();
  toggleVideoCheckbox.checked = false;
});

modalOverlay.addEventListener("click", () => {
  hideVideoModal();
  toggleVideoCheckbox.checked = false;
});

// Button to open video modal
const openVideoModalButton = document.getElementById("open-video-modal-button");
if (openVideoModalButton) {
  openVideoModalButton.addEventListener("click", () => {
    showVideoModal();
    toggleVideoCheckbox.checked = true;
  });
}

function makeMenuDraggable(menuId, handleId) {
  const menu = document.getElementById(menuId);
  const handle = document.getElementById(handleId);

  let offsetX = 0, offsetY = 0, isDragging = false;

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

window.addEventListener("DOMContentLoaded", () => {
  makeMenuDraggable("table-menu", "table-menu-header");
  loadFusionData();
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

document.getElementById("seatings-container").addEventListener("contextmenu", (e) => {
  const targetBox = e.target.closest(".box");
  if (!targetBox) return;

  e.preventDefault();

  colorPicker.style.left = `${e.pageX}px`;
  colorPicker.style.top = `${e.pageY}px`;
  colorPicker.style.display = "block";

  const currentColor = rgbToHex(getComputedStyle(targetBox).backgroundColor);
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

document.addEventListener("click", (e) => {
  if (e.target !== colorPicker) {
    colorPicker.style.display = "none";
  }
});

document.addEventListener("keydown", (e) => {
  // Close video modal on Escape key
  if (e.key === "Escape" && !videoModal.classList.contains("hidden")) {
    hideVideoModal();
    toggleVideoCheckbox.checked = false;
    return;
  }

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



