let currData = [];
let streamCheck = false;
let namelistJSON = null;
let latestDetection = null;
const detectionList = document.getElementById("detections-list");
const countryFlagImg = document.getElementById("country-flag-img");

// Load fusion.json data
const loadNamelistJSON = async () => {
  try {
    const response = await fetch('/data/namelist.json');
    if (response.ok) {
      namelistJSON = await response.json();
    } else {
      console.warn('Could not load namelist.json');
    }
  } catch (error) {
    console.error('Error loading namelist.json:', error);
  }
};

window.addEventListener("DOMContentLoaded", () => {
  loadNamelistJSON();
});

// Get country flag path for a given name
const getCountryFlag = (name) => {
  if (!namelistJSON || !namelistJSON.details) return null;
  
  const person = namelistJSON.details.find(detail => {
    // Match by name (case-insensitive, partial match)
    return detail.name.toLowerCase().includes(name.toLowerCase()) || 
           name.toLowerCase().includes(detail.name.toLowerCase());
  });
  
  if (person && person.country_flag) {
    // Construct the full path relative to data directory
    const flagFolderPath = namelistJSON.flag_folder_path || '';
    return `/data/${flagFolderPath}/${person.country_flag}`;
  }
  
  return null;
};

const getDescription = (name) => {
  if (!namelistJSON || !namelistJSON.details) return null;
  
  const person = namelistJSON.details.find(detail => {
    // Match by name (case-insensitive, partial match)
    return detail.name.toLowerCase().includes(name.toLowerCase()) || 
           name.toLowerCase().includes(detail.name.toLowerCase());
  });
  
  if (person && person.description) {
    return person.description
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

// Update detection list
const createDetectionEl = (name, description) => {
  const detectionEl = document.createElement("div");
  detectionEl.innerHTML = `<p class="detectionName">${name}</p> ${description===null ? '' : `<p class="detectionDesc">${description}</p>`}`;
  detectionEl.classList.add("detectionEntry");
  detectionList.appendChild(detectionEl);
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

const clearBBoxes = () => {
  const videoContainer = document.getElementById("video-container");
  detectionList.innerHTML = "";
  const prevBBoxes = videoContainer.querySelectorAll(".bbox");
  prevBBoxes.forEach((element) => {
    element.remove();
  });
  return videoContainer;
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


const updateDetections = (data) => {
  currData = [];
  const videoContainer = clearBBoxes();
  const uniqueLabels = new Set();
  let mostRecentDetection = null;

  // Process detections in order of detection (no sorting)
  data.forEach((detection) => {
    const unknown = detection.label === "Unknown";

    if (!unknown && !uniqueLabels.has(detection.label)) {
      description = getDescription(detection.label)
      createDetectionEl(detection.label, description);
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

  // Update country flag for the latest detection
  if (mostRecentDetection) {
    updateCountryFlag(mostRecentDetection);
  } else {
    // No identified detections in current list, hide flag
    countryFlagImg.style.display = 'none';
    latestDetection = null;
  }
};

