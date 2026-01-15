// Load fusion.json data
export const loadNamelistJSON = async (path) => {
    const data = await fetch(path).then((res) => {
        if (res.ok) {
            return res.json();
        } else {
            console.warn("Could not load namelist.json");
            return null;
        }
    });

    return data;
};

// Get country flag path for a given name
export const getCountryFlag = (name, namelistJSON) => {
    if (!namelistJSON || !namelistJSON.details) return null;

    const person = namelistJSON.details.find((detail) => {
        // Match by name (case-insensitive, partial match)
        return (
            detail.name.toLowerCase().includes(name.toLowerCase()) ||
            name.toLowerCase().includes(detail.name.toLowerCase())
        );
    });

    if (person && person.country_flag) {
        // Construct the full path relative to data directory
        const flagFolderPath = namelistJSON.flag_folder_path;
        if (flagFolderPath) {
            return `/data/${flagFolderPath}/${person.country_flag}`;
        }
    }

    return null;
};

// Get description for a given name
export const getDescription = (name, namelistJSON) => {
    if (!namelistJSON || !namelistJSON.details) return null;

    const person = namelistJSON.details.find((detail) => {
        // Match by name (case-insensitive, partial match)
        return (
            detail.name.toLowerCase().includes(name.toLowerCase()) ||
            name.toLowerCase().includes(detail.name.toLowerCase())
        );
    });

    if (person && person.description) {
        return person.description;
    }

    return null;
};

// Get table for a given name
export const getTable = (name, namelistJSON) => {
    if (!namelistJSON || !namelistJSON.details) return null;

    const person = namelistJSON.details.find((detail) => {
        // Match by name (case-insensitive, partial match)
        return (
            detail.name.toLowerCase().includes(name.toLowerCase()) ||
            name.toLowerCase().includes(detail.name.toLowerCase())
        );
    });

    if (person && person.table) {
        return person.table;
    }

    return null;
};

// Get priority for a given name (lower number = higher priority)
export const getPriority = (name, namelistJSON) => {
    if (!namelistJSON || !namelistJSON.details) return Infinity;

    const person = namelistJSON.details.find((detail) => {
        return (
            detail.name.toLowerCase().includes(name.toLowerCase()) ||
            name.toLowerCase().includes(detail.name.toLowerCase())
        );
    });

    if (person && typeof person.priority === "number") {
        return person.priority;
    }

    return Infinity; // No priority = lowest priority
};

// Sort detection elements by priority, then alphabetically
// Each element should have a data-name attribute for sorting
export const sortDetectionsByPriority = (detectionList, namelistJSON) => {
    return detectionList.sort((a, b) => {
        const nameA = a.dataset.name || a.innerText;
        const nameB = b.dataset.name || b.innerText;
        const priorityA = getPriority(nameA, namelistJSON);
        const priorityB = getPriority(nameB, namelistJSON);

        // Sort by priority first (lower number = higher priority)
        if (priorityA !== priorityB) {
            return priorityA - priorityB;
        }

        // If same priority, sort alphabetically
        return nameA.localeCompare(nameB);
    });
};

// updates the position of a bounding box element
export const setBBoxPos = (bboxEl, bbox, video_width, video_height) => {
    let ratiod_height = video_height,
        ratiod_width = video_width;
    if (video_height / video_width > 9 / 16) {
        ratiod_height = (video_width * 9) / 16;
    } else {
        ratiod_width = (video_height * 16) / 9;
    }

    const left_offset = (video_width - ratiod_width) / 2;
    const top_offset = (video_height - ratiod_height) / 2;

    const org_left = bbox[0] * ratiod_width;
    const org_top = bbox[1] * ratiod_height;
    const org_width = (bbox[2] - bbox[0]) * ratiod_width;
    const org_height = (bbox[3] - bbox[1]) * ratiod_height;

    const width_truncate = Math.max(0, -org_left);
    const height_truncate = Math.max(0, -org_top);

    bboxEl.style.left = `${
        Math.max(left_offset, org_left + left_offset).toFixed(0) - 5
    }px`;
    bboxEl.style.top = `${
        Math.max(top_offset, org_top + top_offset).toFixed(0) - 5
    }px`;
    bboxEl.style.width = `${Math.min(
        org_width - width_truncate,
        ratiod_width - org_left
    ).toFixed(0)}px`;
    bboxEl.style.height = `${Math.min(
        org_height - height_truncate,
        ratiod_height - org_top
    ).toFixed(0)}px`;
};

export const clearBBoxes = (videoContainer) => {
    const prevBBoxes = videoContainer.querySelectorAll(".bbox");
    prevBBoxes.forEach((element) => {
        element.remove();
    });
};

/**
 * Efficiently updates bounding boxes by reusing DOM elements
 * @param {HTMLElement} videoContainer - Container element for bboxes
 * @param {Array} detections - Array of detection objects with bbox, label, score
 * @param {Object} options - Configuration options
 * @param {boolean} options.showLabels - Whether to show labels on boxes (default: false)
 * @param {boolean} options.showUnknown - Whether to show unknown detections (default: true)
 * @param {Function} options.onBBoxCreate - Callback when creating new bbox element
 * @returns {Array} Array of bbox coordinates for resize handling
 */
export const updateBBoxes = (videoContainer, detections, options = {}) => {
    const {
        showLabels = false,
        showUnknown = true,
        onBBoxCreate = null,
    } = options;
    
    const existingBoxes = videoContainer.querySelectorAll(".bbox");
    const bboxData = [];
    
    // Filter detections with bboxes
    const detectionsWithBbox = detections.filter(d => {
        if (!d.bbox) return false;
        if (!showUnknown && d.label === "Unknown") return false;
        return true;
    });
    
    detectionsWithBbox.forEach((detection, idx) => {
        const isUnknown = detection.label === "Unknown";
        let bboxEl;
        
        if (idx < existingBoxes.length) {
            // Reuse existing element
            bboxEl = existingBoxes[idx];
        } else {
            // Create new element only if needed
            bboxEl = document.createElement("div");
            videoContainer.appendChild(bboxEl);
        }
        
        // Update classes: bbox + (bbox-unknown OR bbox-identified)
        bboxEl.className = isUnknown ? "bbox bbox-unknown" : "bbox bbox-identified";
        
        // Update label content with matching classes
        if (showLabels) {
            const labelClass = isUnknown ? "bbox-label bbox-label-unknown" : "bbox-label bbox-label-identified";
            bboxEl.innerHTML = `<p class="${labelClass}">${detection.label} <span class="bbox-score">${detection.score?.toFixed(2) || ""}</span></p>`;
        } else {
            bboxEl.innerHTML = "";
        }
        
        if (onBBoxCreate) onBBoxCreate(bboxEl, detection);

        setBBoxPos(
            bboxEl,
            detection.bbox,
            videoContainer.offsetWidth,
            videoContainer.offsetHeight
        );
        
        bboxData.push(detection.bbox);
    });
    
    // Remove extra boxes if we have fewer detections than before
    for (let i = detectionsWithBbox.length; i < existingBoxes.length; i++) {
        existingBoxes[i].remove();
    }
    
    return bboxData;
};

export const delay = (time) => {
    return new Promise((resolve) => setTimeout(resolve, time));
};
