// Load fusion.json data
export const loadNamelistJSON = async (path) => {
    const data = await fetch(path).then(res => {
        if (res.ok) {
            return res.json();
        } else {
            console.warn('Could not load namelist.json');
            return null;
        }
    })

    return data
};

// Get country flag path for a given name
export const getCountryFlag = (name, namelistJSON) => {
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

// Get description for a given name
export const getDescription = (name, namelistJSON) => {
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

// Get table for a given name
export const getTable = (name, namelistJSON) => {
  if (!namelistJSON || !namelistJSON.details) return null;
  
  const person = namelistJSON.details.find(detail => {
    // Match by name (case-insensitive, partial match)
    return detail.name.toLowerCase().includes(name.toLowerCase()) || 
           name.toLowerCase().includes(detail.name.toLowerCase());
  });
  
  if (person && person.table) {
    return person.table
  }
  
  return null;
};

// updates the position of a bounding box element 
export const setBBoxPos = (bboxEl, bbox, video_width, video_height) => {
  let ratiod_height = video_height, ratiod_width = video_width;
  if ((video_height / video_width) > (9 / 16)) {
    ratiod_height = video_width * 9 / 16;
  } else {
    ratiod_width = video_height * 16 / 9;
  }

  const left_offset = (video_width - ratiod_width) / 2;
  const top_offset = (video_height - ratiod_height) / 2;

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

export const clearBBoxes = (videoContainer) => {
  const prevBBoxes = videoContainer.querySelectorAll(".bbox");
  prevBBoxes.forEach((element) => {
    element.remove();
  });
};

export const delay = (time) => {
    return new Promise(resolve => setTimeout(resolve, time))
}
