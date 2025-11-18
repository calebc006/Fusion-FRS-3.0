// video feed

const videoFeed = document.getElementById("video-feed")
videoFeed.setAttribute('data', '/vidFeed')

// bbox utils 

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
  const prevBBoxes = videoContainer.querySelectorAll(".bbox");
  prevBBoxes.forEach((element) => {
    element.remove();
  });
  return videoContainer;
};

const updateBBoxes = (data) => {
  currData = [];
  const videoContainer = clearBBoxes();

  // Process detections in order of detection (no sorting)
  data.forEach((detection) => {
    const unknown = detection.label === "Unknown";

    // if you want to hide unknown bboxes
    // if (unknown) {
    //   return;
    // }

    if (!detection.bbox) return;

    const bboxEl = document.createElement("div");
    bboxEl.classList.add("bbox");
    bboxEl.classList.add("bbox-identified");

    // old UI for blue and red boxes
    // bboxEl.innerHTML = `<p class="bbox-label${unknown ? "" : " bbox-label-identified"}">${detection.label} <span class="bbox-score">${detection.score.toFixed(2)}</span></p>`;

    // new UI with all blue boxes
    bboxEl.innerHTML = `<p class="bbox-label${" bbox-label-identified"}"><span class="bbox-score"></span></p>`;

    currData.push(detection.bbox);
    setBBoxPos(bboxEl, detection.bbox, videoContainer.offsetWidth, videoContainer.offsetHeight);
    videoContainer.appendChild(bboxEl);
  });
};

