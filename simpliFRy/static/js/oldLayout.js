import { setBBoxPos, clearBBoxes, loadNamelistJSON, getTable } from "./utils.js";

const detectionList = document.getElementById("table-detection-list")
let namelistJSON = undefined

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("video-feed").setAttribute('data', '/vidFeed')
  let namelistPath = localStorage.getItem("namelistPath")

  loadNamelistJSON(namelistPath).then((data)=> {
    namelistJSON = data
    fetchDetections()
  });
});


// MAIN LOOP
const fetchDetections = () => {
  console.log("FETCHING...");
  let buffer = '';
  let data = [];

  fetch(`/frResults`).then(response => {
    if (!response.ok || !response.body) {
      console.error('Fetch failed, retrying...');
      setTimeout(() => fetchDetections(), 5000);
      return;
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    const processStream = () => {
      reader.read().then(({ done, value }) => {
        if (done) {
          console.log('Stream ended, reconnecting...');
          setTimeout(() => fetchDetections(), 2000);
          return;
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        const parts = buffer.split('\n');

        try {
          if (parts.length > 1) {
            data = JSON.parse(parts[parts.length - 2])?.data || [];
          }
        } catch (err) {
          console.error('Error parsing JSON:', err);
        }

        buffer = parts[parts.length - 1] || '';
        
        updateBBoxes(data);
        updateDetectionList(data)
        processStream();
      })
    }

    processStream();
  }).catch(error => {
    console.error('Error fetching detections:', error);
    setTimeout(() => fetchDetections(), 5000);
  });
};


const updateBBoxes = (data) => {
  const videoContainer = document.getElementById("video-container");
  clearBBoxes(videoContainer);

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

    setBBoxPos(bboxEl, detection.bbox, videoContainer.offsetWidth, videoContainer.offsetHeight);
    videoContainer.appendChild(bboxEl);
  });
};


const updateDetectionList = (data) => {
  let detections = []

  data.forEach(detection => {
    const name = detection.label.toUpperCase()
    if (name == "UNKNOWN") {
      return
    }
    const table = getTable(name, namelistJSON)

    let detectionEl = document.createElement('div')
    detectionEl.classList.add('table-detection-element')
    detectionEl.innerHTML = `${name} (${table})`

    detections.push(detectionEl)
  })

  detections = sortDetections(detections)
  detectionList.replaceChildren(...detections)
}

const sortDetections = (detectionList) => {
  return detectionList.sort((a, b) => a.innerText.localeCompare(b.innerText))
}
