window.addEventListener("resize", () => {
  const videoContainer = document.getElementById("video-container");
  const bboxesEl = videoContainer.querySelectorAll(".bbox");
  bboxesEl.forEach((element, idx) => {
    setBBoxPos(element, currData[idx], videoContainer.offsetWidth, videoContainer.offsetHeight);
  });
});

const videoModal = document.getElementById("video-modal");
const videoContainer = document.getElementById("video-container");
const closeVideoModal = document.getElementById("close-video-modal");
const modalOverlay = document.querySelector(".modal-overlay");

let isVideoVisible = localStorage.getItem("videoVisible") === "true";
if (isVideoVisible) {
  videoModal.classList.remove("hidden");
} else {
  videoModal.classList.add("hidden");
}

const showVideoModal = () => {
  videoModal.classList.remove("hidden");
  isVideoVisible = true
  localStorage.setItem("videoVisible", "true");
};

const hideVideoModal = () => {
  videoModal.classList.add("hidden");
  isVideoVisible = false
  localStorage.setItem("videoVisible", "false");
};

// Button to open video modal
const openVideoModalButton = document.getElementById("open-video-modal-button");
if (openVideoModalButton) {
  openVideoModalButton.addEventListener("click", () => {
    showVideoModal();
  });
}

closeVideoModal.addEventListener("click", (e) => {
    hideVideoModal();
})

document.addEventListener("keydown", (e) => {
  // Close video modal on Escape key
  if (e.key === "Escape" && !videoModal.classList.contains("hidden")) {
    hideVideoModal();
  }
});

// button to end stream
document.getElementById("end_stream_button").addEventListener("click", async (event) => {
    // Handles form submission to end stream

    event.preventDefault()
    endDetections()
    document.getElementById("video-feed").removeAttribute('data')

    fetch('/end', {
        method: 'POST'
    }).then(response => response.json()).then(_data => {
        document.getElementById("main-container").style.display = 'none'
        document.getElementById("init").style.display = 'flex'
    }).catch(error => {
        console.log(error)
    })
})
