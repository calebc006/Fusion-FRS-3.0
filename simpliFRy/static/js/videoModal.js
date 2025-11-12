window.addEventListener("resize", () => {
  const videoContainer = document.getElementById("video-container");
  const bboxesEl = videoContainer.querySelectorAll(".bbox");
  bboxesEl.forEach((element, idx) => {
    setBBoxPos(element, currData[idx], videoContainer.offsetWidth, videoContainer.offsetHeight);
  });
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

