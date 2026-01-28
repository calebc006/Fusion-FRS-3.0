// Reference Images Viewer

let allData = [];

const nameSelect = document.getElementById("name-select");
const imageGallery = document.getElementById("image-gallery");
const imageCount = document.getElementById("image-count");
const noImages = document.getElementById("no-images");

// Lightbox elements (create dynamically)
const lightbox = document.createElement("div");
lightbox.className = "lightbox";
lightbox.innerHTML = `
  <span class="lightbox-close">&times;</span>
  <img src="" alt="Full size" />
`;
document.body.appendChild(lightbox);

const lightboxImg = lightbox.querySelector("img");
const lightboxClose = lightbox.querySelector(".lightbox-close");

lightboxClose.addEventListener("click", () => {
  lightbox.classList.remove("active");
});

lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) {
    lightbox.classList.remove("active");
  }
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    lightbox.classList.remove("active");
  }
});

async function loadReferenceImages() {
  try {
    const response = await fetch("/api/reference_images");
    if (!response.ok) throw new Error("Failed to fetch");
    allData = await response.json();

    if (allData.length === 0) {
      noImages.style.display = "block";
      return;
    }

    // Populate dropdown
    allData.forEach((person) => {
      const option = document.createElement("option");
      option.value = person.name;
      option.textContent = `${person.name} (${person.images.length} images)`;
      nameSelect.appendChild(option);
    });

    // Show all images by default (optional: or show nothing until selected)
    showAllImages();
  } catch (err) {
    console.error("Error loading reference images:", err);
    noImages.textContent = "Error loading reference images.";
    noImages.style.display = "block";
  }
}

function showAllImages() {
  imageGallery.innerHTML = "";
  let totalImages = 0;

  allData.forEach((person) => {
    person.images.forEach((imgPath) => {
      createImageCard(person.name, imgPath);
      totalImages++;
    });
  });

  imageCount.textContent = `Showing ${totalImages} images from ${allData.length} people`;
}

function showPersonImages(name) {
  const person = allData.find((p) => p.name === name);
  if (!person) {
    showAllImages();
    return;
  }

  imageGallery.innerHTML = "";
  person.images.forEach((imgPath) => {
    createImageCard(person.name, imgPath);
  });

  imageCount.textContent = `Showing ${person.images.length} images for "${name}"`;
}

function createImageCard(name, imgPath) {
  const card = document.createElement("div");
  card.className = "image-card";

  const img = document.createElement("img");
  img.src = imgPath;
  img.alt = name;
  img.loading = "lazy";

  const label = document.createElement("div");
  label.className = "image-name";
  // Extract filename from path
  const filename = imgPath.split("/").pop();
  label.textContent = filename;
  label.title = filename;

  card.appendChild(img);
  card.appendChild(label);

  // Click to open lightbox
  card.addEventListener("click", () => {
    lightboxImg.src = imgPath;
    lightbox.classList.add("active");
  });

  imageGallery.appendChild(card);
}

nameSelect.addEventListener("change", (e) => {
  const selected = e.target.value;
  if (selected) {
    showPersonImages(selected);
  } else {
    showAllImages();
  }
});

// Initialize
loadReferenceImages();
