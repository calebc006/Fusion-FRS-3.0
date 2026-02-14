// Reference Images Viewer

let imageData = [];
let initialized = false;

const nameSelect = document.getElementById("name-select");
const imageGallery = document.getElementById("image-gallery");
const imageCount = document.getElementById("image-count");
const noImages = document.getElementById("no-images");

const lightbox = document.getElementById("lightbox");
const lightboxImg = lightbox?.querySelector("img");
const lightboxCloseBtn = lightbox?.querySelector(".lightbox-close");

// Initialize on the standalone references page
window.addEventListener("DOMContentLoaded", async () => {
    initReferencesUI({ autoLoad: true });
});


export function initReferencesUI({ autoLoad = true } = {}) {
    if (initialized) {
        if (autoLoad) {
            loadReferenceImages({ preserveSelection: true });
        }
        return;
    }

    initialized = true;

    if (lightboxCloseBtn) {
        lightboxCloseBtn.addEventListener("click", () => {
            handleLightboxClose();
        });
    }

    lightbox?.addEventListener("click", () => {
        handleLightboxClose();
    });

    document.addEventListener("keydown", () => {
        handleLightboxClose();
    });

    nameSelect?.addEventListener("change", (e) => {
        const selected = e.target.value;
        if (selected) {
            showPersonImages(selected);
        } else {
            showAllImages();
        }
    });

    if (autoLoad) {
        loadReferenceImages({ preserveSelection: true });
    }
}

export function refreshReferenceImages({ preserveSelection = true } = {}) {
    if (!initialized) {
        initReferencesUI({ autoLoad: false });
    }
    loadReferenceImages({ preserveSelection });
}

// ------ Helper functions ------

const removeDeleteButton = () => {
    const deleteBtn = document.getElementById("delete-img-btn");
    if (deleteBtn?.parentElement) {
        deleteBtn.parentElement.removeChild(deleteBtn);
    }
};

const handleLightboxClose = () => {
    lightbox?.classList.remove("active");
    removeDeleteButton();
};

async function loadReferenceImages({ preserveSelection = true } = {}) {
    if (!nameSelect || !imageGallery || !imageCount || !noImages) return;

    const selectedValue = preserveSelection ? nameSelect.value : "";

    try {
        const response = await fetch("/api/reference_images");
        if (!response.ok) throw new Error("Failed to fetch reference images");
        imageData = await response.json();

        if (imageData.length === 0) {
            noImages.style.display = "block";
            imageCount.textContent = "";
            imageGallery.innerHTML = "";
            updateNameSelect("");
            return;
        }

        noImages.style.display = "none";
        updateNameSelect(selectedValue);
        refreshView();
    } catch (err) {
        console.error("Error loading reference images:", err);
        noImages.textContent = "Error loading reference images.";
        noImages.style.display = "block";
    }
}

function showAllImages() {
    imageGallery.innerHTML = "";
    let totalImages = 0;

    if (imageData.length === 0) {
        noImages.style.display = "block";
        imageCount.textContent = "";
        return;
    }

    noImages.style.display = "none";

    imageData.forEach((person) => {
        person.images.forEach((imgPath) => {
            createImageCard(person.name, imgPath);
            totalImages++;
        });
    });

    imageCount.textContent = `Showing ${totalImages} images from ${imageData.length} people`;
}

function showPersonImages(name) {
    const person = imageData.find((p) => p.name === name);
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

function updateNameSelect(selectedValue) {
    nameSelect.innerHTML = '<option value="">-- All --</option>';
    imageData.forEach((person) => {
        const option = document.createElement("option");
        option.value = person.name;
        option.textContent = `${person.name} (${person.images.length} images)`;
        nameSelect.appendChild(option);
    });

    if (selectedValue && imageData.some((p) => p.name === selectedValue)) {
        nameSelect.value = selectedValue;
    }
}

function removeImageFromData(imgPath) {
    imageData = imageData
        .map((person) => ({
            ...person,
            images: person.images.filter((p) => p !== imgPath),
        }))
        .filter((person) => person.images.length > 0);
}

function refreshView() {
    const selected = nameSelect.value;
    updateNameSelect(selected);
    if (selected && imageData.some((p) => p.name === selected)) {
        showPersonImages(selected);
    } else {
        showAllImages();
    }
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

    // Click to open lightbox, add delete-img-btn
    card.addEventListener("click", () => {
        lightboxImg.src = imgPath;
        lightbox.classList.add("active");

        const deleteBtn = document.createElement("button")
        deleteBtn.id = "delete-img-btn"
        deleteBtn.className = "delete-img-btn"
        deleteBtn.innerHTML = "Remove Image"
        lightbox.appendChild(deleteBtn)

        deleteBtn.addEventListener("click", async (e) => {
            e.stopPropagation();
            const relativePath = imgPath.replace("/data/captures", "");
            const res = await fetch("/api/remove_image", {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json', 
                },
                body: JSON.stringify({"image_path": relativePath}),
            });

            if (res.ok) {
                removeImageFromData(imgPath);
                handleLightboxClose();
                refreshView();
            }
        })
    });

    imageGallery.appendChild(card);
}


