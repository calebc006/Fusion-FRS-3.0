const addStreamEntry = (frUrl) => {
    const streamsListEl = document.getElementById('streams-list')
    
    const newStreamEntryChildren = `
            <p class="stream-url entry-text">${frUrl}</p>
            <button type="button" class="remove-button" onclick="handleRemove('${frUrl}')">
                <svg  xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-trash"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 7l16 0" /><path d="M10 11l0 6" /><path d="M14 11l0 6" /><path d="M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2 -2l1 -12" /><path d="M9 7v-3a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v3" /></svg>
            </button>
        `
    const newStreamEntry = document.createElement('div')
    newStreamEntry.classList.add("stream-entry", "entry")
    newStreamEntry.innerHTML = newStreamEntryChildren

    streamsListEl.appendChild(newStreamEntry)
}

const removeStreamEntry = (frUrl) => {
    const streamsListEl = document.getElementById('streams-list')
    for (const childEl of streamsListEl.children) {
        if (childEl.children[0].textContent === frUrl) {
            streamsListEl.removeChild(childEl)
            return;
        }
    }
}

const handleRemove = (frUrl) => {
    const params = new URLSearchParams({
        frUrl: frUrl
    })

    fetch(`/stopCollate?${params}`, {
        method: "DELETE"
    }).then(response => {
        if (!response.ok) throw new Error('Bad response')
        return response.json()
    }).then(_data => {
        removeStreamEntry(frUrl)
    }).catch(error => {
        console.log("Delete operation failed", error)
        alert(`Error removing results stream, ${frUrl}`)
    })
}

const streamForm = document.getElementById('streamForm')
streamForm.addEventListener('submit', (event) => {
    event.preventDefault()

    const formData = new FormData(streamForm)
    frUrl = formData.get('frUrl')
    updateInterval = formData.get('updateInterval')
    
    fetch(streamForm.action, {
        method: streamForm.method,
        body: formData
    }).then(response => {
        if (!response.ok) throw new Error('Failed to submit form')

        return response.json()
    }).then(_data => {
        addStreamEntry(frUrl)
        streamForm.reset()
    }).catch(error => {
        console.log("Fetch operation failed", error)
        alert(`Error adding results stream, ${formData.get('frUrl')}`)
    })
})


//File upload
const fileInputEl = document.getElementById('fileInput')
const fileNameLabelEl = document.getElementById('fileInputLabel')

fileInputEl.addEventListener('change', () => {
    console.log(fileInputEl.files)
    if (fileInputEl.files.length === 0) {
        fileNameLabelEl.innerHTML = "Select File"
        return
    }

    const fileName = fileInputEl.files[0].name;
    fileNameLabelEl.innerHTML = fileName
})

const fileForm = document.getElementById('fileForm')
fileForm.addEventListener('submit', (event) => {
    event.preventDefault()

    const formData = new FormData(fileForm)
    
    fetch(fileForm.action, {
        method: fileForm.method,
        body: formData
    }).then(response => {
        if (!response.ok) throw new Error('Failed to submit form')
        return response.json()
    }).then(_data => {
        fileNameLabelEl.innerHTML = "Select File"
        fileForm.reset()
    }).catch(error => {
        console.log("Fetch operation failed", error)
        alert(`Error loading JSON file`)
    })
})
