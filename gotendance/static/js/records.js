const present = `<svg  value="True" xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-checkbox"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M9 11l3 3l8 -8" /><path d="M20 12v6a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2h9" />1</svg>`
const absent = `<svg  value="False" xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-square"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M3 3m0 2a2 2 0 0 1 2 -2h14a2 2 0 0 1 2 2v14a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2z" />0</svg>`

class Records {
    constructor() {
        this.parentEl = document.getElementById("records-list-data");
        this.presentVal = 0
        this.data = {}; //"Name": {attendance: True, tags: [], buttonEl: <button></button>, rowEl: <tr>}
        this.allTags = new Set();
        this.selectedTags = new Set();
        this.currentFilterStatus = 'all';
    }
    
    createRecordsEl() {
        Object.entries(this.data).map(([name, details]) => {
            const recordsEntryEl = document.createElement('tr')

            const nameStr = `'${name}'`
            const displayName = details.name || name  // Use the name field from JSON, fallback to map key
            const recordsEntryChildren = `
                <td>${displayName}</td>
                <td>
                    <button type="button" class="toggle-attendance" onclick="handleMark(${nameStr})">
                         ${details.attendance ? present : absent}
                    </button>
                </td>
                `

            recordsEntryEl.classList.add('record-entry', 'entry')
            recordsEntryEl.innerHTML = recordsEntryChildren
            this.parentEl.appendChild(recordsEntryEl)

            this.data[name].buttonEl = recordsEntryEl.children[1]
            this.data[name].rowEl = recordsEntryEl; // Store the row element
        })
        this.generateTagButtons();
    }

    generateTagButtons() {
        const tagButtonsContainer = document.getElementById('tag-filter-buttons');
        if (!tagButtonsContainer) return;
        
        tagButtonsContainer.innerHTML = ''; // Clear existing buttons
        
        // Add "Select All" button
        const selectAllBtn = document.createElement('button');
        selectAllBtn.textContent = 'Select All';
        selectAllBtn.onclick = () => this.selectAllTags();
        selectAllBtn.className = 'tag-filter-btn select-all-btn';
        if (this.selectedTags.size === this.allTags.size && this.allTags.size > 0) {
            selectAllBtn.classList.add('active');
        }
        tagButtonsContainer.appendChild(selectAllBtn);
        
        // Add buttons for each unique tag
        Array.from(this.allTags).sort().forEach(tag => {
            const btn = document.createElement('button');
            btn.textContent = tag;
            btn.onclick = () => this.filterByTag(tag);
            btn.className = 'tag-filter-btn';
            if (this.selectedTags.has(tag)) {
                btn.classList.add('active');
            }
            tagButtonsContainer.appendChild(btn);
        });
    }

    selectAllTags() {
        if (this.selectedTags.size === this.allTags.size) {
            // If all are selected, deselect all
            this.selectedTags.clear();
        } else {
            // Select all tags
            this.selectedTags = new Set(this.allTags);
        }
        this.generateTagButtons();
        this.applyAllFilters();
    }

    loadData(data) {
        console.log("Initiating records...")
        console.log("Raw data received:", data)
        Object.entries(data).map(([name, details]) => {
            this.data[name] = {attendance: details.attendance, tags: details.tags || []}
            // Collect all unique tags
            if (details.tags) {
                details.tags.forEach(tag => {
                    this.allTags.add(tag);
                });
            }
        })
        console.log("All tags found:", Array.from(this.allTags))
        console.log(this.data)
        this.createRecordsEl()
        this.presentVal = this.updateNumbers(this.data)
        updateAttendance(this.presentVal, Object.keys(this.data).length);
    }

    updateNumbers(dataInput){
        let tempAttendance = 0
        Object.keys(dataInput).forEach(key =>{
            if(dataInput[key]["attendance"] == true){
               tempAttendance++;
            }
        });
        return tempAttendance
    }

    updateData(data) {
        Object.entries(data).map(([name, details]) => {
            const nameStr = `'${name}'`
            this.data[name].buttonEl.innerHTML = `
            <button type="button" class="toggle-attendance" onclick="handleMark(${nameStr})">
                         ${details.attendance ? present : absent}
                    </button>
            `
        })
    }

    mark(name) {
        const attendance = !this.data[name].attendance
        this.data[name].attendance = attendance; // Update the attendance status in the data
        const nameStr = `'${name}'`
        this.data[name].buttonEl.innerHTML = attendance ? present : absent
    }
    
    fetchRecords(init = false) {
        fetch('/fetchAttendance').then(response => {
            if (!response.ok) throw new Error('Bad response')
            return response.json()
        }).then(data => {
            init && this.loadData(data)
            init || this.updateData(data)
            let tempAttendance = 0
            tempAttendance = this.updateNumbers(data)
            updateAttendance(tempAttendance,Object.keys(data).length)
            return data
        }).catch(error => {
            console.log("Fetch operation failed", error)
        })
    }

    searchFilter(query) {
        for (const entryEl of this.parentEl.children) {
            const name = entryEl.children[0].textContent.toLowerCase()
            entryEl.style.display = name.includes(query) ? "" : "none"
        }
    }

    filterRecords(status) {
        this.currentFilterStatus = status;
        this.applyAllFilters();
    }

    filterByTag(tag) {
        if (this.selectedTags.has(tag)) {
            this.selectedTags.delete(tag);
        } else {
            this.selectedTags.add(tag);
        }
        this.applyAllFilters();
        this.generateTagButtons();
    }

    applyAllFilters() {
        Object.entries(this.data).map(([name, details]) => {
            const entryEl = this.data[name].rowEl;
            if (entryEl) {
                let showByStatus = false;
                
                // Apply attendance status filter
                if (this.currentFilterStatus === 'all') {
                    showByStatus = true;
                } else if (this.currentFilterStatus === 'present' && details.attendance) {
                    showByStatus = true;
                } else if (this.currentFilterStatus === 'absent' && !details.attendance) {
                    showByStatus = true;
                }

                // Apply tag filter
                let showByTag = true;
                if (this.selectedTags.size > 0) {
                    showByTag = details.tags && details.tags.some(tag => this.selectedTags.has(tag));
                }

                // Show only if both filters pass
                entryEl.style.display = (showByStatus && showByTag) ? '' : 'none';
            }
        });
    }
}


// Attendance Functionalities
const records = new Records()
records.fetchRecords(true)

const updateRecords = () => {
    records.fetchRecords()
}

const handleMark = (name) => {
    records.mark(name)
    const params = new URLSearchParams({
        name: name
    })

    fetch(`/changeAttendance?${params}`, {
        method: "PATCH",
    }).then(response => {
        if (!response.ok) throw new Error('Bad response')
        return response.json()
    }).then(_data => {
        console.log("Marked:", name)
    }).catch(error => {
        console.log("Patch operation failed", error)
        alert(`Error marking attendance of ${name}`)
    })
}

setInterval(updateRecords, 1000)

// Search
const searchBarEl = document.getElementById('search')
const search = () => {
    query = searchBarEl.value.toLowerCase()
    console.log(query)
    if(query!=""){
        records.searchFilter(query)
    }
    else{
        console.log("show stuff")
        for (const entryEl of records.parentEl.children) {
            entryEl.style.display = ""
            console.log(entryEl)
            console.log(entryEl.style.display)
        }
    }
}

searchBarEl.addEventListener("input", search)

const searchContainer = document.querySelector('.search-container')

searchBarEl.addEventListener('focus', () => {
    console.log("HI")
    searchContainer.classList.add('focused')
})

searchBarEl.addEventListener('blur', () => {
    searchContainer.classList.remove('focused')
})

// Attendance script for total amount of people
function updateAttendance(attended, total) {
    let percentage = (attended / total) * 100;
    document.getElementById("attendance").innerText = `${attended}/${total}`;
    document.getElementById("percentage").innerText = `${Math.round(percentage)}%`;
    
    let red = Math.round((100 - percentage) * 2.55);
    let green = Math.round(percentage * 2.55);
    document.getElementById("percentage").style.color = `rgb(${red}, ${green}, 0)`;
}

// Reset attendance button
const resetAttendanceBtn = document.getElementById('reset-attendance-btn')
if (resetAttendanceBtn) {
    resetAttendanceBtn.addEventListener('click', () => {
        if (confirm('Are you sure you want to reset all attendance?')) {
            fetch('/resetAttendance', {
                method: 'POST'
            }).then(response => {
                if (!response.ok) throw new Error('Failed to reset')
                return response.json()
            }).then(_data => {
                alert('All attendance has been reset')
            }).catch(error => {
                console.log("Reset operation failed", error)
                alert('Error resetting attendance')
            })
        }
    })
}

// Load and display streams list
function loadStreamsList() {
    fetch('/getStreamsList')
        .then(response => {
            if (!response.ok) throw new Error('Failed to fetch streams')
            return response.json()
        })
        .then(streams => {
            const streamsListEl = document.getElementById('streams-list')
            if (!streamsListEl) return
            
            streamsListEl.innerHTML = '' // Clear existing
            
            if (!streams || streams.length === 0) {
                streamsListEl.innerHTML = '<p style="color: #888;">No active streams</p>'
                return
            }
            
            streams.forEach(stream => {
                const streamEntryEl = document.createElement('div')
                streamEntryEl.className = 'stream-entry entry'
                streamEntryEl.innerHTML = `
                    <p class="stream-url entry-text">${stream.url}</p>
                    <button type="button" class="remove-button" onclick="handleRemove('${stream.url}')">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="icon icon-tabler icons-tabler-outline icon-tabler-trash"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M4 7l16 0" /><path d="M10 11l0 6" /><path d="M14 11l0 6" /><path d="M5 7l1 12a2 2 0 0 0 2 2h8a2 2 0 0 0 2 -2l1 -12" /><path d="M9 7v-3a1 1 0 0 1 1 -1h4a1 1 0 0 1 1 1v3" /></svg>
                    </button>
                `
                streamsListEl.appendChild(streamEntryEl)
            })
        })
        .catch(error => {
            console.log("Error loading streams list", error)
        })
}

function handleRemove(frUrl) {
    const params = new URLSearchParams({
        frUrl: frUrl
    })

    fetch(`/stopCollate?${params}`, {
        method: "DELETE"
    }).then(response => {
        if (!response.ok) throw new Error('Bad response')
        return response.json()
    }).then(_data => {
        loadStreamsList() // Refresh the list
    }).catch(error => {
        console.log("Delete operation failed", error)
        alert(`Error removing results stream, ${frUrl}`)
    })
}

// Load streams list on page load and refresh every 2 seconds
loadStreamsList()
setInterval(loadStreamsList, 2000)
