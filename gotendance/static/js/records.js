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
        this.searchQuery = ''; // Store current search query
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
        this.createRecordsEl()
        this.applyAllFilters(); // This will update the attendance display with correct counts
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
            // Update the data object with new values
            if (this.data[name]) {
                this.data[name].attendance = details.attendance;
                this.data[name].tags = details.tags || [];
            }
            
            const nameStr = `'${name}'`
            this.data[name].buttonEl.innerHTML = `
            <button type="button" class="toggle-attendance" onclick="handleMark(${nameStr})">
                         ${details.attendance ? present : absent}
                    </button>
            `
        })
        // Apply filters after updating data
        this.applyAllFilters();
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
            if (init) {
                this.loadData(data)
            } else {
                this.updateData(data)
            }
            // applyAllFilters (called by loadData/updateData) handles updating the attendance display
            return data
        }).catch(error => {
            console.log("Fetch operation failed", error)
        })
    }

    searchFilter(query) {
        this.searchQuery = query.toLowerCase(); // Store the search query
        this.applyAllFilters(); // Apply all filters together
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
        let filteredTotal = 0;
        let filteredPresent = 0;
        
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

                // Apply search filter
                const displayName = details.name || name;
                let showBySearch = true;
                if (this.searchQuery !== '') {
                    showBySearch = displayName.toLowerCase().includes(this.searchQuery) || name.toLowerCase().includes(this.searchQuery);
                }

                // Show only if all filters pass
                const isVisible = showByStatus && showByTag && showBySearch;
                entryEl.style.display = isVisible ? '' : 'none';
                
                // Count based on tag filter only (ignore status filter for count)
                if (showByTag) {
                    filteredTotal++;
                    if (details.attendance) {
                        filteredPresent++;
                    }
                }
            }
        });
        
        // Update attendance display with filtered counts
        updateAttendance(filteredPresent, filteredTotal);
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
    records.searchFilter(query) // Always call searchFilter to ensure filters are applied
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


