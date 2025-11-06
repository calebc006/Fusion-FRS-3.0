const present = `<svg  value="True" xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-checkbox"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M9 11l3 3l8 -8" /><path d="M20 12v6a2 2 0 0 1 -2 2h-12a2 2 0 0 1 -2 -2v-12a2 2 0 0 1 2 -2h9" />1</svg>`
const absent = `<svg  value="False" xmlns="http://www.w3.org/2000/svg"  width="24"  height="24"  viewBox="0 0 24 24"  fill="none"  stroke="currentColor"  stroke-width="2"  stroke-linecap="round"  stroke-linejoin="round"  class="icon icon-tabler icons-tabler-outline icon-tabler-square"><path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M3 3m0 2a2 2 0 0 1 2 -2h14a2 2 0 0 1 2 2v14a2 2 0 0 1 -2 2h-14a2 2 0 0 1 -2 -2z" />0</svg>`

class Records {
    constructor() {
        this.parentEl = document.getElementById("records-list-data");
        this.presentVal = 0
        this.data = {}; //"Name": {attendance: True, buttonEl: <button></button>, rowEl: <tr>}
    }
    
    createRecordsEl() {
        Object.entries(this.data).map(([name, details]) => {
            const recordsEntryEl = document.createElement('tr')

            const nameStr = `'${name}'`
            const recordsEntryChildren = `
                <td>${name}</td>
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
    }

    loadData(data) {
        console.log("Initiating records...")
        Object.entries(data).map(([name, details]) => {
            this.data[name] = {attendance: details.attendance}
        })
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
            // details.attendance ? present : absent
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
            entryEl.style.display = name.includes(query) ? "" : "none" // Changed 'show' to '' for correct display
        }
    }

    filterRecords(status) {
        Object.entries(this.data).map(([name, details]) => {
            const entryEl = this.data[name].rowEl; // Use the stored row element
            if (entryEl) {
                if (status === 'all') {
                    entryEl.style.display = ''; // Show all
                } else if (status === 'present' && details.attendance) {
                    entryEl.style.display = ''; // Show if present
                } else if (status === 'absent' && !details.attendance) {
                    entryEl.style.display = ''; // Show if absent
                } else {
                    entryEl.style.display = 'none'; // Hide otherwise
                }
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
        // No need to call updateRecords here, as setInterval already handles it
        // If you want immediate visual update without waiting for setInterval,
        // you might need to adjust the logic.
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