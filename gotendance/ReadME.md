# Gotendance

A lightweight attendance tracking application built with Go that processes real-time results from facial recognition systems.

![Project Logo](static/favicon.ico)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building from Source](#building-from-source)
  - [Distribution](#distribution)
- [How to Use](#how-to-use)
  - [Starting the Application](#starting-the-application)
  - [Loading Personnel Data](#loading-personnel-data)
  - [Connecting Result Streams](#connecting-result-streams)
  - [Viewing and Managing Attendance](#viewing-and-managing-attendance)
- [API Endpoints](#api-endpoints)
- [For Developers](#for-developers)

---

## Overview

**Gotendance** is an attendance tracking app designed to work with [**simpliFRy**](../simpliFRy/) for automated facial recognition attendance taking. Built primarily with **Go** and a vanilla **HTML/CSS/JavaScript** web interface, it is lightweight, portable, and easy to deploy without requiring complex runtime dependencies.

The application listens to real-time result streams (such as facial recognition data) and automatically updates attendance records, while also allowing manual attendance management through a clean web interface.

**Technology Stack:**
- Backend: Go (Golang)
- Frontend: HTML, CSS, vanilla JavaScript
- No external runtime dependencies (Node.js, etc.) required

---

## Key Features

### 🚀 Easy Installation and Deployment

Golang programs compile to a single binary executable, and the vanilla JavaScript UI runs in any modern browser without needing Node.js or other JavaScript runtimes. Once built, the application can be distributed by simply copying:
- The executable file (`gotendance.exe`)
- The `templates` folder
- The `static` folder

No installation process is needed on the target machine.

---

## Getting Started

### Prerequisites

- [Go](https://go.dev/doc/install) (version 1.16 or later recommended)

### Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/CJBuzz/Real-time-FRS-2.0.git
   ```

2. Navigate to the gotendance directory:
   ```bash
   cd gotendance
   ```

3. Build the executable:
   ```bash
   go build
   ```

This will generate a `gotendance.exe` file (on Windows) or `gotendance` binary (on Linux/Mac).

### Distribution

Once built, the application can be distributed as a self-contained package. Copy these items together:

```
folder/
├── static/          # CSS, JavaScript, and other static assets
├── templates/       # HTML templates for the web interface
└── gotendance.exe   # The compiled executable
```

Ensure all three components remain in the same directory. The application will run on any machine without requiring Go to be installed.

---

## How to Use

### Starting the Application

1. Run the executable:
   - **Windows**: Double-click `gotendance.exe`
   - **Linux/Mac**: Run `./gotendance` in terminal

2. The server will start on port 1500

3. Open your web browser and navigate to:
   ```
   http://localhost:1500
   ```
   or
   ```
   http://127.0.0.1:1500
   ```

   > **Recommended browser**: Chromium-based browser

### Loading Personnel Data

![Home Page](assets/main_page.png)

1. On the home page, locate the **Load Personnel List** section

2. Prepare a JSON file with the personnel list in the following format:
   ```json
   {
     "data": [
       {"label": "John Doe"},
       {"label": "Jane Smith"},
       {"label": "Alex Johnson"}
     ]
   }
   ```
   
   A sample file is provided at `assets/test.json`. For more details on data preparation, see the [simpliFRy data preparation guide](../simpliFRy/ReadME.md#data-preparation).

3. Click the file input, select your JSON file, and click the upload icon

4. The personnel list will be loaded into the system

### Connecting Result Streams

Gotendance listens to HTTP streaming responses from services like simpliFRy to automatically update attendance records.

1. **Ensure your result stream service is running** (e.g., simpliFRy)

2. In the **FR Results Stream URL** section, enter the stream URL:
   - For simpliFRy: `http://<IP>:<PORT>/frResults`
   - Example: `http://192.168.1.100:5000/frResults`
   
   The stream should provide data in this format:
   ```json
   {
     "data": [
       {"label": "John Doe"},
       {"label": "Jane Smith"}
     ]
   }
   ```

3. Set the **Update Interval** (in seconds):
   - This controls how often gotendance processes the incoming stream
   - Lower values = more frequent updates but higher CPU usage
   - Recommended: Keep it less than the [Holding Time](../simpliFRy/Developer%20Guide.md#holding-time) of simpliFRy to avoid missing detections
   - Default suggestion: 1-5 seconds

4. Click **Submit** to add the stream

5. **Managing streams**:
   - Successfully added streams appear under **Results Stream** section
   - Multiple streams can be added and will be processed concurrently
   - To remove a stream, click the remove button next to its URL
   - Gotendance will immediately stop listening to removed streams

### Viewing and Managing Attendance

![Records Page](assets/records.png)

1. Navigate to the records page:
   ```
   http://localhost:1500/records
   ```

2. **Understanding the display**:
   - Names with a ✓ checkmark have been detected by the recognition system
   - The checkbox icon shows the current attendance status

3. **Manual attendance management**:
   - Click any checkbox to toggle between Present/Absent
   - This allows manual overrides when needed (e.g., for technical issues or special cases)

4. **Resetting attendance**:
   - Use the reset function on the home page to clear all attendance records
   - Useful for starting a new session or event

5. **Exporting data**:
   - Attendance records are automatically saved to `output.json`
   - Records persist across application restarts

---

## API Endpoints

For developers integrating gotendance with other services, the following REST API endpoints are available:

### GET `/fetchAttendance`
Fetch current attendance list with detection details
- **Response**: JSON object with attendance status, detection status, and timestamps for each person

### POST `/changeAttendance?name={name}`
Toggle attendance status for a specific person
- **Parameters**: `name` (query parameter, must match exact name in system)

### GET `/getCount`
Get summary statistics
- **Response**: Total count, detected count, and attended count

### POST `/initData`
Load personnel list from JSON file
- **Body**: Multipart form with JSON file

### POST `/startCollate`
Add a new result stream
- **Parameters**: `frUrl` (string), `updateInterval` (float)

### POST `/stopCollate?frUrl={url}`
Remove a result stream
- **Parameters**: `frUrl` (query parameter)

### POST `/resetAttendance`
Reset all attendance records to absent

For detailed API documentation and information on creating custom result streams, refer to the [Developer Guide](Developer%20Guide.md).

---

## For Developers

If you are a developer looking to:
- Understand the internal architecture
- Create custom result streams
- Integrate gotendance with other services
- Build additional features

Please refer to the comprehensive [Developer Guide](Developer%20Guide.md) for detailed technical documentation.

To reset the attendance, repeat *Step 1*.
