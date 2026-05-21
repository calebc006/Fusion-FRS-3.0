# Gotendance

A lightweight attendance tracking application built with Go that processes real-time results from facial recognition systems.

![Project Logo](./static/favicon.ico)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)

---

## Overview

**Gotendance** is an attendance tracking app designed to work with [**simpliFRy**](../simpliFRy/) for automated facial recognition attendance taking. Built primarily with **Go** and a vanilla **HTML/CSS/JavaScript** web interface, it is lightweight, portable, and easy to deploy without requiring complex runtime dependencies.

The application listens to real-time result streams (such as facial recognition data) and automatically updates attendance records, while also allowing manual attendance management through a clean web interface.

**Technology Stack:**
- Backend: Go (Golang)
- Frontend: HTML, CSS, vanilla JavaScript
- No external runtime dependencies (Node.js, etc.) required

---

## Installation

> For complete setup instructions including Docker, refer to the [main README](../ReadME.md#installation--setup)

### Quick Start

**Using Docker (Recommended):**
```bash
docker compose up gotendance
```

**For Local Development:**
```bash
cd gotendance
go build
./gotendance.exe  # Linux/macOS: ./gotendance
```

Access at **http://localhost:1500**

### Prerequisites
- Go 1.23.0+ (for local development)
- Docker & Docker Compose (for containerized setup)

---

## Usage

### Web Interface

1. **Load Personnel Data** (home page):
   - Upload a JSON file with personnel details
   - Same format as SimpliFRy's namelist

2. **Connect to Recognition Stream**:
   - Enter the SimpliFRy results URL
   - **In Docker**: `http://simplifry:1333/api/frResults`
   - **On Host**: `http://host.docker.internal:1333/api/frResults`
   - **External**: `http://192.168.x.x:1333/api/frResults`

3. **Attendance Management**:
   - View real-time recognized individuals
   - Manually adjust attendance records
   - View attendance summary by count

4. **Export Records**:
   - Download attendance data in JSON format
   - Historical data persisted to `output.json`

### Expected Input Format

SimpliFRy must stream JSON in this format:

```json
{
  "data": [
    {"label": "John Doe"},
    {"label": "Jane Smith"}
  ]
}
```

Gotendance automatically:
- Parses the streaming response
- Matches labels to loaded personnel
- Updates attendance records in real-time
- Persists state to `output.json`

---

## API Endpoints

All endpoints return `{"status": "ok"}` on success. Stream data is auto-loaded from `output.json` on startup.

### Data Management

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/initData` | POST | Load personnel list from JSON file |
| `/fetchAttendance` | GET | Retrieve current attendance records |
| `/getCount` | GET | Get attendance count summary |
| `/changeAttendance` | POST | Manually update attendance status |
| `/resetAttendance` | POST | Clear all attendance records |

### Stream Management

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/startCollate` | POST | Start listening to a result stream |
| `/stopCollate` | POST | Stop listening to a stream |
| `/getStreamsList` | GET | List active streams with health status |

### Static Files

`/static/` - Serves CSS, JavaScript, and image assets
`/records` - View attendance records page
`/` - Home/main interface 
   - Lower values = more frequent updates

4. Click **Submit** to add the stream

### 1. Create Personnel JSON

Use the same JSON format as SimpliFRy:

```json
{
  "details": [
    {
      "name": "John Doe",
      "tags": ["Army", "Combat"],
      "description": "Commander"
    },
    {
      "name": "3SG CALEB",
      "tags": ["Air Force"]
    }
  ]
}
```

### 2. Start Attendance Tracking

1. Upload the JSON file via home page
2. Connect to SimpliFRy's streaming endpoint
3. Gotendance automatically builds attendance as faces are recognized
4. View records at `/records` page

### 3. Manual Adjustments

- Click checkboxes on `/records` page to toggle attendance manually
- Use "Reset Attendance" button to clear all records
- Changes persist to `output.json` automatically

### 4. Exporting Data

- Scroll to the bottom of the page to export attendance data as a `.csv` file.
- The exported namelist will list names in the same order as the imported `.json` file.

---

## API Endpoints

### Detailed Endpoint Reference

**POST `/initData`**
- Load personnel list from JSON file
- Body: Multipart form data with `jsonFile` field

**GET `/fetchAttendance`**
- Returns current attendance records with detection status
- Response: `{"attendance": {...}, "personnel": [...]}`

**POST `/changeAttendance?name={name}`**
- Toggle attendance status for a person
- Query: `name` (must match exactly)

**GET `/getCount`**
- Attendance summary statistics
- Response: `{"total": N, "detected": N, "attended": N}`

**POST `/startCollate`**
- Start listening to a result stream
- Body: `frUrl` and `updateInterval` (float, in seconds)

**POST `/stopCollate?frUrl={url}`**
- Stop listening to a stream
- Query: `frUrl` (exact stream URL)

**GET `/getStreamsList`**
- List all active streams with health status
- Response: Array of stream objects with health info
- Response Example:
```json
[
  {
    "url": "http://simplifry:1333/api/frResults",
    "isHealthy": true,
    "lastError": "",
    "failureCount": 0
  },
  {
    "url": "http://192.168.1.100:1333/api/frResults",
    "isHealthy": false,
    "lastError": "stream ended or error occurred: EOF",
    "failureCount": 2
  }
]
```
- **isHealthy**: Current connection status (true = actively streaming)
- **lastError**: Description of last error (empty if healthy)
- **failureCount**: Number of consecutive connection failures

**POST `/resetAttendance`**
- Clear all attendance records
- Response: `{"status": "ok"}`

**GET `/records`**
- Renders attendance records page in browser

---

## Stream Resilience & Recovery

Gotendance includes automatic recovery for network failures:

**Connection Handling:**
- Each stream independently monitors its connection status
- Failed streams do **not** affect other active streams
- Automatic reconnection with exponential backoff (1s, 2s, 4s, 8s...)
- After 5 consecutive failures, a stream is permanently stopped

**Monitoring Stream Health:**
- Call `/getStreamsList` to check all streams' health status
- `isHealthy: true` = stream is actively receiving data
- `isHealthy: false` = stream is disconnected or retrying
- `failureCount` = consecutive failures since last successful connection

Example cmd command to check this continually:
```
for /l %i in () do @(curl http://localhost:1500/getStreamsList & timeout /t 1 >nul)
```

**Example: Handling Network Recovery**
1. SimpliFRy server goes offline
2. Stream marked as `isHealthy: false` with error description
3. Gotendance automatically retries with delays
4. SimpliFRy comes back online
5. Stream reconnects and resumes data collection
6. Other streams continue operating normally

---

## Deployment Notes

### Portable Binary Distribution

To distribute Gotendance as a standalone package:

```
gotendance-package/
├── static/              # CSS, JavaScript, assets
├── templates/           # HTML templates
├── gotendance.exe       # Compiled binary (Windows)
└── output.json          # Persisted attendance data
```

No additional dependencies required—just copy and run.

### Integration with SimpliFRy

When using SimpliFRy as the recognition source:
- SimpliFRy's `/api/frResults` endpoint streams recognition data
- Gotendance consumes and persists this stream
- Both apps share the same personnel JSON format


