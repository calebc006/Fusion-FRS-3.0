# Developer's Guide for Gotendance

> This file provides information about gotendance's API endpoints that other services can interact with, and how to use gotendance with other result streams.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Result Stream](#result-stream)
- [API Reference](#api-reference)
- [Stream Resilience & Recovery](#stream-resilience--recovery)

---

## Architecture Overview

Gotendance uses a concurrent architecture for real-time stream processing:

- **Store**: Thread-safe in-memory data store for attendance records with mutex locks
- **Stream Manager**: Manages multiple concurrent result streams, each in its own goroutine
- **Stream Health Monitor**: Tracks connection status and implements automatic recovery for each stream
- **HTTP Server**: Lightweight web server on port 1500 with RESTful API endpoints

Data persists to `output.json` and is automatically loaded on startup.

### Key Design Principles

- **Stream Independence**: Each stream operates independently with its own health tracking and retry logic
- **Resilience**: Failed streams automatically attempt reconnection without affecting other streams
- **Visibility**: Real-time health status available through `/getStreamsList` endpoint
- **Thread Safety**: All shared data protected with mutex locks for concurrent access

---

## Result Stream

**Gotendance** updates its attendance list by listening to a results stream from a separate service. Though intended to work with [**simpliFRy**](../simpliFRy/)'s `/frResults` endpoint, gotendance can be used with other services sending a results stream as well.

Below is the format of a results stream's JSON output, which is repeatedly given in a ***HTTP Streaming Response***:

```json
{
  "data": [
    {
      "label": "John Doe"
    },
    {
      "label": "Jane Smith"
    }
  ]
}
```

For more information on the results stream, refer to the `/frResults` endpoint [here](../simpliFRy/Developer%20Guide.md#4-access-fr-results). Take note that gotendance does not need the `bbox` and `score` fields.

### Generate Result Stream

Flask app example (with `/resultsStream` as an endpoint):

```python
import json
from flask import Flask, Response

app = Flask(__name__)

# data can be a variable that changes continuously
data = [ 
    {
        "label": "John Doe"
    },
    {
        "label": "Jane Smith"
    }
]

def broadcast():
    while True:
        yield json.dumps({'data': data}) + '\n'

@app.route("/resultsStream")
def resultsStream():
    return Response(
        broadcast(), mimetype="application/json"
    )

if __name__ == "__main__":
    app.run()
```

---

## API Reference

All endpoints return JSON responses. Successful operations return `{"status": "ok"}` unless otherwise specified.

### Data Management

#### Load Personnel List

**Endpoint:** `POST /initData`

**Description:** Initialize or update the personnel list from a JSON file.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with file field named `jsonFile`

**File Format:**
```json
{
  "img_folder_path": "/path/to/images",
  "details": [
    {
      "name": "John Doe",
      "images": ["john1.jpg", "john2.jpg"],
      "tags": ["staff", "department-a"]
    },
    {
      "name": "Jane Smith",
      "images": ["jane1.jpg"],
      "tags": ["student"]
    }
  ]
}
```

**Response:**
```json
{
  "status": "ok"
}
```

**Notes:**
- Previous attendance data is preserved when reloading
- Names must be unique
- The `tags` field is stored but not currently used by the UI

---

#### Fetch Attendance Records

**Endpoint:** `GET /fetchAttendance`

**Description:** Retrieve the current attendance list with all detection details and timestamps.

**Request:**
- Method: `GET`
- Parameters: None

**Response:**
```json
{
  "Jane Smith": {
    "attendance": false,
    "detected": true,
    "firstSeen": "2026-01-12T14:30:45Z",
    "lastSeen": "2026-01-12T14:32:10Z"
  },
  "John Doe": {
    "attendance": true,
    "detected": false,
    "firstSeen": "0001-01-01T00:00:00Z",
    "lastSeen": "0001-01-01T00:00:00Z"
  }
}
```

**Field Descriptions:**
- `attendance` (boolean): Current attendance status (shown in UI)
- `detected` (boolean): Whether detected by any result stream
- `f1. Load Personnel List

- **Endpoint**: `POST /initData`
- **Description**: Initialize or update the personnel list from a JSON file
- **Request**: Multipart form data with file field named `jsonFile`
- **Response**: `{"status": "ok"}`

---

### 2. Fetch Attendance Records

- **Endpoint**: `GET /fetchAttendance`
- **Description**: Retrieve the current attendance list with detection details and timestamps
- **Request**: No parameters required
- **Response**:
  ```json
  {
    "Jane Smith": {
      "attendance": false,
      "detected": true,
      "firstSeen": "2026-01-12T14:30:45Z",
      "lastSeen": "2026-01-12T14:32:10Z"
    },
    "John Doe": {
      "attendance": true,
      "detected": false,
      "firstSeen": "0001-01-01T00:00:00Z",
      "lastSeen": "0001-01-01T00:00:00Z"
    }
  }
  ```
- **Notes**: 
  - `attendance` is the value shown on the "Records" page
  - `detected` shows whether the individual has been sent from the results stream
  - Zero timestamps indicate never detected
  - Records are saved to `output.json` when this endpoint is called

---

### 3. Get Attendance Summary

- **Endpoint**: `GET /getCount`
- **Description**: Get total, detected, and attended counts
- **Request**: No parameters required
- **Response**:
  ```json
  {
    "total": 10,
    "detected": 5,
    "attended": 7
  }
  ```

---

### 4. Start Listening to Stream

- **Endpoint**: `POST /startCollate`
- **Description**: Add a new result stream URL for gotendance to monitor
- **Request**: Form data with `frUrl` (string) and `updateInterval` (float, in seconds)
- **Response**: `{"status": "ok"}`
- **Notes**: URL is tested before adding; duplicate URLs are ignored

---

### 5. Stop Listening to Stream

- **Endpoint**: `POST /stopCollate?frUrl={url}`
- **Description**: Remove a result stream and stop monitoring it
- **Request**: Query parameter `frUrl` (string, required)
- **Response**: `{"status": "ok"}`

---

### 6. List Active Streams

- **Endpoint**: `GET /getStreamsList`
- **Description**: Get list of all currently monitored result streams with health status
- **Request**: No parameters required
- **Response**:
  ```json
  [
    {
      "url": "http://192.168.1.100:5000/frResults",
      "isHealthy": true,
      "lastError": "",
      "failureCount": 0
    },
    {
      "url": "http://192.168.1.101:5000/frResults",
      "isHealthy": false,
      "lastError": "stream ended or error occurred: EOF",
      "failureCount": 2
    }
  ]
  ```
- **Response Fields:**
  - `url` (string): The stream URL being monitored
  - `isHealthy` (boolean): Current connection status (true = actively streaming data)
  - `lastError` (string): Description of the last error (empty if healthy)
  - `failureCount` (integer): Number of consecutive connection failures
- **Notes**: Each stream maintains independent health status and automatic recovery

---

### 7. Toggle Attendance Status

- **Endpoint**: `POST /changeAttendance?name={name}`
- **Description**: Manually toggle the attendance status for a specific individual
- **Request**: Query parameter `name` (string, required, must match exactly)
- **Response**: `{"status": "ok"}`

---

### 8. Reset All Attendance

- **Endpoint**: `POST /resetAttendance`
- **Description**: Reset all attendance records to absent and clear all detection data
- **Request**: No parameters required
- **Response**: `{"status": "ok"}`
- **Notes**: Does not affect active stream connections

---

## Stream Resilience & Recovery

Gotendance implements robust stream recovery to ensure individual stream failures don't impact other active streams:

### Connection Management

Each stream operates independently with:
- **Automatic reconnection**: Failed connections retry with exponential backoff (1s, 2s, 4s, 8s, 16s)
- **Retry limit**: Maximum 5 consecutive failures before permanent stream termination
- **Health tracking**: Real-time status available via `/getStreamsList` endpoint
- **Isolation**: One stream's failure does not affect other streams

### Monitoring Stream Health

Use the `/getStreamsList` endpoint to check health status:

```bash
curl http://localhost:1500/getStreamsList
```

Response includes:
- `isHealthy`: `true` if actively streaming, `false` if failed or retrying
- `lastError`: Description of connection issue (empty when healthy)
- `failureCount`: Number of consecutive failures since last recovery

### Recovery Scenarios

**Scenario 1: Temporary Network Interruption**
1. Stream connection drops (network latency/packet loss)
2. `isHealthy` set to `false` with error message
3. Gotendance automatically retries with backoff
4. Network restored
5. Stream reconnects and resumes data collection
6. `isHealthy` set back to `true`, `failureCount` reset

**Scenario 2: Source Service Restart**
1. Recognition service (SimpliFRy) goes offline
2. Stream times out and marks as unhealthy
3. Gotendance continues retrying
4. Recognition service restarts
5. Stream successfully reconnects
6. Attendance tracking resumes immediately

**Scenario 3: Persistent Failure**
1. Stream fails 5 consecutive times
2. Error persists despite retries
3. Stream is permanently stopped
4. Other streams continue operating
5. Manual stream restart via `/stopCollate` then `/startCollate` if needed

### Best Practices

1. **Monitor `/getStreamsList` periodically** from your client to detect unhealthy streams
2. **Log stream health changes** for debugging network issues
3. **Do not restart all streams** if one fails—Gotendance recovers automatically
4. **Use different recognition sources** for critical deployments (redundancy)
5. **Check `failureCount`** to distinguish temporary issues from persistent problems