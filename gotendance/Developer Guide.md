# Developer's Guide for Gotendance

> This file provides information about some of gotendance's API endpoints that other services can interact with, and also how to use gotendance with other result streams.

## Table of Contents

- [Result Stream](#result-stream)
- [API Endpoints](#api-endpoints)

---

## Result Stream

**Gotendance** updates its attendance list by listening to a results stream from a separate service. Though intended to work with [**simpliFRy**](../simpliFRy/)'s `/frResults` endpoint, gotendance can be used with other services sending a results stream as well. 

Below is the format of a results stream's JSON output, which is repeatedly given in a ***HTTP Streaming Response***.

```js
{
    "data": [
    {
        "label": "John Doe", // Individual's name
    },
    {
        "label": "Jane Smith"
    } 
    // Other similar objects below
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

## API Endpoints

#### 1. Fetch Current Attendance List

- **Endpoint**: `/fetchAttendance`
- **Method**: `GET`
- **Description**: Fetch current attendance list. `attendance` is the value shown on the "Records" page whereas `detection` shows whether the individual has actually been sent from the results stream of another service (e.g. facial recognition in simpliFRy) or not. 
- **Request**: No parameters required
- **Response**:
  - Status: `200 OK`
  - Body:
    ```js
    {
      "Jane Smith" : {
        "attendance": false, 
        "detected": false, // not sent from service sending results stream yet (e.g. simpliFRy)
        "firstSeen": "0001-01-01T00:00:00Z", // first time name appears in results stream
        "lastSeen": "0001-01-01T00:00:00Z" // last time name appears in results stream
      }, 
      "John Doe": {
        "attendance": true, // manually marked as present
        "detected": false,
        "firstSeen": "0001-01-01T00:00:00Z",
        "lastSeen": "0001-01-01T00:00:00Z"
      }
    }
    ```

#### 2. Change Attendance

- **Endpoint**: `/changeAttendance?name={name}`
- **Method**: `POST`
- **Description**: Changes `attendance` field of a specified individual (`true` to `false` and vice versa).
- **Request**: 
  - `name` (string, required): query parameter specifying name of individual whose boolean value in the `attendance` field will be changed. This must match the string of the person loaded into gotendance exactly.
- **Response**:
  - Status: `200 OK`
  - Body:
    ```json
    {
      "statue": "ok"
    }
    ```

#### 3. Get Detected Count

- **Endpoint**: `/getCount`
- **Method**: `GET`
- **Description**: Get total number of individuals in the attendance list, and number of detected and attended individuals. 
- **Request**: No parameters required
- **Response**:
  - Status: `200 OK`
  - Body:
    ```js
    {
      "total": 10, // total number of individuals in attendance list
      "detected": 5, // number of individuals sent from the results stream of another service
      "attended": 5, // number of individuals considered "present" in the /records page
    }
    ```
