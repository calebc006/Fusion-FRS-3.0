# FRS 3 Changelog

## div-wps-may26 branch: 
- No custom background, use old_layout
- "ORANGE", "YELLOW", "RED" tables change color of displayed name in oldLayout.js detection list 

## v3.2 (In testing, as of 270426)
- UI updates, improved documentation
- Updated docker compose configuration to include both Gotendance and SimpliFRy.
- Introduced multiple branches for different events.
- Changed Gotendance export format to csv. Order of names kept identical to input json file. 

## v3.1 (Released 230326)

### SimpliFRy (Backend reused from InteractiveFR)
- Refactored backend to use dependency injection. Separated `VideoPlayer` and `FREngine` classes
- Converted buffer to `np.ndarray` before storing in `VideoPlayer`
- Factored out Voyager index to separate `EmbeddingIndex` class
- Improved backend perf-logging.
- Moved hold_time implementation from backend to frontend. Switched to using a persistor queue with max-length for old_detections.
- Separated input resolution from inference resolution. Improved default quality and resolution of stream; configurable from env.
- Switched to non-square model input (640x480 default) 
- Reworked init page logic, improved UI

### Gotendance
- Confirmed that we aren't dropping any detections, even with high update interval
- Export as csv instead of json

## v3.0 (Released 110226)
- Massively improved back-end performance and reliability via code optimizations. Eliminated previous issues of lag, crashing and instability.
- Added optional input configuration for users: description, table number (for new seating feature), sorting index (for priority of display), filter tag(s).
- Deployed for 29th SMEAC @ SAFTI MI and 2026 SSPP @ JPJC.

## Future
### TODO:
- [ ] Large scale setup testing for future events
- [ ] Need for tuning of parameters

### Potential Changes
- [ ] Convert all thresholds to cosine *similarity* (higher = better match)
- [ ] Two separate FFMPEG processes for RAW and MJPEG streams for lower latency
- [ ] Implement TLS (https) for safer multi-location implementation
- [ ] Port over perf log from InteractiveFR

