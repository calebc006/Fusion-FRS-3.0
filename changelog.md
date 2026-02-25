# FRS 3 Changelog

## v3.0 (Released 110226)
- Massively improved back-end performance and reliability via code optimizations. Eliminated previous issues of lag, crashing and instability.
- Added optional input configuration for users: description, table number (for new seating feature), sorting index (for priority of display), filter tag(s).
- Introduced InteractiveFR, using a similar backend as SimpliFRy.
- Deployed for 29th SMEAC @ SAFTI MI and 2026 SSPP @ JPJC.

## v3.1 (CAA 170226)
### InteractiveFR
- [x] Refactored backend to use dependency injection. Separated `VideoPlayer` and `InteractiveFREngine` classes
- [x] Converted buffer to `np.ndarray` before storing in `VideoPlayer`
- [x] Factored out Voyager index to separate `EmbeddingIndex` class
- [x] Improved backend perf-logging; exposed perf-log to frontend
- [x] Removed hold_time from the backend. Switched to using a queue with max-length for old_detections.
- [x] Separated input resolution from inference resolution. Improved default quality and resolution of stream; configurable from env.
- [x] Switched to non-square model input (640x480 default) 
- [x] Catch and warn user when capturing with an existing name. Confirmation required via separate API path `/capture/confirm`.
- [x] UI changes: Enlarged video feed, improved capture/remove image toasts, changed bbox labels (only display for target and identified faces), added settings submit toast, capture on "ENTER"
- [ ] Reliability and performance testing
- [ ] Update docs and developer guide

### SimpliFRy
- [x] Reused similar backend to InteractiveFR
- [x] Ported over UI changes, reworked init page logic, implemented holding_time on frontend
- [ ] Reliability and performance testing
- [ ] Update docs

### Gotendance
- [x] Confirm that we aren't dropping any detections, even with high update interval
- [ ] Update docs

## Future
- [ ] Convert all thresholds to cosine *similarity* (higher = better match)
- [ ] Implement TLS (https) for safer multi-location implementation
- [ ] Two separate FFMPEG processes for RAW and MJPEG streams
- [ ] Lazy loading for reference images
- [ ] Port over perf log to SimpliFRy
- [ ] ibpng warning: iCCP: known incorrect sRGB profile (warning during png decode)
- [ ] Manual marking in Gotendance overwrites 
