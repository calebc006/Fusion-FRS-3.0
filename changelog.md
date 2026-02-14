# FRS 3 Changelog

## v3.0 (Released 110226)
- These are the changes from v2.2... 
- Used for 29th SMEAC and 2026 SSPP @ JPJC

## v3.1-beta (CAA 150226)
### InteractiveFR
- [x] Distinguished input resolution from inference resolution. Improved default quality and resolution of stream. Can be configured on frontend.
- [x] Switched to non-square model input (640x480 default) 
- [x] Removed hold_time from the backend. Switched to using a queue with max-length for old_detections.
- [x] Catch and warn user when capturing with an existing name. Confirmation required via separate API path.
- [x] UI changes: Enlarged video feed, improved capture/remove image toasts, changed bbox labels (only display for target and identified faces).

- [ ] Improve backend perf-logging, expose to frontend
- [ ] Generally improve UI responsiveness
- [ ] Reliability and performance testing
- [ ] Update docs

### SimpliFRy
- [ ] Reuse similar backend to InteractiveFR
- [ ] Port over QOL UI changes (perf-logging, settings), refactor UI
- [ ] Reliability and performance testing
- [ ] Update docs

### Gotendance
- [ ] Auto import namelist file, use full path
- [ ] Manual marking overrules automatic detection
- [ ] Update docs