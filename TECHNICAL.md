# TheObservers — technical document (draft)

*Placeholder. It gets filled in as the milestones land (outline from SPEC §10).*

1. Approach in one paragraph
2. Assets used / dropped
3. Training data (IDSSE / DFL open data; nothing labelled on BAR-ATH)
4. Handling missing and noisy tracking
5. Possession rules implemented
6. Results (IDSSE LOMO CV; public window as a labelled sanity check only)
7. Limitations and future work
8. Reproduction (commands, runtime, hardware, sha256 of the submitted file)

## Pipeline validation (M0)

Our ATD loader, slot-to-team mapping, frame alignment (`video_frame = frame_id - 1`) and CSV writer
reproduce all four of the organisers' published baselines (`baseline_raw`, `_smooth25`, `_smooth50`,
`_smooth125`) byte for byte (`scripts/reproduce_baseline.py`).
