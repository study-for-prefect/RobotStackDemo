# Two-stage Visual Pick

The public command remains `tools/workflows/two_stage_visual_pick.py`.

- `arguments.py`: CLI options
- `scene.py`: target lookup and camera/base vector conversion
- `planning.py`: plan checks and second-snapshot XY correction
- `commands.py`: command construction and capture retries
- `io.py`: subprocess and JSON helpers
- `app.py`: top-level workflow order
