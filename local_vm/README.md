# Local YOLOv8 processing (Part 1)

Create an isolated environment and install the local processing dependencies
from the repository root:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r local_vm/requirements.txt
```

Capture a frame every five seconds, count people with the Ultralytics YOLOv8
Nano model, and append each result to the shared CSV format:

```sh
.venv/bin/python -m local_vm.process_local
```

The first run downloads `yolov8n.pt`. Stop the process with `Ctrl+C`; capture
errors are reported and the next interval is attempted. By default the result
file is `results/local.csv`. Options include `--cam`, `--interval`, `--count`,
`--model`, `--confidence`, `--image-size`, and `--csv`; use `--help` for
details. The default inference size is 960 pixels, which detects distant
pedestrians in the selected webcam that the default 640-pixel YOLO input misses.
Use a smaller `--image-size` to trade detection sensitivity for faster inference.
For example, process ten frames and write them to a separate CSV:

```sh
.venv/bin/python -m local_vm.process_local --count 10 --csv results/local-test.csv
```

Generate the people-per-frame time-series graph after collecting results:

```sh
.venv/bin/python -m local_vm.plot_local
```

By default the graph is saved as `results/local_people_over_time.png`. Both
commands can also be run with custom `--csv` and, for the plot, `--out` paths.
The CSV uses the common fields in `common/results.py`; vehicles are left empty
because Part 1 counts people only.
