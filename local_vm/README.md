# Local YOLOv8 processing (Part 1)

Install the local processing dependencies from the repository root:

```sh
python -m pip install -r local_vm/requirements.txt
```

Capture a frame every five seconds, count people with the Ultralytics YOLOv8
Nano model, and append each result to the shared CSV format:

```sh
python -m local_vm.process_local
```

The first run downloads `yolov8n.pt`. Stop the process with `Ctrl+C`; capture
errors are reported and the next interval is attempted. By default the result
file is `results/local.csv`. Options include `--cam`, `--interval`, `--count`,
`--model`, `--confidence`, and `--csv`; use `--help` for details. For example,
process ten frames and write them to a separate CSV:

```sh
python -m local_vm.process_local --count 10 --csv results/local-test.csv
```

Generate the people-per-frame time-series graph after collecting results:

```sh
python -m local_vm.plot_local
```

By default the graph is saved as `results/local_people_over_time.png`. Both
commands can also be run with custom `--csv` and, for the plot, `--out` paths.
The CSV uses the common fields in `common/results.py`; vehicles are left empty
because Part 1 counts people only.
