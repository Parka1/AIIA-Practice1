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

## Part 2: Azure VM processing

The VM currently runs Ubuntu 24.04 on a Standard B2ts v2 in Spain Central with
about 1 GiB RAM. To avoid loading PyTorch and its larger memory footprint onto
the small VM, export the same YOLOv8 Nano weights to a fixed-size ONNX model
locally. The VM runs YOLOv8 through OpenCV DNN on CPU; it still performs
YOLOv8 inference, but does not need the Ultralytics/PyTorch runtime.

Export the model from the repository root:

```sh
.venv/bin/python -m local_vm.export_vm_model
```

Deploy the app to the existing Linux VM. The public web endpoint uses Apache
on port 80 as a reverse proxy to the private inference service. The deploy
script installs the Python service dependencies and a pinned OpenCV DNN wheel
compatible with the exported ONNX model,
sets up a single-worker systemd service, preserves the existing Apache welcome
page as a backup, and creates a random upload token at
`~/.config/aiia-practice1/vm-upload-token` with owner-only permissions.

```sh
VM_SSH_TARGET=azureuser@VM_PUBLIC_IP \
VM_SSH_KEY="$HOME/.ssh/azure_vm_key" \
  ./local_vm/deploy_vm.sh
```

The deployment expects TCP port 80 to be allowed through the Azure network
security group. It does not change Azure networking rules. The frame upload
endpoint requires the generated Bearer token; keep it private and use this
HTTP endpoint only for the course demonstration.

Start uploading frames from the local machine at five-second intervals:

```sh
export AIIA_VM_URL=http://VM_PUBLIC_IP
.venv/bin/python -m local_vm.send_to_vm --count 10 \
  --save-frames-dir /tmp/part2-frames
```

The client reads the token from the generated configuration file by default.
The dashboard is at `http://VM_PUBLIC_IP/`; it displays the people-count
histogram, annotated preview, recent results, process CPU, system and process
memory, JPEG payload size and rate, response latency and estimated compute
cost. The VM records the common result schema under
`/var/lib/aiia-part2/results.csv` and per-frame
performance measurements in `metrics.csv`. For an initial health check, open
`http://VM_PUBLIC_IP/api/health`.

The upload bandwidth rate covers the latest uninterrupted upload run; gaps
longer than 60 seconds start a new run so time when no test was running does not
artificially reduce its measured payload rate. It divides that run's JPEG
payload by the interval from the first capture through completion of its last
result. The response-time values compare the local capture timestamp with the
VM completion time, so accurate clock synchronization is required.

The cost estimate multiplies the public Linux retail rate for the VM's SKU and
region by the VM uptime since its last boot. It excludes disk, network egress,
taxes and subscription-specific discounts and is not an Azure invoice amount.
The result `frame_bytes` is the encoded JPEG payload size; total interface
traffic is also reported separately and includes system traffic.
Use `/api/dashboard` for the raw dashboard data.

To measure count-level detection effectiveness, annotate the
`ground_truth_count` column in `report/data/part2_vm_ground_truth.csv` by
counting every visible person in each corresponding saved frame. Keep the
existing `yolo_count` values unchanged, then run:

```sh
.venv/bin/python -m local_vm.evaluate_vm_counts
```

The evaluator reports overall and per-camera exact-count accuracy, mean
absolute count error, root mean square count error and signed count bias. This
is a frame-count evaluation, not box-level precision or recall; those require
manually labelled boxes and a specified matching rule.

## Part 4: VM multi-camera capacity

The capture map includes eight independently verified Part 4 SkylineWebcams
video streams (Santa Pola, both Benidorm beaches, Madrid Calle de Alcalá,
Benalmádena, La Orotava, Villafranca del Bierzo and Calafell), plus six
periodically refreshed low-resolution live-preview JPEGs for load-only
testing. For Task 4.1 cadence measurements, select the video streams only and
use `--persistent-streams` so each worker samples a recent decoded frame at
the configured interval. Snapshot JPEGs are not fresh enough to establish a
five-second webcam cadence. A finite `--count` is the number of successful
frames for each camera; optional local copies of the uploaded results and
original JPEG frames make the run reproducible:

```sh
AIIA_VM_URL=http://VM_PUBLIC_IP \
  .venv/bin/python -m local_vm.send_to_vm --persistent-streams \
  --cams santa_pola benidorm_poniente benidorm_levante madrid_alcala \
         benalmadena la_orotava villafranca_del_bierzo calafell \
  --interval 5 --count 10 \
  --results-csv report/data/part4_vm_scale_results.csv \
  --save-frames-dir report/part4_frames
```

Increase the camera set gradually and use the VM dashboard API before and
after each batch. It reports per-request CPU, process RSS and system RAM,
JPEG payload sizes and response times. The local CSV records each successful
frame in the shared results format. `--cams` starts one capture/upload worker
per source; pair it with `--persistent-streams` for HLS sources to keep each
video connection open and sample a recent decoded frame for every upload.
This avoids repeatedly reopening HLS streams, which can delay frame
acquisition and break the five-second cadence. Snapshot JPEG sources cannot
use this option because their refresh rate is lower than the required cadence.
The interval is measured from the capture timestamp; uploads that take longer
than the interval delay the next sample instead of causing a catch-up burst.
The VM's single inference lock serializes YOLO requests, so capture timestamps
and response times reveal when work starts queueing.
The sender stops with a clear error after three consecutive capture or upload
failures for any selected camera rather than leaving a finite experiment stuck
waiting indefinitely for an unavailable stream. The JPEG preview sources
refresh less often than the five-second request cadence and are not suitable
for detection-accuracy evaluation.
For a controlled VM-only capacity test, `--replay-dir` accepts one saved
`<cam_id>.jpg` per selected camera and reuses those bytes at each interval;
this removes webcam retrieval time from the measurement without claiming that
the replayed images are new live frames.
