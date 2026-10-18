# Local and VM processing with YOLOv8 (Parts 1, 2 and 4.2)

## Part 1: local

Capture a frame every 5 s, count people with YOLOv8n (960-px input,
confidence 0.25) and plot the counts over time:

```sh
python -m local_vm.process_local --count 10      # -> results/local.csv
python -m local_vm.plot_local                    # -> results/local_people_over_time.png
```

Other options: `--cam`, `--interval`, `--image-size`, `--confidence`, `--csv`.

## Part 2: Azure VM

The VM (Ubuntu 24.04, Standard B2ts v2, 1 GiB RAM) runs YOLOv8n as an ONNX
model with OpenCV DNN, which avoids installing PyTorch. Flask + Gunicorn serve
the API behind Apache on port 80, managed by systemd.

```sh
python -m local_vm.export_vm_model                                   # yolov8n.pt -> ONNX
VM_SSH_TARGET=azureuser@VM_IP VM_SSH_KEY=~/.ssh/azure_vm_key ./local_vm/deploy_vm.sh
```

The deploy script creates the upload token in
`~/.config/aiia-practice1/vm-upload-token`. Send frames from the local machine:

```sh
export AIIA_VM_URL=http://VM_IP
python -m local_vm.send_to_vm --count 10 --save-frames-dir frames/vm
```

- Dashboard: `http://VM_IP/` (histogram, last frame, CPU, RAM, bandwidth, latency, cost); raw data at `/api/dashboard`.
- The VM stores `results.csv` and `metrics.csv` in `/var/lib/aiia-part2/`.
- Detection accuracy against the manual counts in `report/data/part2_vm_ground_truth.csv`: `python -m local_vm.evaluate_vm_counts`

Response time compares the local capture time with the VM completion time, so
both clocks must be synchronized. The cost is the public hourly rate × uptime
(estimate, not an invoice).

## Part 4.2: several cameras

One upload thread per camera; `--persistent-streams` keeps each HLS stream
open so every camera really sends one frame every 5 s:

```sh
python -m local_vm.send_to_vm --persistent-streams --interval 5 --count 10 \
  --cams santa_pola benidorm_poniente madrid_alcala benalmadena calafell \
  --results-csv report/data/part4_vm_scale_results.csv
```

Add cameras gradually and read `/api/dashboard` after each batch. To measure
the VM alone, without depending on the webcams, `--replay-dir DIR` resends one
saved `<cam_id>.jpg` per camera. The VM serializes inference, so queueing shows
up as a growing response time.

Tests: `python -m unittest discover -s local_vm/tests -t .`
