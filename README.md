# AIIA Practice 1: Smart Lighting with cloud computing

Counts people in public webcam frames (SkylineWebcams) to decide when to switch
on the street lights. The same capture code and CSV format are used by three
architectures, so their results can be compared frame by frame.

| Folder | What it contains |
|---|---|
| [`common/`](common/README.md) | Webcam capture, common results CSV, client-side optimizations |
| [`local_vm/`](local_vm/README.md) | Part 1 (YOLOv8 locally), Part 2 (YOLOv8 on an Azure VM) and Part 4.2 (VM with several cameras) |
| [`serverless/`](serverless/README.md) | Part 3 (Azure Function + AI Vision + static website) and Part 4.3 (serverless with several cameras) |
| [`advanced/`](advanced/README.md) | Advanced task 3: resolution, compression, motion detection, hysteresis |

## Setup

Python 3.12. Run every command from the repository root with the virtual
environment active:

```sh
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r local_vm/requirements.txt -r serverless/requirements.txt
```

Secrets are not in the repository: `credentials.env` (Azure keys, serverless
part) and the VM upload token are created locally, see each README.

Tests: `python -m unittest discover -p "test_*.py"`
