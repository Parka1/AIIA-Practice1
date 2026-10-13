"""CPU inference helpers for a YOLOv8 model exported to ONNX."""

import time

import cv2
import numpy as np


def load_model(model_path: str):
    """Load an ONNX model with OpenCV DNN on the CPU."""
    net = cv2.dnn.readNetFromONNX(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net


def letterbox(frame: np.ndarray, image_size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize and pad an image to the square dimensions used by YOLO."""
    height, width = frame.shape[:2]
    scale = min(image_size / width, image_size / height)
    resized_width = round(width * scale)
    resized_height = round(height * scale)
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (image_size - resized_width) // 2
    pad_y = (image_size - resized_height) // 2
    padded = np.full((image_size, image_size, 3), 114, dtype=np.uint8)
    padded[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return padded, scale, pad_x, pad_y


def detect_people(
    net,
    frame: np.ndarray,
    confidence: float = 0.25,
    image_size: int = 960,
    nms_threshold: float = 0.45,
) -> tuple[list[tuple[int, int, int, int, float]], float]:
    """Return person boxes as xywh plus confidence, and inference time in ms."""
    height, width = frame.shape[:2]
    padded, scale, pad_x, pad_y = letterbox(frame, image_size)
    blob = cv2.dnn.blobFromImage(
        padded,
        scalefactor=1.0 / 255.0,
        size=(image_size, image_size),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    started = time.perf_counter()
    prediction = net.forward()
    inference_ms = (time.perf_counter() - started) * 1000

    output = np.asarray(prediction)
    if output.ndim == 3:
        if output.shape[0] != 1:
            raise RuntimeError(f"expected one model output batch, received shape {output.shape}")
        output = output[0]
    if output.ndim != 2:
        raise RuntimeError(f"unexpected YOLO output shape {output.shape}")
    if output.shape[0] == 84 and output.shape[1] != 84:
        output = output.T
    if output.shape[1] != 84:
        raise RuntimeError(f"expected YOLOv8 COCO output with 84 values per box, got {output.shape}")

    person_scores = output[:, 4]
    candidate_indices = np.flatnonzero(np.isfinite(person_scores) & (person_scores >= confidence))
    boxes = []
    scores = []
    for index in candidate_indices:
        center_x, center_y, box_width, box_height = output[index, :4]
        x1 = round((center_x - box_width / 2 - pad_x) / scale)
        y1 = round((center_y - box_height / 2 - pad_y) / scale)
        x2 = round((center_x + box_width / 2 - pad_x) / scale)
        y2 = round((center_y + box_height / 2 - pad_y) / scale)
        x1 = max(0, min(width, x1))
        y1 = max(0, min(height, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        scores.append(float(person_scores[index]))

    selected = cv2.dnn.NMSBoxes(boxes, scores, confidence, nms_threshold)
    selected_indices = np.asarray(selected).reshape(-1) if len(selected) else []
    detections = [
        (*boxes[index], scores[index])
        for index in selected_indices
    ]
    return detections, inference_ms


def draw_detections(
    frame: np.ndarray,
    detections: list[tuple[int, int, int, int, float]],
) -> np.ndarray:
    """Draw person boxes and confidence labels on a copy of the frame."""
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    label_rects = []
    for x, y, width, height, confidence in detections:
        cv2.rectangle(annotated, (x, y), (x + width, y + height), (40, 200, 80), 2)
        label = f"person {confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            scale,
            thickness,
        )
        box_width, box_height = text_width + 8, text_height + baseline + 6
        label_x = max(2, min(x, frame.shape[1] - box_width - 2))
        preferred_y = max(2, y - box_height - 3)
        candidates = [preferred_y]
        for left, top, right, bottom in label_rects:
            if label_x < right and label_x + box_width > left:
                candidates.extend((top - box_height - 3, bottom + 3))
        label_y = next(
            (
                candidate
                for candidate in sorted(candidates, key=lambda value: abs(value - preferred_y))
                if candidate >= 2
                and candidate + box_height < frame.shape[0] - 2
                and all(
                    label_x + box_width <= left
                    or label_x >= right
                    or candidate + box_height <= top
                    or candidate >= bottom
                    for left, top, right, bottom in label_rects
                )
            ),
            max(2, min(preferred_y, frame.shape[0] - box_height - 2)),
        )
        label_rects.append((label_x, label_y, label_x + box_width, label_y + box_height))
        cv2.rectangle(
            annotated,
            (label_x, label_y),
            (label_x + box_width, label_y + box_height),
            (40, 200, 80),
            cv2.FILLED,
        )
        cv2.putText(
            annotated,
            label,
            (label_x + 4, label_y + text_height + 2),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return annotated
