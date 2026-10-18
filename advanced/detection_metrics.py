"""Box-level and count-level detection metrics against the manual ground-truth boxes.

Boxes are (x1, y1, x2, y2) in pixels of the original 1920x1080 frame. A detection
matches a ground-truth person when their IoU is at least MATCH_IOU; each person
and each detection can be matched once (greedy, highest IoU first). The threshold
is lower than the usual 0.5 because the people are only 10-40 pixels tall and the
manual boxes are approximate.
"""

from dataclasses import dataclass

MATCH_IOU = 0.3

Box = tuple[float, float, float, float]


@dataclass
class FrameScore:
    true_positives: int
    detections: int
    people: int

    @property
    def count_error(self) -> int:
        return self.detections - self.people


def iou(a: Box, b: Box) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def score_frame(detections: list[Box], ground_truth: list[Box], min_iou: float = MATCH_IOU) -> FrameScore:
    pairs = sorted(
        ((iou(d, g), i, j) for i, d in enumerate(detections) for j, g in enumerate(ground_truth)),
        reverse=True,
    )
    used_d, used_g = set(), set()
    for value, i, j in pairs:
        if value < min_iou:
            break
        if i not in used_d and j not in used_g:
            used_d.add(i)
            used_g.add(j)
    return FrameScore(len(used_d), len(detections), len(ground_truth))


def summarize(scores: list[FrameScore]) -> dict[str, float]:
    tp = sum(s.true_positives for s in scores)
    detections = sum(s.detections for s in scores)
    people = sum(s.people for s in scores)
    errors = [s.count_error for s in scores]
    return {
        "frames": len(scores),
        "recall": tp / people if people else 0.0,
        "precision": tp / detections if detections else 0.0,
        "count_mae": sum(abs(e) for e in errors) / len(errors),
        "count_bias": sum(errors) / len(errors),
        "mean_detected": detections / len(scores),
    }


def scale_boxes(boxes: list[Box], factor: float) -> list[Box]:
    return [tuple(v * factor for v in box) for box in boxes]
