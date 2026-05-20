"""视频 ROI 检测、B 模式过滤和 ROI 裁剪工具。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


DEFAULT_ROI_CONF_THRESHOLD = 0.25
DEFAULT_ROI_CLASS_ID = 0
DEFAULT_AXIS_CLASS_ID = 1
DEFAULT_COLOR_RATIO_THRESHOLD = 0.005


class RoiPreprocessError(ValueError):
    """当视频帧无法转换为 ROI 检测模型输入时抛出。"""

    pass


@dataclass(frozen=True)
class RoiDetection:
    """一条已恢复到原图坐标系的 ROI 检测结果。"""

    box: tuple[int, int, int, int]
    score: float
    class_id: int


def raw_frame_to_detector_input(frame: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    """将 RGB 视频帧缩放并转换为 ROI 检测模型输入张量。"""

    if len(expected_shape) != 4 or frame.ndim != 3:
        raise RoiPreprocessError(
            f"unsupported ROI frame shape {frame.shape}; expected engine shape {expected_shape}"
        )

    _, channels, height, width = expected_shape
    if frame.shape[2] != channels:
        raise RoiPreprocessError(
            f"unsupported ROI frame channels {frame.shape[2]}; expected {channels}"
        )

    resized = cv2.resize(frame.astype(np.uint8), (width, height), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def clip_box(box: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int] | None:
    """将浮点检测框裁剪到图像边界内。"""

    x1, y1, x2, y2 = box
    x1 = int(max(0, min(width - 1, round(x1))))
    y1 = int(max(0, min(height - 1, round(y1))))
    x2 = int(max(0, min(width, round(x2))))
    y2 = int(max(0, min(height, round(y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def decode_detection_row(
    row: np.ndarray,
    frame_width: int,
    frame_height: int,
    input_width: int,
    input_height: int,
) -> tuple[tuple[int, int, int, int], float, int] | None:
    """解码单条检测输出，并将检测框恢复到原图坐标系。"""

    values = row.astype(np.float32).reshape(-1)
    if values.size < 6:
        return None

    if not np.isfinite(values).all():
        return None

    coords = values[:4]
    if values.size == 7:
        object_conf = float(values[4])
        class_scores = values[5:]
        class_id = int(np.argmax(class_scores))
        score = object_conf * float(class_scores[class_id])

        if np.max(coords) <= 1.5:
            coords = coords * np.array([frame_width, frame_height, frame_width, frame_height], dtype=np.float32)
        else:
            coords = coords * np.array(
                [
                    frame_width / float(input_width),
                    frame_height / float(input_height),
                    frame_width / float(input_width),
                    frame_height / float(input_height),
                ],
                dtype=np.float32,
            )

        cx, cy, w, h = (float(v) for v in coords)
        clipped = clip_box(
            (
                cx - w / 2.0,
                cy - h / 2.0,
                cx + w / 2.0,
                cy + h / 2.0,
            ),
            frame_width,
            frame_height,
        )
        if clipped is None:
            return None
        return clipped, score, class_id

    if values.size == 6 and abs(float(values[5]) - round(float(values[5]))) <= 1e-4:
        score = float(values[4])
        class_id = int(round(float(values[5])))
    else:
        class_scores = values[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])

    if np.max(coords) <= 1.5:
        coords = coords * np.array([frame_width, frame_height, frame_width, frame_height], dtype=np.float32)
    else:
        coords = coords * np.array(
            [
                frame_width / float(input_width),
                frame_height / float(input_height),
                frame_width / float(input_width),
                frame_height / float(input_height),
            ],
            dtype=np.float32,
        )

    x1, y1, x2, y2 = (float(v) for v in coords)
    if x2 <= x1 or y2 <= y1:
        cx, cy, w, h = x1, y1, x2, y2
        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0

    clipped = clip_box((x1, y1, x2, y2), frame_width, frame_height)
    if clipped is None:
        return None
    return clipped, score, class_id


def iter_detection_rows(outputs: dict[str, np.ndarray]):
    """从常见检测模型输出格式中逐条取出候选框。"""

    for output in outputs.values():
        array = np.asarray(output)
        squeezed = np.squeeze(array)
        if squeezed.ndim == 1:
            if squeezed.size >= 6:
                yield squeezed[:6]
        elif squeezed.ndim == 2:
            rows = None
            if squeezed.shape[1] >= 6:
                rows = squeezed
            elif squeezed.shape[0] >= 6:
                rows = squeezed.T

            if rows is None:
                continue

            for row in rows:
                yield row


def get_yolo_rows_by_best_class(outputs: dict[str, np.ndarray]) -> list[tuple[np.ndarray, int, float]]:
    """对 YOLO 原始输出，每个类别只保留置信度最高的一条候选。"""

    selected: list[tuple[np.ndarray, int, float]] = []
    for output in outputs.values():
        rows = np.squeeze(np.asarray(output))
        if rows.ndim != 2:
            continue
        if rows.shape[1] < 7 and rows.shape[0] >= 7:
            rows = rows.T
        if rows.ndim != 2 or rows.shape[1] != 7:
            continue

        object_conf = rows[:, 4]
        class_scores = rows[:, 5:]
        combined_scores = object_conf[:, None] * class_scores
        for class_id in range(combined_scores.shape[1]):
            row_index = int(np.argmax(combined_scores[:, class_id]))
            selected.append((rows[row_index], class_id, float(combined_scores[row_index, class_id])))
    return selected


def color_doppler_ratio(frame: np.ndarray) -> float:
    """估计中心区域内强红/蓝彩色多普勒像素比例。"""

    h, w = frame.shape[:2]
    roi = frame[h // 8 : h * 7 // 8, w // 8 : w * 7 // 8, :3].astype(np.int16)
    r = roi[..., 0]
    g = roi[..., 1]
    b = roi[..., 2]
    max_channel = np.maximum(np.maximum(r, g), b)
    min_channel = np.minimum(np.minimum(r, g), b)
    saturation = max_channel - min_channel

    red_pixels = (r > g + 40) & (r > b + 40)
    blue_pixels = (b > r + 40) & (b > g + 40)
    color_pixels = (max_channel > 60) & (saturation > 50) & (red_pixels | blue_pixels)
    return float(np.mean(color_pixels))


def is_color_doppler_frame(frame: np.ndarray, threshold: float = DEFAULT_COLOR_RATIO_THRESHOLD) -> bool:
    return color_doppler_ratio(frame) >= threshold


class VideoRoiPreprocessor:
    """过滤非 B 模式帧，并裁剪检测到的超声 ROI。"""

    def __init__(
        self,
        detector,
        target_class_id: int = DEFAULT_ROI_CLASS_ID,
        axis_class_id: int = DEFAULT_AXIS_CLASS_ID,
        conf_threshold: float = DEFAULT_ROI_CONF_THRESHOLD,
        color_ratio_threshold: float = DEFAULT_COLOR_RATIO_THRESHOLD,
        skip_no_roi_frames: bool = True,
        b_mode_only: bool = True,
        roi_preview_callback: Callable[[np.ndarray, np.ndarray, tuple[int, int, int, int], float], None] | None = None,
    ) -> None:
        self.detector = detector
        self.target_class_id = target_class_id
        self.axis_class_id = axis_class_id
        self.conf_threshold = conf_threshold
        self.color_ratio_threshold = color_ratio_threshold
        self.skip_no_roi_frames = skip_no_roi_frames
        self.b_mode_only = b_mode_only
        self.roi_preview_callback = roi_preview_callback
        self.last_filter_reason: str | None = None

    def process(self, frame: np.ndarray) -> np.ndarray | None:
        """返回裁剪后的 B 模式 ROI；需要过滤该帧时返回 None。"""

        self.last_filter_reason = None
        if self.b_mode_only and is_color_doppler_frame(frame, self.color_ratio_threshold):
            self.last_filter_reason = "color_doppler"
            return None

        detections = self.detect(frame)
        if self.b_mode_only and any(detection.class_id == self.axis_class_id for detection in detections):
            self.last_filter_reason = "spectrum_axis"
            return None

        roi_candidates = [detection for detection in detections if detection.class_id == self.target_class_id]
        detection = max(roi_candidates, key=lambda item: item.score, default=None)
        if detection is None:
            self.last_filter_reason = "no_roi"
            return None if self.skip_no_roi_frames else frame

        x1, y1, x2, y2 = detection.box
        cropped = frame[y1:y2, x1:x2].copy()
        if self.roi_preview_callback is not None:
            self.roi_preview_callback(frame, cropped, detection.box, detection.score)
        return cropped

    def detect(self, frame: np.ndarray) -> list[RoiDetection]:
        """运行 ROI 检测模型，返回原图坐标系下通过阈值的检测结果。"""

        input_shape = self.detector.tensor_shapes[self.detector.single_input_name]
        detector_input = raw_frame_to_detector_input(frame, input_shape)
        outputs = self.detector.infer(detector_input)
        height, width = frame.shape[:2]
        _, _, input_height, input_width = input_shape

        detections: list[RoiDetection] = []
        best_rows = get_yolo_rows_by_best_class(outputs)
        if best_rows:
            rows_to_decode = (row for row, _, _ in best_rows)
        else:
            rows_to_decode = iter_detection_rows(outputs)

        for row in rows_to_decode:
            decoded = decode_detection_row(row, width, height, input_width, input_height)
            if decoded is None:
                continue

            box, score, class_id = decoded
            if score < self.conf_threshold:
                continue

            detections.append(RoiDetection(box=box, score=score, class_id=class_id))

        return detections

    def detect_roi(self, frame: np.ndarray) -> RoiDetection | None:
        """返回单帧中置信度最高的 ROI 类检测结果。"""

        roi_candidates = [detection for detection in self.detect(frame) if detection.class_id == self.target_class_id]
        return max(roi_candidates, key=lambda item: item.score, default=None)
