"""调试显示与模型输出格式化工具。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ROI 调试辅助函数。


def show_roi_crop_preview(
    frame: np.ndarray,
    cropped: np.ndarray,
    box: tuple[int, int, int, int],
    score: float,
    window_name: str = "ROI crop preview",
) -> None:
    """显示原图 ROI 框和裁剪后的 ROI 图像。"""

    x1, y1, x2, y2 = box
    original_preview = frame.copy()
    cv2.rectangle(original_preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        original_preview,
        f"ROI {score:.3f}",
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    original_h, original_w = original_preview.shape[:2]
    crop_h, crop_w = cropped.shape[:2]
    combined_h = max(original_h, crop_h)
    combined_w = original_w + crop_w
    combined = np.zeros((combined_h, combined_w, 3), dtype=np.uint8)
    combined[:original_h, :original_w] = original_preview
    combined[:crop_h, original_w:original_w + crop_w] = cropped

    cv2.imshow(window_name, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def roi_output_desc(detector) -> str:
    """返回 ROI 检测模型输出张量的简要描述。"""

    return ", ".join(f"{name} {detector.tensor_shapes[name]}" for name in detector.output_names)


# ViewCls 调试辅助函数。


def show_viewcls_backbone_input(
    backbone_input: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    window_name: str = "ViewCls backbone input",
) -> None:
    """反归一化并显示实际送入 ViewCls backbone 的图像。"""

    image = backbone_input[0].transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


def summarize_raw_outputs(outputs: dict[str, np.ndarray]) -> str:
    """格式化未知 TensorRT 输出，便于控制台查看。"""

    parts: list[str] = []
    for name, value in outputs.items():
        flat = value.reshape(-1)
        if flat.size > 1:
            parts.append(f"{name}: shape={value.shape}, argmax={int(np.argmax(flat))}, values={flat.tolist()}")
        else:
            parts.append(f"{name}: shape={value.shape}, value={float(flat[0]):.6f}")
    return " | ".join(parts)


def summarize_swinhead_outputs(
    outputs: dict[str, np.ndarray],
    view_classes: tuple[str, ...],
    transition_classes: tuple[str, ...],
) -> str:
    """格式化 swinhead 的切面、质量和过渡帧输出。"""

    view_logits: np.ndarray | None = None
    quality_score: np.ndarray | None = None
    transition_logits: np.ndarray | None = None

    for value in outputs.values():
        flat = value.reshape(-1)
        if flat.size == len(view_classes):
            view_logits = flat
        elif flat.size == 1:
            quality_score = flat
        elif flat.size == len(transition_classes):
            transition_logits = flat

    if view_logits is None or quality_score is None or transition_logits is None:
        return summarize_raw_outputs(outputs)

    view_index = int(np.argmax(view_logits))
    transition_index = int(np.argmax(transition_logits))

    return (
        f"view={view_classes[view_index]} "
        f"view_score={float(view_logits[view_index]):.6f} "
        f"quality_score={float(quality_score[0]):.6f} "
        f"transition={transition_classes[transition_index]}"
    )


def backbone_output_desc(backbone) -> str:
    """返回 backbone 输出张量的简要描述。"""

    name = backbone.single_output_name
    return f"{name} {backbone.tensor_shapes[name]}"


# KeyFrame 调试辅助函数。


def keyframe_label(peak: Any) -> str:
    """按 C++ 的正负索引约定返回 ED 或 ES 标签。"""

    return "ED" if int(peak.index) >= 0 else "ES"


def keyframe_frame_index(peak: Any) -> int:
    """从带符号峰值索引中取出真实帧序号。"""

    return abs(int(peak.index))


def summarize_keyframe_result(result: Any) -> str:
    """格式化关键帧峰值和概率曲线，便于控制台查看。"""

    peaks = getattr(result, "peaks", [])
    ed_probs = getattr(result, "ed_probs", [])
    es_probs = getattr(result, "es_probs", [])
    peak_parts = [
        f"{keyframe_label(peak)}@{keyframe_frame_index(peak)}={float(peak.value):.6f}"
        for peak in peaks
    ]
    return (
        f"peaks=[{', '.join(peak_parts)}] "
        f"ed_probs={_format_float_list(ed_probs)} "
        f"es_probs={_format_float_list(es_probs)}"
    )


def annotate_keyframe_image(frame: np.ndarray, peak: Any) -> np.ndarray:
    """在单帧 RGB 图像上标注 ED/ES 关键帧信息。"""

    image = frame.copy()
    label = f"{keyframe_label(peak)} {keyframe_frame_index(peak)} {float(peak.value):.3f}"
    color = (0, 255, 0) if keyframe_label(peak) == "ED" else (255, 160, 0)
    cv2.putText(image, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    return image


def show_keyframe_result_images(
    frames: list[np.ndarray],
    result: Any,
    window_name: str = "KeyFrame result",
) -> None:
    """以拼图形式显示检测到的 ED/ES 关键帧。"""

    montage = build_keyframe_montage(frames, getattr(result, "peaks", []))
    if montage is None:
        return
    cv2.imshow(window_name, cv2.cvtColor(montage, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)


def save_keyframe_result_images(
    frames: list[np.ndarray],
    result: Any,
    output_dir: Path | str,
    prefix: str = "keyframe",
) -> list[Path]:
    """保存检测到的 ED/ES 关键帧图像，并返回文件路径。"""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for peak in getattr(result, "peaks", []):
        frame_index = keyframe_frame_index(peak)
        if frame_index >= len(frames):
            continue
        image = annotate_keyframe_image(frames[frame_index], peak)
        filename = output_path / f"{prefix}_{frame_index:04d}_{keyframe_label(peak)}.png"
        cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        written.append(filename)
    return written


def build_keyframe_montage(frames: list[np.ndarray], peaks: list[Any]) -> np.ndarray | None:
    """根据检测到的峰值帧构造 RGB 拼图。"""

    images: list[np.ndarray] = []
    for peak in peaks:
        frame_index = keyframe_frame_index(peak)
        if frame_index >= len(frames):
            continue
        image = annotate_keyframe_image(frames[frame_index], peak)
        images.append(cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA))

    if not images:
        return None

    return np.concatenate(images, axis=1)


def _format_float_list(values: list[float]) -> str:
    """压缩格式化概率数组，避免控制台输出过长。"""

    if not values:
        return "[]"
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"
