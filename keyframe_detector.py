"""从 C++ 部署代码迁移的关键帧检测流程。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from predict_view_cls import CudaRuntime, PROJECT_ROOT, TrtRunner


KEYFRAME_LGTA_BACKBONE_ENGINE = (
    PROJECT_ROOT / "resources" / "engine" / "keyframe" / "A2C_A4C" / "lgta_backbone_20250401_v2.engine"
)
KEYFRAME_LGTA_ENGINE = PROJECT_ROOT / "resources" / "engine" / "keyframe" / "A2C_A4C" / "lgta_20250401_v2.engine"
KEYFRAME_PLAX_BACKBONE_ENGINE = PROJECT_ROOT / "resources" / "engine" / "keyframe" / "PLAX" / "plax_backbone.engine"
KEYFRAME_PLAX_SGTA_ENGINE = PROJECT_ROOT / "resources" / "engine" / "keyframe" / "PLAX" / "plax_sgta.engine"

KEYFRAME_LGTA_VIEWS = {"A2C", "A3C", "A4C", "A5C"}
KEYFRAME_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
KEYFRAME_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
KEYFRAME_PLAX_MEAN = np.array([114.7748, 107.7354, 99.475], dtype=np.float32)
KEYFRAME_PLAX_STD = np.array([1.0, 1.0, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class PeakInfo:
    """关键帧峰值信息；正索引表示 ED，负索引表示 ES。"""

    index: int
    value: float
    width: float = 1.0
    prominence: float = 0.0


@dataclass(frozen=True)
class KeyframeResult:
    """单段视频窗口的关键帧检测结果。"""

    peaks: list[PeakInfo]
    ed_probs: list[float]
    es_probs: list[float]


def raw_frame_to_keyframe_input(
    frame: np.ndarray,
    expected_shape: tuple[int, ...],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """将 RGB ROI 帧转换为关键帧 backbone 输入张量。"""

    if len(expected_shape) != 4 or frame.ndim != 3:
        raise ValueError(f"unsupported keyframe frame shape {frame.shape}; expected {expected_shape}")

    _, channels, height, width = expected_shape
    if frame.shape[2] != channels:
        raise ValueError(f"unsupported keyframe frame channels {frame.shape[2]}; expected {channels}")

    # C++ 侧输入是 OpenCV BGR Mat，当前 Python 视频/ROI 流程内部使用 RGB。
    bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    array = (array - mean) / std
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def find_local_maxima(values: list[float] | np.ndarray) -> list[int]:
    """按 C++ 逻辑寻找局部极大值，不强制首尾帧作为峰值。"""

    arr = np.asarray(values, dtype=np.float32)
    if arr.size < 3:
        return []

    peaks: list[int] = []
    i = 1
    while i < arr.size - 1:
        if arr[i - 1] < arr[i]:
            ahead = i + 1
            while ahead < arr.size - 1 and arr[ahead] == arr[i]:
                ahead += 1
            if ahead < arr.size and arr[ahead] < arr[i]:
                peaks.append((i + ahead - 1) // 2)
                i = ahead
        i += 1
    return peaks


def filter_peaks_by_distance(peaks: list[PeakInfo], distance: int) -> list[PeakInfo]:
    """当两个峰值距离过近时，只保留置信度更高的峰值。"""

    if not peaks:
        return []

    selected = [peaks[0]]
    for peak in peaks:
        if abs(peak.index - selected[-1].index) >= distance:
            selected.append(peak)
        elif selected[-1].value < peak.value:
            selected[-1] = peak
    return selected


def find_peaks(values: list[float] | np.ndarray, height: float, distance: int, threshold: float = 0.01) -> list[PeakInfo]:
    """使用与 C++ 兼容的高度和距离条件筛选关键帧概率峰值。"""

    arr = np.asarray(values, dtype=np.float32)
    peak_infos = [PeakInfo(index=int(idx), value=float(arr[idx])) for idx in find_local_maxima(arr)]
    peak_infos = filter_peaks_by_distance(peak_infos, distance)
    return [peak for peak in peak_infos if peak.value >= height]


def check_prob_close(probs: list[float]) -> float:
    """返回概率最大最小差，用于过滤过于平坦的概率曲线。"""

    if not probs:
        return 0.0
    return float(np.max(probs) - np.min(probs))


def average_query_probs(total_probs: list[list[float]], sta_length: int, lgta: bool) -> list[float]:
    """将每次 query 的概率回填到对应帧并求平均。"""

    if not total_probs:
        return []

    each_frame_probs: list[list[float]] = [[] for _ in total_probs]
    for frame_idx, probs in enumerate(total_probs):
        counter = 0
        if lgta:
            offsets = range(-(sta_length - 1), 1)
        else:
            offsets = range(-sta_length // 2, sta_length // 2 + 1)

        for offset in offsets:
            if counter >= len(probs):
                break
            sample_idx = min(max(frame_idx + offset, 0), len(total_probs) - 1)
            each_frame_probs[sample_idx].append(float(probs[counter]))
            counter += 1

    return [float(np.mean(probs)) if probs else 0.0 for probs in each_frame_probs]


class KeyframeInferer:
    """单个切面族使用的有状态 backbone + SGTA/LGTA 关键帧检测器。"""

    def __init__(
        self,
        backbone: TrtRunner,
        temporal_head: TrtRunner,
        sta_length: int,
        memory_length: int,
        ed_only: bool = True,
        avg_prob: bool = False,
        lgta: bool = False,
        mean: np.ndarray = KEYFRAME_IMAGE_MEAN,
        std: np.ndarray = KEYFRAME_IMAGE_STD,
    ) -> None:
        self.backbone = backbone
        self.temporal_head = temporal_head
        self.sta_length = sta_length
        self.memory_length = memory_length
        self.ed_only = ed_only
        self.avg_prob = avg_prob
        self.lgta = lgta
        self.mean = mean
        self.std = std
        self.features: list[np.ndarray] = []
        self.frames: list[np.ndarray] = []

    def clear(self) -> None:
        """清空累计的帧和特征缓存。"""

        self.features.clear()
        self.frames.clear()

    def add_frame(self, frame: np.ndarray) -> None:
        """对单帧运行 backbone，并缓存特征张量。"""

        input_shape = self.backbone.tensor_shapes[self.backbone.single_input_name]
        backbone_input = raw_frame_to_keyframe_input(frame, input_shape, self.mean, self.std)
        outputs = self.backbone.infer(backbone_input)
        feat = outputs[self.backbone.single_output_name]
        self.features.append(np.ascontiguousarray(feat.reshape(self._feature_shape()), dtype=np.float32))
        self.frames.append(frame.copy())

    def infer_clip(self, frame: np.ndarray | None = None) -> KeyframeResult:
        """可选追加当前帧，然后运行时序关键帧推理。"""

        if frame is not None:
            self.add_frame(frame)

        if not self.features:
            return KeyframeResult(peaks=[], ed_probs=[], es_probs=[])

        mem_feats = self._memory_features()
        total_ed_probs: list[list[float]] = []
        total_es_probs: list[list[float]] = []
        ed_probs: list[float] = []
        es_probs: list[float] = []

        for index in range(len(self.features)):
            query_feats = self._query_features(index)
            outputs = self.temporal_head.infer(
                {
                    self._memory_input_name(): mem_feats,
                    self._query_input_name(): query_feats,
                }
            )
            prob = next(iter(outputs.values())).reshape(-1).astype(np.float32)

            if not self.avg_prob:
                if self.ed_only:
                    ed_probs.append(float(prob[0]))
                else:
                    ed_probs.append(float(prob[1]))
                    es_probs.append(float(prob[0]))
            elif self.ed_only:
                total_ed_probs.append([float(v) for v in prob[: self.sta_length]])
            else:
                total_ed_probs.append([float(v) for v in prob[: self.sta_length]])
                total_es_probs.append([float(v) for v in prob[self.sta_length : self.sta_length * 2]])

        if self.avg_prob:
            ed_probs = average_query_probs(total_ed_probs, self.sta_length, self.lgta)
            es_probs = average_query_probs(total_es_probs, self.sta_length, self.lgta) if not self.ed_only else []

        return KeyframeResult(peaks=self._find_keyframe_peaks(ed_probs, es_probs), ed_probs=ed_probs, es_probs=es_probs)

    def _memory_input_name(self) -> str:
        return "memory_feat" if "memory_feat" in self.temporal_head.input_names else self.temporal_head.input_names[0]

    def _query_input_name(self) -> str:
        if "query_feat" in self.temporal_head.input_names:
            return "query_feat"
        return self.temporal_head.input_names[1 if self.temporal_head.input_names[0] == self._memory_input_name() else 0]

    def _feature_shape(self) -> tuple[int, int, int]:
        """从 SGTA/LGTA 时序头输入中反推 C,H,W 特征形状。"""

        memory_shape = self.temporal_head.tensor_shapes[self._memory_input_name()]
        if len(memory_shape) < 4:
            raise ValueError(f"unsupported keyframe memory input shape: {memory_shape}")
        return tuple(int(dim) for dim in memory_shape[-3:])

    def _with_expected_batch(self, array: np.ndarray, input_name: str) -> np.ndarray:
        """当时序头 engine 需要 batch 维时自动补齐。"""

        expected_shape = self.temporal_head.tensor_shapes[input_name]
        if array.shape == expected_shape:
            return np.ascontiguousarray(array, dtype=np.float32)
        if len(expected_shape) == array.ndim + 1 and expected_shape[0] == 1 and array.shape == expected_shape[1:]:
            return np.ascontiguousarray(array[None, ...], dtype=np.float32)
        return np.ascontiguousarray(array, dtype=np.float32)

    def _memory_features(self) -> np.ndarray:
        selected: list[np.ndarray] = []
        if self.lgta:
            for i in range(0, self.memory_length * 2, 2):
                selected.append(self.features[min(i, len(self.features) - 1)])
                if len(selected) == self.memory_length:
                    break
        else:
            for i in range(self.memory_length):
                selected.append(self.features[min(i, len(self.features) - 1)])

        memory = np.stack(selected, axis=0)
        return self._with_expected_batch(memory, self._memory_input_name())

    def _query_features(self, index: int) -> np.ndarray:
        selected: list[np.ndarray] = []
        if self.lgta:
            for i in range(self.sta_length - 1, -1, -1):
                selected.append(self.features[min(max(index - i, 0), len(self.features) - 1)])
        else:
            sample_num = self.sta_length // 2
            for offset in range(-sample_num, sample_num + 1):
                selected.append(self.features[min(max(index + offset, 0), len(self.features) - 1)])

        query = np.stack(selected, axis=0)
        return self._with_expected_batch(query, self._query_input_name())

    def _find_keyframe_peaks(self, ed_probs: list[float], es_probs: list[float]) -> list[PeakInfo]:
        if self.ed_only:
            ed_peaks = find_peaks(ed_probs, 0.5, 12 if self.lgta else 8, 0.01)
            neg_probs = [-value for value in ed_probs]
            es_peaks = [
                PeakInfo(index=-peak.index, value=peak.value)
                for peak in find_peaks(neg_probs, -0.5, 12 if self.lgta else 8, 0.01)
            ]
            return es_peaks + ed_peaks

        if self.lgta and (check_prob_close(ed_probs) < 0.1 or check_prob_close(es_probs) < 0.1):
            return []

        height = 0.3 if self.lgta else 0.5
        distance = 12 if self.lgta else 8
        ed_peaks = find_peaks(ed_probs, height, distance, 0.01)
        es_peaks = [PeakInfo(index=-peak.index, value=peak.value) for peak in find_peaks(es_probs, height, distance, 0.01)]
        return es_peaks + ed_peaks


class KeyframeDetector:
    """按切面名称路由到 LGTA 或 PLAX 关键帧检测器。"""

    def __init__(self, trt, cudart: CudaRuntime, stream) -> None:
        lgta_backbone = TrtRunner(KEYFRAME_LGTA_BACKBONE_ENGINE, trt, cudart, stream)
        lgta_head = TrtRunner(KEYFRAME_LGTA_ENGINE, trt, cudart, stream)
        self.lgta_inferer = KeyframeInferer(lgta_backbone, lgta_head, 5, 5, ed_only=False, avg_prob=True, lgta=True)

        plax_backbone = TrtRunner(KEYFRAME_PLAX_BACKBONE_ENGINE, trt, cudart, stream)
        plax_head = TrtRunner(KEYFRAME_PLAX_SGTA_ENGINE, trt, cudart, stream)
        self.plax_inferer = KeyframeInferer(
            plax_backbone,
            plax_head,
            5,
            15,
            ed_only=True,
            avg_prob=False,
            lgta=False,
            mean=KEYFRAME_PLAX_MEAN,
            std=KEYFRAME_PLAX_STD,
        )

    def close(self) -> None:
        """释放所有关键帧检测器持有的 TensorRT 显存。"""

        self.lgta_inferer.backbone.close()
        self.lgta_inferer.temporal_head.close()
        self.plax_inferer.backbone.close()
        self.plax_inferer.temporal_head.close()

    def clear(self) -> None:
        """清空所有关键帧检测器的特征缓存。"""

        self.lgta_inferer.clear()
        self.plax_inferer.clear()

    def add_frame(self, frame: np.ndarray, view_name: str) -> None:
        """为指定切面族累计一帧特征。"""

        self._inferer_for_view(view_name).add_frame(frame)

    def infer_clip(self, view_name: str, frame: np.ndarray | None = None) -> KeyframeResult:
        """对指定切面族运行关键帧检测。"""

        return self._inferer_for_view(view_name).infer_clip(frame)

    def _inferer_for_view(self, view_name: str) -> KeyframeInferer:
        if view_name in KEYFRAME_LGTA_VIEWS:
            return self.lgta_inferer
        if view_name == "PLAX":
            return self.plax_inferer
        raise ValueError(f"unsupported keyframe view: {view_name}")
