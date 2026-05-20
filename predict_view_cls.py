"""基于 TensorRT 的切面分类推理流程和工作线程。"""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import cv2

from debug_utils import backbone_output_desc, roi_output_desc, show_viewcls_backbone_input
from debug_utils import show_keyframe_result_images, summarize_keyframe_result, summarize_swinhead_outputs
from video_roi_preprocessor import VideoRoiPreprocessor


def get_project_root() -> Path:
    """返回普通 Python 运行或 PyInstaller onedir 打包后的项目根目录。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


PROJECT_ROOT = get_project_root()
BACKBONE_ENGINE = (
    PROJECT_ROOT
    / "resources"
    / "engine"
    / "view_classification"
    / "viewcls_backbone_20250425.engine"
)
SWINHEAD_ENGINE = (
    PROJECT_ROOT
    / "resources"
    / "engine"
    / "view_classification"
    / "viewcls_swinhead_20250425.engine"
)
ROI_ENGINE = PROJECT_ROOT / "resources" / "engine" / "roi_det" / "roi_detection_0328.engine"

WINDOW_SIZE = 24
VIEW_CLASSES = ("A2C", "A3C", "A4C", "A5C", "OTHER", "PLAX", "PSAXA", "PSAXGV", "PSAXMV", "PSAXPM")
TRANSITION_CLASSES = ("no", "yes")
DEFAULT_COLOR_RATIO_THRESHOLD = 0.005
DEFAULT_VALID_FRAME_THRESHOLD = 0.02
VIEWCLS_BGR_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
VIEWCLS_BGR_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class ViewClsInferenceConfig:
    """推理工作线程使用的运行时配置。"""

    video_path: Path
    prepared_input: Path | None
    roi_engine: Path | None
    backbone_engine: Path
    swinhead_engine: Path
    gpu: int
    max_frames: int | None
    max_windows: int | None
    skip_color_doppler: bool
    color_ratio_threshold: float
    valid_frame_threshold: float
    skip_invalid_frames: bool
    enable_roi_preprocess: bool
    roi_conf_threshold: float
    roi_class_id: int
    axis_class_id: int
    skip_no_roi_frames: bool
    b_mode_only: bool
    show_viewcls_input: bool
    enable_keyframe_detection: bool
    show_keyframe_images: bool


class PreparedInputRequiredError(ValueError):
    """当输入帧无法匹配 TensorRT 输入形状时抛出。"""

    pass


class CudaRuntime:
    """CUDA Runtime 的 ctypes 轻量封装，用于 TensorRT 显存缓冲区。"""

    HOST_TO_DEVICE = 1
    DEVICE_TO_HOST = 2

    def __init__(self, device_index: int) -> None:
        self.lib = self._load_library()
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p

        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaGetDevice.restype = ctypes.c_int
        self.lib.cudaMemGetInfo.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        self.lib.cudaMemGetInfo.restype = ctypes.c_int
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.set_device(device_index)

    @staticmethod
    def _load_library() -> ctypes.CDLL:
        if sys.platform == "win32":
            search_dirs: list[Path] = []

            cuda_path = os.environ.get("CUDA_PATH")
            if cuda_path:
                search_dirs.append(Path(cuda_path) / "bin")

            cuda_base = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")
            if cuda_base.is_dir():
                search_dirs.extend(sorted(cuda_base.glob(r"v*\bin"), reverse=True))

            conda_prefix = os.environ.get("CONDA_PREFIX")
            if conda_prefix:
                search_dirs.append(Path(conda_prefix) / "Library" / "bin")

            dll_names = [
                "cudart64_110.dll",  # CUDA 11.x，包括 CUDA 11.7
                "cudart64_12.dll",   # CUDA 12.x
            ]

            for dll_dir in search_dirs:
                if not dll_dir.is_dir():
                    continue

                os.add_dll_directory(str(dll_dir))
                os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")

                for dll_name in dll_names:
                    dll_path = dll_dir / dll_name
                    if dll_path.is_file():
                        print(f"[INFO] CUDA Runtime DLL: {dll_path}")
                        return ctypes.WinDLL(str(dll_path))

            for dll_name in dll_names:
                try:
                    print(f"[INFO] Trying CUDA Runtime DLL from PATH: {dll_name}")
                    return ctypes.WinDLL(dll_name)
                except FileNotFoundError:
                    pass

            raise FileNotFoundError(
                "Could not find CUDA Runtime DLL. "
                "Please add CUDA\\bin to PATH. "
                "Expected cudart64_110.dll or cudart64_12.dll."
            )

        return ctypes.CDLL("libcudart.so")

    def check(self, code: int, action: str) -> None:
        if code == 0:
            return
        message = self.lib.cudaGetErrorString(code).decode("utf-8", errors="replace")
        raise RuntimeError(f"{action} failed: cuda error {code}: {message}")

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr

    def set_device(self, device_index: int) -> None:
        self.check(self.lib.cudaSetDevice(device_index), "cudaSetDevice")

    def get_device(self) -> int:
        device = ctypes.c_int()
        self.check(self.lib.cudaGetDevice(ctypes.byref(device)), "cudaGetDevice")
        return int(device.value)

    def mem_info(self) -> tuple[int, int]:
        free_bytes = ctypes.c_size_t()
        total_bytes = ctypes.c_size_t()
        self.check(self.lib.cudaMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)), "cudaMemGetInfo")
        return int(free_bytes.value), int(total_bytes.value)

    def free(self, ptr: ctypes.c_void_p) -> None:
        if ptr.value:
            self.check(self.lib.cudaFree(ptr), "cudaFree")

    def create_stream(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def destroy_stream(self, stream: ctypes.c_void_p) -> None:
        if stream.value:
            self.check(self.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def synchronize(self, stream: ctypes.c_void_p) -> None:
        self.check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")

    def copy_htod_async(self, dst: ctypes.c_void_p, src: np.ndarray, stream: ctypes.c_void_p) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                dst,
                ctypes.c_void_p(src.ctypes.data),
                src.nbytes,
                self.HOST_TO_DEVICE,
                stream,
            ),
            "cudaMemcpyAsync H2D",
        )

    def copy_dtoh_async(self, dst: np.ndarray, src: ctypes.c_void_p, stream: ctypes.c_void_p) -> None:
        self.check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(dst.ctypes.data),
                src,
                dst.nbytes,
                self.DEVICE_TO_HOST,
                stream,
            ),
            "cudaMemcpyAsync D2H",
        )


def find_tensorrt_root() -> Path | None:
    candidates: list[Path] = []
    env_root = os.environ.get("TENSORRT_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    if sys.platform == "win32":
        base_dir = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit")
        if base_dir.is_dir():
            candidates.extend(sorted(base_dir.glob("TensorRT-*"), reverse=True))

    for candidate in candidates:
        if (candidate / "lib").is_dir() or (candidate / "bin").is_dir():
            return candidate.resolve()
    return None


def add_tensorrt_dll_dirs(tensorrt_root: Path | None) -> None:
    if sys.platform != "win32" or tensorrt_root is None:
        return

    path_entries: list[str] = []
    for subdir in ("lib", "bin"):
        dll_dir = tensorrt_root / subdir
        if dll_dir.is_dir():
            os.add_dll_directory(str(dll_dir))
            path_entries.append(str(dll_dir))

    if path_entries:
        os.environ["PATH"] = os.pathsep.join(path_entries + [os.environ.get("PATH", "")])


def load_tensorrt():
    add_tensorrt_dll_dirs(find_tensorrt_root())
    import tensorrt as trt

    return trt


def trt_dtype_to_numpy(trt, dtype):
    mapping = {
        trt.DataType.FLOAT: np.float32,
        trt.DataType.HALF: np.float16,
        trt.DataType.INT8: np.int8,
        trt.DataType.INT32: np.int32,
        trt.DataType.BOOL: np.bool_,
    }
    if hasattr(trt.DataType, "INT64"):
        mapping[trt.DataType.INT64] = np.int64
    if dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {dtype}")
    return mapping[dtype]


class TrtRunner:
    """单个 TensorRT engine 的执行器，持有固定的 GPU/CPU 缓冲区。"""

    def __init__(self, engine_path: Path, trt, cudart: CudaRuntime, stream: ctypes.c_void_p) -> None:
        if not engine_path.is_file():
            raise FileNotFoundError(engine_path)

        self.engine_path = engine_path
        self.trt = trt
        self.cudart = cudart
        self.stream = stream
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self.tensor_shapes: dict[str, tuple[int, ...]] = {}
        self.tensor_dtypes: dict[str, np.dtype] = {}
        self.device_buffers: dict[str, ctypes.c_void_p] = {}
        self.output_hosts: dict[str, np.ndarray] = {}

        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            shape = tuple(int(dim) for dim in self.engine.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise ValueError(f"Dynamic tensor shape is not configured: {name} {shape}")

            dtype = np.dtype(trt_dtype_to_numpy(self.trt, self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            device_ptr = self.cudart.malloc(nbytes)
            self.context.set_tensor_address(name, int(device_ptr.value))

            self.tensor_shapes[name] = shape
            self.tensor_dtypes[name] = dtype
            self.device_buffers[name] = device_ptr

            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            else:
                self.output_names.append(name)
                self.output_hosts[name] = np.empty(shape, dtype=dtype)

    def infer(self, inputs: dict[str, np.ndarray] | np.ndarray) -> dict[str, np.ndarray]:
        if isinstance(inputs, np.ndarray):
            if len(self.input_names) != 1:
                raise ValueError("Array input is only allowed for single-input engines")
            inputs = {self.input_names[0]: inputs}

        for name in self.input_names:
            if name not in inputs:
                raise KeyError(f"Missing input tensor: {name}")
            array = np.ascontiguousarray(inputs[name], dtype=self.tensor_dtypes[name])
            expected_shape = self.tensor_shapes[name]
            if array.shape != expected_shape:
                raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {array.shape}")
            self.cudart.copy_htod_async(self.device_buffers[name], array, self.stream)

        ok = self.context.execute_async_v3(stream_handle=int(self.stream.value))
        if not ok:
            raise RuntimeError(f"TensorRT execution failed: {self.engine_path}")

        for name in self.output_names:
            self.cudart.copy_dtoh_async(self.output_hosts[name], self.device_buffers[name], self.stream)

        self.cudart.synchronize(self.stream)
        return {name: output.copy() for name, output in self.output_hosts.items()}

    @property
    def single_output_name(self) -> str:
        if len(self.output_names) != 1:
            raise ValueError(f"Engine has {len(self.output_names)} outputs, not 1")
        return self.output_names[0]

    @property
    def single_input_name(self) -> str:
        if len(self.input_names) != 1:
            raise ValueError(f"Engine has {len(self.input_names)} inputs, not 1")
        return self.input_names[0]

    def close(self) -> None:
        for ptr in self.device_buffers.values():
            self.cudart.free(ptr)
        self.device_buffers.clear()


def iter_video_frames(video_path: Path):
    """使用 OpenCV 逐帧读取视频，并输出 RGB 图像。"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            yield frame_rgb
    finally:
        cap.release()


def iter_prepared_inputs(path: Path):
    """从单个文件或目录中读取已经准备好的 .npy 输入张量。"""

    if path.is_dir():
        files = sorted(path.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found under: {path}")
        for file in files:
            array = np.load(file)
            yield from split_prepared_array(array)
        return

    array = np.load(path)
    yield from split_prepared_array(array)


def split_prepared_array(array: np.ndarray):
    """将支持的 prepared-input 数组格式拆成单帧 backbone 输入。"""

    if array.ndim == 3:
        yield array
    elif array.ndim == 4:
        if array.shape[0] == 1 and array.shape[1] == 3:
            yield array
        else:
            for item in array:
                yield item
    elif array.ndim == 5:
        for item in array:
            yield item
    else:
        raise ValueError(f"Unsupported prepared input array shape: {array.shape}")


def frame_to_backbone_input(frame: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    """接收已预处理张量或等价 HWC 图像，并转为 backbone 输入。"""

    if frame.shape == expected_shape:
        return np.ascontiguousarray(frame, dtype=np.float32)

    if len(expected_shape) == 4:
        _, channels, height, width = expected_shape
        if frame.shape == (height, width, channels):
            return np.ascontiguousarray(frame.astype(np.float32).transpose(2, 0, 1)[None, ...])

    raise PreparedInputRequiredError(
        "unsupported backbone input; expected prepared tensor, HWC equivalent, or raw HWC frame "
        f"for shape {expected_shape}, got {frame.shape}"
    )


def raw_frame_to_backbone_input(frame: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    """将 RGB ROI 帧转换为 ViewCls backbone 需要的归一化 NCHW 张量。"""

    if len(expected_shape) != 4 or frame.ndim != 3:
        raise PreparedInputRequiredError(
            f"unsupported raw frame shape {frame.shape}; expected engine shape {expected_shape}"
        )

    _, channels, height, width = expected_shape
    if frame.shape[2] != channels:
        raise PreparedInputRequiredError(
            f"unsupported raw frame channels {frame.shape[2]}; expected {channels}"
        )

    # 对齐部署预处理：RGB ROI -> BGR -> resize -> ImageNet 均值方差归一化。
    bgr = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
    resized = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_LINEAR)
    array = resized.astype(np.float32) / 255.0
    array = (array - VIEWCLS_BGR_MEAN) / VIEWCLS_BGR_STD
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None, ...], dtype=np.float32)


def decode_swinhead_outputs(
    outputs: dict[str, np.ndarray],
) -> tuple[str, float, float, str]:
    """将 swinhead 输出解码为切面类别、质量分数和过渡帧标签。"""

    view_logits: np.ndarray | None = None
    quality_score: np.ndarray | None = None
    transition_logits: np.ndarray | None = None

    for value in outputs.values():
        flat = value.reshape(-1)
        if flat.size == len(VIEW_CLASSES):
            view_logits = flat
        elif flat.size == 1:
            quality_score = flat
        elif flat.size == len(TRANSITION_CLASSES):
            transition_logits = flat

    if view_logits is None or quality_score is None or transition_logits is None:
        summary = summarize_swinhead_outputs(outputs, VIEW_CLASSES, TRANSITION_CLASSES)
        raise ValueError(f"Unsupported swinhead outputs: {summary}")

    view_index = int(np.argmax(view_logits))
    transition_index = int(np.argmax(transition_logits))
    return (
        VIEW_CLASSES[view_index],
        float(view_logits[view_index]),
        float(quality_score[0]),
        TRANSITION_CLASSES[transition_index],
    )


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


def is_color_doppler_frame(frame: np.ndarray, threshold: float) -> bool:
    return color_doppler_ratio(frame) >= threshold


def valid_ultrasound_ratio(frame: np.ndarray) -> float:
    """估计中心区域是否包含足够的非黑色超声内容。"""

    gray = frame[..., :3].mean(axis=2)
    h, w = gray.shape
    roi = gray[h // 8 : h * 7 // 8, w // 8 : w * 7 // 8]
    return float(np.mean(roi > 20.0))


def is_valid_ultrasound_frame(frame: np.ndarray, threshold: float) -> bool:
    return valid_ultrasound_ratio(frame) >= threshold


def predict_video(
    video_path: Path,
    backbone: TrtRunner,
    swinhead: TrtRunner,
    max_frames: int | None,
    max_windows: int | None,
    skip_color_doppler: bool,
    color_ratio_threshold: float,
    valid_frame_threshold: float,
    skip_invalid_frames: bool,
    frame_preprocessor: Callable[[np.ndarray], np.ndarray | None] | None = None,
    show_viewcls_input: bool = False,
    keyframe_detector=None,
    show_keyframe_images: bool = False,
) -> None:
    """运行视频推理；只有连续有效的 B 模式 ROI 帧才会组成 24 帧窗口。"""

    backbone_output_name = backbone.single_output_name
    feature_window: list[np.ndarray] = []
    roi_frame_window: list[np.ndarray] = []
    frame_count = 0
    window_count = 0
    skipped_color_frames = 0
    skipped_invalid_frames = 0
    skipped_no_roi_frames = 0
    skipped_spectrum_frames = 0
    skipped_bmode_color_frames = 0
    color_segment_start: int | None = None
    invalid_segment_start: int | None = None

    print(f"[INFO] reading video: {video_path}")
    for frame in iter_video_frames(video_path):
        frame_count += 1
        if skip_invalid_frames and not is_valid_ultrasound_frame(frame, valid_frame_threshold):
            skipped_invalid_frames += 1
            if invalid_segment_start is None:
                invalid_segment_start = frame_count
                if feature_window:
                    print(
                        f"[INVALID] frame {frame_count}: invalid ultrasound frame, "
                        f"discarding {len(feature_window)} pending backbone features"
                    )
                    feature_window.clear()
                    roi_frame_window.clear()
            if max_frames is not None and frame_count >= max_frames:
                break
            continue

        if invalid_segment_start is not None:
            print(f"[INVALID] skipped frames {invalid_segment_start}-{frame_count - 1}")
            invalid_segment_start = None

        if skip_color_doppler and is_color_doppler_frame(frame, color_ratio_threshold):
            skipped_color_frames += 1
            if color_segment_start is None:
                color_segment_start = frame_count
                if feature_window:
                    print(
                        f"[COLOR] frame {frame_count}: color Doppler detected, "
                        f"discarding {len(feature_window)} pending backbone features"
                    )
                    feature_window.clear()
                    roi_frame_window.clear()
            if max_frames is not None and frame_count >= max_frames:
                break
            continue

        if color_segment_start is not None:
            print(f"[COLOR] skipped frames {color_segment_start}-{frame_count - 1}")
            color_segment_start = None

        if frame_preprocessor is not None:
            frame = frame_preprocessor(frame)
            if frame is None:
                owner = getattr(frame_preprocessor, "__self__", None)
                reason = getattr(owner, "last_filter_reason", None)
                if reason == "color_doppler":
                    skipped_bmode_color_frames += 1
                elif reason == "spectrum_axis":
                    skipped_spectrum_frames += 1
                else:
                    skipped_no_roi_frames += 1

                if feature_window:
                    print(
                        f"[ROI] frame {frame_count}: filtered by {reason or 'roi_preprocess'}, "
                        f"discarding {len(feature_window)} pending backbone features"
                    )
                    feature_window.clear()
                    roi_frame_window.clear()
                if max_frames is not None and frame_count >= max_frames:
                    break
                continue

        try:
            backbone_input = raw_frame_to_backbone_input(frame, backbone.tensor_shapes[backbone.single_input_name])
            if show_viewcls_input:
                show_viewcls_backbone_input(backbone_input, VIEWCLS_BGR_MEAN, VIEWCLS_BGR_STD)
        except PreparedInputRequiredError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            break
        backbone_outputs = backbone.infer(backbone_input)
        feature_window.append(backbone_outputs[backbone_output_name][0])
        roi_frame_window.append(frame.copy())

        if len(feature_window) == WINDOW_SIZE:
            swinhead_input = np.ascontiguousarray(np.stack(feature_window, axis=0), dtype=np.float32)
            swinhead_outputs = swinhead.infer(swinhead_input)
            view_name, view_score, quality_score, transition_name = decode_swinhead_outputs(swinhead_outputs)
            window_count += 1
            start_frame = frame_count - WINDOW_SIZE + 1
            print(
                f"[WINDOW {window_count:04d}] frames {start_frame}-{frame_count}: "
                f"view={view_name} "
                f"view_score={view_score:.6f} "
                f"quality_score={quality_score:.6f} "
                f"transition={transition_name}"
            )
            if keyframe_detector is not None and transition_name == "no":
                if view_name in {"A2C", "A3C", "A4C", "A5C", "PLAX"}:
                    keyframe_detector.clear()
                    for roi_frame in roi_frame_window:
                        keyframe_detector.add_frame(roi_frame, view_name)
                    keyframe_result = keyframe_detector.infer_clip(view_name)
                    print(
                        f"[KEYFRAME {window_count:04d}] frames {start_frame}-{frame_count}: "
                        f"view={view_name} {summarize_keyframe_result(keyframe_result)}"
                    )
                    if show_keyframe_images:
                        show_keyframe_result_images(
                            roi_frame_window,
                            keyframe_result,
                            window_name="KeyFrame result",
                        )
                else:
                    print(f"[KEYFRAME {window_count:04d}] skipped unsupported view={view_name}")
            elif keyframe_detector is not None:
                print(f"[KEYFRAME {window_count:04d}] skipped transition={transition_name}")
            feature_window.clear()
            roi_frame_window.clear()

            if max_windows is not None and window_count >= max_windows:
                break

        if max_frames is not None and frame_count >= max_frames:
            break

    if color_segment_start is not None:
        print(f"[COLOR] skipped frames {color_segment_start}-{frame_count}")
    if invalid_segment_start is not None:
        print(f"[INVALID] skipped frames {invalid_segment_start}-{frame_count}")

    print(
        f"[DONE] video={video_path.name}, frames={frame_count}, "
        f"swinhead_windows={window_count}, skipped_color_frames={skipped_color_frames}, "
        f"skipped_invalid_frames={skipped_invalid_frames}, "
        f"skipped_bmode_color_frames={skipped_bmode_color_frames}, "
        f"skipped_spectrum_frames={skipped_spectrum_frames}, "
        f"skipped_no_roi_frames={skipped_no_roi_frames}"
    )
    if frame_count == 0:
        print(f"[WARN] no frames were read from video: {video_path}")
    elif window_count == 0:
        print(
            "[WARN] no classification window was produced; "
            "need at least 24 frames after ROI/B-mode filtering."
        )


def predict_prepared_inputs(
    prepared_input: Path,
    backbone: TrtRunner,
    swinhead: TrtRunner,
    max_frames: int | None,
    max_windows: int | None,
) -> None:
    """使用已经准备好的 backbone 输入运行切面分类。"""

    backbone_output_name = backbone.single_output_name
    feature_window: list[np.ndarray] = []
    frame_count = 0
    window_count = 0
    expected_shape = backbone.tensor_shapes[backbone.single_input_name]

    for prepared_frame in iter_prepared_inputs(prepared_input):
        frame_count += 1
        backbone_input = frame_to_backbone_input(prepared_frame, expected_shape)
        backbone_outputs = backbone.infer(backbone_input)
        feature_window.append(backbone_outputs[backbone_output_name][0])

        if len(feature_window) == WINDOW_SIZE:
            swinhead_input = np.ascontiguousarray(np.stack(feature_window, axis=0), dtype=np.float32)
            swinhead_outputs = swinhead.infer(swinhead_input)
            window_count += 1
            start_frame = frame_count - WINDOW_SIZE + 1
            print(
                f"[WINDOW {window_count:04d}] inputs {start_frame}-{frame_count}: "
                f"{summarize_swinhead_outputs(swinhead_outputs, VIEW_CLASSES, TRANSITION_CLASSES)}"
            )
            feature_window.clear()

            if max_windows is not None and window_count >= max_windows:
                break

        if max_frames is not None and frame_count >= max_frames:
            break

    print(f"[DONE] prepared_input={prepared_input}, inputs={frame_count}, swinhead_windows={window_count}")


def inspect_raw_video(
    video_path: Path,
    max_frames: int | None,
    skip_color_doppler: bool,
    color_ratio_threshold: float,
    valid_frame_threshold: float,
    skip_invalid_frames: bool,
) -> None:
    """不运行 TensorRT 推理，仅统计无效帧和彩色多普勒帧。"""

    frame_count = 0
    skipped_invalid_frames = 0
    skipped_color_frames = 0

    for frame in iter_video_frames(video_path):
        frame_count += 1
        if skip_invalid_frames and not is_valid_ultrasound_frame(frame, valid_frame_threshold):
            skipped_invalid_frames += 1
        elif skip_color_doppler and is_color_doppler_frame(frame, color_ratio_threshold):
            skipped_color_frames += 1

        if max_frames is not None and frame_count >= max_frames:
            break

    print(
        f"[DONE] raw_video={video_path.name}, frames={frame_count}, "
        f"invalid_frames={skipped_invalid_frames}, color_doppler_frames={skipped_color_frames}"
    )


class ViewClsInferenceThread(threading.Thread):
    """持有 TensorRT 上下文并执行完整推理流程的工作线程。"""

    def __init__(self, config: ViewClsInferenceConfig) -> None:
        super().__init__(name="ViewClsInferenceThread")
        self.config = config
        self.exception: BaseException | None = None

    def run(self) -> None:
        try:
            self._run_inference()
        except BaseException as exc:
            self.exception = exc

    def raise_if_failed(self) -> None:
        if self.exception is not None:
            raise self.exception

    def _run_inference(self) -> None:
        trt = load_tensorrt()
        cudart = CudaRuntime(self.config.gpu)
        stream = cudart.create_stream()

        roi_detector: TrtRunner | None = None
        backbone: TrtRunner | None = None
        swinhead: TrtRunner | None = None
        keyframe_detector = None
        try:
            free_bytes, total_bytes = cudart.mem_info()
            print(
                f"[INFO] CUDA device: {cudart.get_device()}, "
                f"free={free_bytes / 1024 / 1024:.0f} MiB, "
                f"total={total_bytes / 1024 / 1024:.0f} MiB"
            )
            roi_preprocessor: VideoRoiPreprocessor | None = None
            if self.config.enable_roi_preprocess:
                if self.config.roi_engine is None:
                    raise ValueError("ROI preprocessing is enabled but roi_engine is not configured")
                roi_detector = TrtRunner(self.config.roi_engine, trt, cudart, stream)
                roi_preprocessor = VideoRoiPreprocessor(
                    detector=roi_detector,
                    target_class_id=self.config.roi_class_id,
                    axis_class_id=self.config.axis_class_id,
                    conf_threshold=self.config.roi_conf_threshold,
                    color_ratio_threshold=self.config.color_ratio_threshold,
                    skip_no_roi_frames=self.config.skip_no_roi_frames,
                    b_mode_only=self.config.b_mode_only,
                )
                roi_input_name = roi_detector.single_input_name
                print(f"[INFO] ROI detector input: {roi_input_name} {roi_detector.tensor_shapes[roi_input_name]}")
                print(f"[INFO] ROI detector outputs: {roi_output_desc(roi_detector)}")
            backbone = TrtRunner(self.config.backbone_engine, trt, cudart, stream)
            swinhead = TrtRunner(self.config.swinhead_engine, trt, cudart, stream)
            if self.config.enable_keyframe_detection and self.config.prepared_input is None:
                from keyframe_detector import KeyframeDetector

                keyframe_detector = KeyframeDetector(trt, cudart, stream)
            print(f"[INFO] inference thread: {self.name}")
            print(f"[INFO] prepared input: {self.config.prepared_input}")
            print(f"[INFO] backbone input: {backbone.single_input_name} {backbone.tensor_shapes[backbone.single_input_name]}")
            print(f"[INFO] backbone output: {backbone_output_desc(backbone)}")
            print(f"[INFO] swinhead input: {swinhead.single_input_name} {swinhead.tensor_shapes[swinhead.single_input_name]}")
            print("[INFO] preprocessing: resize + RGB->BGR + /255 + mean/std + NCHW")
            print(f"[INFO] keyframe detection: {keyframe_detector is not None}")
            if self.config.prepared_input is not None:
                predict_prepared_inputs(
                    self.config.prepared_input,
                    backbone,
                    swinhead,
                    self.config.max_frames,
                    self.config.max_windows,
                )
            else:
                print(
                    f"[INFO] invalid frame skip: {self.config.skip_invalid_frames}, "
                    f"threshold={self.config.valid_frame_threshold}"
                )
                print(
                    f"[INFO] color Doppler skip: {self.config.skip_color_doppler}, "
                    f"threshold={self.config.color_ratio_threshold}"
                )
                predict_video(
                    self.config.video_path,
                    backbone,
                    swinhead,
                    self.config.max_frames,
                    self.config.max_windows,
                    self.config.skip_color_doppler,
                    self.config.color_ratio_threshold,
                    self.config.valid_frame_threshold,
                    self.config.skip_invalid_frames,
                    roi_preprocessor.process if roi_preprocessor is not None else None,
                    self.config.show_viewcls_input,
                    keyframe_detector,
                    self.config.show_keyframe_images,
                )
        finally:
            if keyframe_detector is not None:
                keyframe_detector.close()
            if swinhead is not None:
                swinhead.close()
            if backbone is not None:
                backbone.close()
            if roi_detector is not None:
                roi_detector.close()
            cudart.destroy_stream(stream)
