"""ROI 预处理、切面分类和关键帧检测的命令行入口。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from predict_view_cls import (
    BACKBONE_ENGINE,
    DEFAULT_COLOR_RATIO_THRESHOLD,
    DEFAULT_VALID_FRAME_THRESHOLD,
    PROJECT_ROOT,
    ROI_ENGINE,
    SWINHEAD_ENGINE,
    ViewClsInferenceConfig,
    ViewClsInferenceThread,
    inspect_raw_video,
)
from video_roi_preprocessor import DEFAULT_AXIS_CLASS_ID, DEFAULT_ROI_CLASS_ID, DEFAULT_ROI_CONF_THRESHOLD


TEST_VIDEO_DIR = PROJECT_ROOT / "test-videos"
# 可切换为完整测试视频。
# RUN_VIDEO = TEST_VIDEO_DIR / "CHEN_FUZHONG.mp4"
RUN_VIDEO = TEST_VIDEO_DIR / "CHEN_FUZHONG - Trim.mp4"


def resolve_video(path_or_name: str) -> Path:
    """解析绝对视频路径，或 test-videos 目录下的视频文件名。"""
    path = Path(path_or_name)
    if path.is_file():
        return path.resolve()

    candidate = TEST_VIDEO_DIR / path_or_name
    if candidate.is_file():
        return candidate.resolve()

    raise FileNotFoundError(f"Video not found: {path_or_name}")


def parse_args() -> argparse.Namespace:
    """解析测试视频、ROI 预处理、切面分类和关键帧检测相关参数。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default=str(RUN_VIDEO),
        help="Raw video path or file name under test-videos. Used only for raw-video inspection mode.",
    )
    parser.add_argument(
        "--prepared-input",
        type=Path,
        default=None,
        help=(
            "Path to a .npy file or directory of .npy files containing already prepared "
            "backbone inputs. Supported shapes: 1x3x320x320, 3x320x320, 320x320x3, "
            "or batched variants."
        ),
    )
    parser.add_argument(
        "--raw-video-inspect",
        action="store_true",
        help="Inspect raw video for invalid/color frames without sending frames to TensorRT.",
    )
    parser.add_argument("--backbone-engine", default=BACKBONE_ENGINE, type=Path)
    parser.add_argument("--swinhead-engine", default=SWINHEAD_ENGINE, type=Path)
    parser.add_argument("--roi-engine", default=ROI_ENGINE, type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index. Default: 0")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument(
        "--disable-roi-preprocess",
        action="store_true",
        help="Disable ROI detector preprocessing and classify the full frame.",
    )
    parser.add_argument(
        "--roi-conf-threshold",
        type=float,
        default=DEFAULT_ROI_CONF_THRESHOLD,
        help="Minimum confidence for the ROI detector box. Default: 0.25",
    )
    parser.add_argument(
        "--roi-class-id",
        type=int,
        default=DEFAULT_ROI_CLASS_ID,
        help="Detector class id used as the ultrasound ROI. The axis class is ignored. Default: 0",
    )
    parser.add_argument(
        "--axis-class-id",
        type=int,
        default=DEFAULT_AXIS_CLASS_ID,
        help="Detector class id used as the spectrum axis. Frames with this class are skipped in B-mode-only mode. Default: 1",
    )
    parser.add_argument(
        "--skip-no-roi-frames",
        action="store_true",
        default=True,
        help="Skip frames when the ROI detector finds no ROI. Enabled by default for B-mode classification.",
    )
    parser.add_argument(
        "--allow-no-roi-fallback",
        action="store_true",
        help="Use the full frame when no ROI is found.",
    )
    parser.add_argument(
        "--disable-bmode-only",
        action="store_true",
        help="Disable B-mode filtering. Color Doppler and spectrum frames will not be filtered by the ROI preprocessor.",
    )
    parser.add_argument(
        "--hide-viewcls-input",
        action="store_true",
        help="Do not show the exact 320x320 BGR frame sent to the view classification backbone.",
    )
    parser.add_argument(
        "--disable-keyframe-detection",
        action="store_true",
        help="Disable keyframe detection after supported non-transition view-classification windows.",
    )
    parser.add_argument(
        "--hide-keyframe-images",
        action="store_true",
        help="Do not show detected ED/ES keyframe images.",
    )
    parser.add_argument(
        "--color-ratio-threshold",
        type=float,
        default=DEFAULT_COLOR_RATIO_THRESHOLD,
        help=(
            "Frame is treated as color Doppler when red/blue high-saturation pixels "
            "in the center ROI exceed this ratio. Default: 0.005"
        ),
    )
    parser.add_argument(
        "--valid-frame-threshold",
        type=float,
        default=DEFAULT_VALID_FRAME_THRESHOLD,
        help=(
            "Frame is skipped as loading/blank when center ROI nonblack pixel ratio "
            "is below this value. Only used with --enable-valid-frame-skip. Default: 0.02"
        ),
    )
    parser.add_argument(
        "--enable-valid-frame-skip",
        action="store_true",
        help="Skip loading/blank frames. Disabled by default to match the C++ deployment path.",
    )
    parser.add_argument(
        "--enable-color-skip",
        action="store_true",
        help="Skip color Doppler frames. Disabled by default to match the C++ deployment path.",
    )
    parser.add_argument(
        "--disable-color-skip",
        action="store_true",
        help="Deprecated compatibility flag. Color Doppler skipping is already disabled by default.",
    )
    return parser.parse_args()


def main() -> int:
    """构建推理配置，启动工作线程，并抛出线程中的异常。"""
    args = parse_args()
    skip_color_doppler = args.enable_color_skip and not args.disable_color_skip
    skip_invalid_frames = args.enable_valid_frame_skip
    video_path = resolve_video(args.video) if args.raw_video_inspect or args.prepared_input is None else Path(args.video)
    enable_roi_preprocess = args.prepared_input is None and not args.disable_roi_preprocess
    if args.prepared_input is None:
        print(f"[INFO] video input: {video_path}")
    else:
        print(f"[INFO] prepared input: {args.prepared_input}")
    if args.raw_video_inspect:
        inspect_raw_video(
            video_path,
            args.max_frames,
            skip_color_doppler,
            args.color_ratio_threshold,
            args.valid_frame_threshold,
            skip_invalid_frames,
        )
        return 0

    config = ViewClsInferenceConfig(
        video_path=video_path,
        prepared_input=args.prepared_input,
        roi_engine=args.roi_engine,
        backbone_engine=args.backbone_engine,
        swinhead_engine=args.swinhead_engine,
        gpu=args.gpu,
        max_frames=args.max_frames,
        max_windows=args.max_windows,
        skip_color_doppler=skip_color_doppler,
        color_ratio_threshold=args.color_ratio_threshold,
        valid_frame_threshold=args.valid_frame_threshold,
        skip_invalid_frames=skip_invalid_frames,
        enable_roi_preprocess=enable_roi_preprocess,
        roi_conf_threshold=args.roi_conf_threshold,
        roi_class_id=args.roi_class_id,
        axis_class_id=args.axis_class_id,
        skip_no_roi_frames=args.skip_no_roi_frames and not args.allow_no_roi_fallback,
        b_mode_only=not args.disable_bmode_only,
        show_viewcls_input=not args.hide_viewcls_input,
        enable_keyframe_detection=not args.disable_keyframe_detection,
        show_keyframe_images=not args.hide_keyframe_images,
    )
    infer_thread = ViewClsInferenceThread(config)
    infer_thread.start()
    infer_thread.join()
    infer_thread.raise_if_failed()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        no_pause = (
            os.environ.get("PREDICT_VIEW_CLS_NO_PAUSE") == "1"
            or os.environ.get("PREDICT_VIDEOS_NO_PAUSE") == "1"
        )
        if getattr(sys, "frozen", False) and not no_pause:
            input("Press Enter to exit...")
        raise
