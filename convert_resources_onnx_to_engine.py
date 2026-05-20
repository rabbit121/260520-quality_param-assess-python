#!/usr/bin/env python3
"""Batch convert ONNX models under resources/onnx to TensorRT engine files."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert every .onnx file under resources/onnx to resources/engine "
            "with TensorRT Python API."
        )
    )
    parser.add_argument(
        "--resources",
        default="resources/onnx",
        type=Path,
        help="Directory to scan recursively for .onnx files. Default: resources/onnx",
    )
    parser.add_argument(
        "--engine-dir",
        default="resources/engine",
        type=Path,
        help="Directory to write .engine files. Default: resources/engine",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild engines even when the .engine file already exists.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Build FP16 engines when the platform supports fast FP16.",
    )
    parser.add_argument(
        "--workspace-mib",
        default=4096,
        type=int,
        help="TensorRT workspace memory limit in MiB. Default: 4096",
    )
    parser.add_argument(
        "--tensorrt-root",
        default=os.environ.get("TENSORRT_ROOT"),
        type=Path,
        help=(
            "TensorRT install directory. Defaults to TENSORRT_ROOT env var; "
            "on Windows, the script also searches NVIDIA GPU Computing Toolkit."
        ),
    )
    parser.add_argument(
        "--shape",
        action="append",
        default=[],
        metavar="INPUT:DIMS",
        help=(
            "Optimization shape for dynamic inputs, e.g. "
            "--shape input:1x3x224x224. Repeat for multiple inputs. "
            "The same shape is used as min/opt/max."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without building engines.",
    )
    return parser.parse_args()


def find_tensorrt_root(cli_root: Path | None) -> Path | None:
    candidates: list[Path] = []
    if cli_root is not None:
        candidates.append(cli_root)

    env_root = os.environ.get("TENSORRT_ROOT")
    if env_root:
        candidates.append(Path(env_root))

    if sys.platform == "win32":
        base_dirs = [
            Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit"),
            Path(r"C:\Program Files\NVIDIA Corporation"),
        ]
        for base_dir in base_dirs:
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


def import_tensorrt(tensorrt_root: Path | None):
    add_tensorrt_dll_dirs(tensorrt_root)
    if tensorrt_root is not None:
        print(f"[INFO] TensorRT root: {tensorrt_root}")

    try:
        import tensorrt as trt
    except (ImportError, FileNotFoundError) as exc:
        print(
            "[ERROR] failed to import TensorRT Python package.\n"
            f"        Python: {sys.executable}\n"
            f"        TensorRT root: {tensorrt_root or 'not found'}\n"
            "        Make sure this interpreter has the matching tensorrt wheel installed,\n"
            "        and TensorRT lib/bin contains nvinfer_10.dll.\n"
            f"        Original error: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    return trt


def parse_dims(value: str) -> tuple[int, ...]:
    separators = ("x", ",")
    for separator in separators[1:]:
        value = value.replace(separator, separators[0])
    dims = tuple(int(part) for part in value.split(separators[0]) if part)
    if not dims:
        raise ValueError("empty dims")
    return dims


def parse_shapes(shape_specs: list[str]) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {}
    for spec in shape_specs:
        if ":" not in spec:
            raise ValueError(f"shape must be INPUT:DIMS, got: {spec}")
        name, dims = spec.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"missing input name in shape: {spec}")
        shapes[name] = parse_dims(dims)
    return shapes


def set_workspace_limit(trt, config, workspace_mib: int) -> None:
    workspace_bytes = workspace_mib * 1024 * 1024
    if hasattr(config, "set_memory_pool_limit") and hasattr(trt, "MemoryPoolType"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
    elif hasattr(config, "max_workspace_size"):
        config.max_workspace_size = workspace_bytes


def parse_onnx(trt, logger, network, onnx_path: Path) -> bool:
    parser = trt.OnnxParser(network, logger)
    if hasattr(parser, "parse_from_file"):
        ok = parser.parse_from_file(str(onnx_path))
    else:
        ok = parser.parse(onnx_path.read_bytes())

    if ok:
        return True

    print(f"[ERROR] failed to parse ONNX: {onnx_path}", file=sys.stderr)
    for index in range(parser.num_errors):
        print(f"  {parser.get_error(index)}", file=sys.stderr)
    return False


def has_dynamic_shape(shape: tuple[int, ...]) -> bool:
    return any(dim < 0 for dim in shape)


def default_dynamic_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(1 if dim < 0 else int(dim) for dim in shape)


def add_optimization_profile(builder, config, network, shapes: dict[str, tuple[int, ...]]) -> None:
    dynamic_inputs = []
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        tensor_shape = tuple(tensor.shape)
        if has_dynamic_shape(tensor_shape):
            dynamic_inputs.append((tensor.name, tensor_shape))

    if not dynamic_inputs:
        return

    profile = builder.create_optimization_profile()
    for name, tensor_shape in dynamic_inputs:
        shape = shapes.get(name)
        if shape is None:
            shape = default_dynamic_shape(tensor_shape)
            print(
                f"[WARN] dynamic input {name} has no --shape; using {shape} as min/opt/max"
            )

        if len(shape) != len(tensor_shape):
            raise ValueError(
                f"shape rank mismatch for {name}: ONNX rank {len(tensor_shape)}, "
                f"provided rank {len(shape)}"
            )

        profile.set_shape(name, shape, shape, shape)
        print(f"[PROFILE] {name}: min/opt/max = {shape}")

    config.add_optimization_profile(profile)


def build_engine(
    trt,
    logger,
    onnx_path: Path,
    engine_path: Path,
    fp16: bool,
    workspace_mib: int,
    shapes: dict[str, tuple[int, ...]],
) -> bool:
    builder = trt.Builder(logger)
    if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    else:
        flags = 0
    network = builder.create_network(flags)
    config = builder.create_builder_config()

    if not parse_onnx(trt, logger, network, onnx_path):
        return False

    set_workspace_limit(trt, config, workspace_mib)

    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            print("[WARN] platform does not report fast FP16 support; building FP32")

    add_optimization_profile(builder, config, network, shapes)

    serialized_engine = None
    if hasattr(builder, "build_serialized_network"):
        serialized_engine = builder.build_serialized_network(network, config)
    else:
        engine = builder.build_engine(network, config)
        if engine is not None:
            serialized_engine = engine.serialize()

    if serialized_engine is None:
        print(f"[ERROR] TensorRT failed to build engine: {onnx_path}", file=sys.stderr)
        return False

    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized_engine))
    return True


def engine_path_for(onnx_path: Path, onnx_root: Path, engine_root: Path) -> Path:
    relative_path = onnx_path.relative_to(onnx_root)
    return (engine_root / relative_path).with_suffix(".engine")


def main() -> int:
    args = parse_args()
    resources_dir = args.resources.resolve()
    engine_dir = args.engine_dir.resolve()

    if not resources_dir.is_dir():
        print(f"[ERROR] resources directory does not exist: {resources_dir}", file=sys.stderr)
        return 2

    onnx_files = sorted(resources_dir.rglob("*.onnx"))
    if not onnx_files:
        print(f"[WARN] no .onnx files found under: {resources_dir}")
        return 0

    shapes = parse_shapes(args.shape)

    for onnx_path in onnx_files:
        print(f"[PLAN] {onnx_path} -> {engine_path_for(onnx_path, resources_dir, engine_dir)}")

    if args.dry_run:
        return 0

    tensorrt_root = find_tensorrt_root(args.tensorrt_root)
    trt = import_tensorrt(tensorrt_root)

    logger = trt.Logger(trt.Logger.INFO)
    failed: list[Path] = []

    for onnx_path in onnx_files:
        engine_path = engine_path_for(onnx_path, resources_dir, engine_dir)
        if engine_path.exists() and not args.force:
            print(f"[SKIP] {engine_path} already exists")
            continue

        print(f"[BUILD] {onnx_path}")
        ok = build_engine(
            trt=trt,
            logger=logger,
            onnx_path=onnx_path,
            engine_path=engine_path,
            fp16=args.fp16,
            workspace_mib=args.workspace_mib,
            shapes=shapes,
        )
        if ok:
            print(f"[OK] {engine_path}")
        else:
            failed.append(onnx_path)

    if failed:
        print("\n[ERROR] failed conversions:", file=sys.stderr)
        for path in failed:
            print(f"  - {path}", file=sys.stderr)
        return 1

    print("\n[DONE] all ONNX conversions finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
