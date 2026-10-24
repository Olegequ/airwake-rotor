#!/usr/bin/env python
# Copyright (c) 2026 Olegequ.
# Project: airwake-rotor | Author: Olegequ
"""Independently validate all selected-mode RVMD deletion GIFs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

from .nato_gd_study import load_study_context
from .render_rvmd_deletion_flow_gifs import extract_orthogonal_slices, reconstruct_slice
from .rvmd_3d_flow import RVMD3DFlow


GIF_NAME = re.compile(
    r"^stage_(?P<stage>\d+)_delete_mode_(?P<deleted>\d+)_retain_(?P<retained>\d+)\.gif$"
)
CONTACT_FRAME_INDICES = (0, 99, 199)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_contract(path: Path, expected_mode_count: Optional[int] = None) -> Tuple[int, int, int]:
    match = GIF_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError("unexpected GIF filename: {}".format(path.name))
    stage = int(match.group("stage"))
    deleted = int(match.group("deleted"))
    retained = int(match.group("retained"))
    inferred_mode_count = retained + stage
    if (
        stage < 1
        or deleted != retained + 1
        or (expected_mode_count is not None and inferred_mode_count != expected_mode_count)
    ):
        raise ValueError("filename does not obey cumulative deletion contract: {}".format(path.name))
    return stage, deleted, retained


def validate_all_frames(
    path: Path, expected_frames: int = 200, expected_mode_count: Optional[int] = None
) -> Dict[str, Any]:
    stage, deleted, retained = _parse_contract(path, expected_mode_count)
    durations: List[int] = []
    sample_pixels: List[List[int]] = []
    with Image.open(str(path)) as animation:
        frame_count = int(getattr(animation, "n_frames", 1))
        if frame_count != expected_frames:
            raise ValueError(
                "{} contains {} frames; expected {}".format(path.name, frame_count, expected_frames)
            )
        width, height = (int(value) for value in animation.size)
        if (width, height) != (864, 496):
            raise ValueError("{} has unexpected dimensions {}x{}".format(path.name, width, height))
        for frame_index in range(frame_count):
            animation.seek(frame_index)
            frame = animation.convert("RGB")
            if frame.size != (width, height):
                raise ValueError("{} changes dimensions at frame {}".format(path.name, frame_index))
            sample_pixels.append(list(frame.getpixel((width // 2, height // 2))))
            durations.append(int(animation.info.get("duration", 0)))
    if min(durations) <= 0:
        raise ValueError("{} contains a non-positive frame duration".format(path.name))
    return {
        "stage_index": stage,
        "deleted_mode": deleted,
        "retained_mode_count": retained,
        "frame_count": frame_count,
        "pixel_width": width,
        "pixel_height": height,
        "duration_ms_min": int(min(durations)),
        "duration_ms_max": int(max(durations)),
        "decoded_frame_center_pixel_checksum": hashlib.sha256(
            np.asarray(sample_pixels, dtype=np.uint8).tobytes()
        ).hexdigest(),
        "file_size_bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def extract_contact_frames(
    paths_by_stage: Dict[int, Path], contact_stages: Sequence[int] = (1, 15, 30)
) -> List[Image.Image]:
    frames: List[Image.Image] = []
    for stage in contact_stages:
        path = paths_by_stage[stage]
        with Image.open(str(path)) as animation:
            for frame_index in CONTACT_FRAME_INDICES:
                animation.seek(frame_index)
                frames.append(animation.convert("RGB").copy())
    return frames


def write_contact_sheet(
    frames: Sequence[Image.Image],
    path: Path,
    contact_stages: Sequence[int] = (1, 15, 30),
) -> None:
    if len(frames) != len(contact_stages) * len(CONTACT_FRAME_INDICES):
        raise ValueError("contact-sheet frame count is inconsistent")
    thumbnail_size = (432, 248)
    gutter = 6
    sheet = Image.new(
        "RGB",
        (
            len(CONTACT_FRAME_INDICES) * thumbnail_size[0]
            + (len(CONTACT_FRAME_INDICES) - 1) * gutter,
            len(contact_stages) * thumbnail_size[1] + (len(contact_stages) - 1) * gutter,
        ),
        "white",
    )
    resampling = getattr(Image, "Resampling", Image)
    for index, frame in enumerate(frames):
        thumbnail = ImageOps.fit(frame, thumbnail_size, method=resampling.LANCZOS)
        row, column = divmod(index, len(CONTACT_FRAME_INDICES))
        sheet.paste(
            thumbnail,
            (column * (thumbnail_size[0] + gutter), row * (thumbnail_size[1] + gutter)),
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(path), format="PNG", optimize=True)


def mean_only_staticity(flow: RVMD3DFlow, hub_xyz_m: Sequence[float]) -> Dict[str, Any]:
    slices = extract_orthogonal_slices(flow, hub_xyz_m)
    source_indices = [0, flow.frame_count // 2, flow.frame_count - 1]
    slice_pairs = (
        (slices.xy_mean, slices.xy_modes),
        (slices.xz_mean, slices.xz_modes),
        (slices.yz_mean, slices.yz_modes),
    )
    reference_fields = []
    maximum_difference = 0.0
    for source_index in source_indices:
        coefficient = flow.coefficients[source_index] - flow.coefficient_mean
        fields = [reconstruct_slice(mean, modes, coefficient, 0) for mean, modes in slice_pairs]
        if not reference_fields:
            reference_fields = fields
            continue
        maximum_difference = max(
            maximum_difference,
            max(float(np.max(np.abs(field - reference))) for field, reference in zip(fields, reference_fields)),
        )
    return {
        "source_frame_indices": source_indices,
        "retained_mode_count": 0,
        "maximum_absolute_velocity_difference_mps": maximum_difference,
        "pass": bool(maximum_difference <= np.finfo(float).eps),
    }


def audit(config: str, allow_provisional: bool) -> Dict[str, Any]:
    context = load_study_context(config, allow_provisional=allow_provisional)
    mode_count = int(context.mode_count)
    gif_directory = context.output_dir / "07_flow_gifs_200frames"
    paths = sorted(gif_directory.glob("*.gif"))
    if len(paths) != mode_count:
        raise ValueError("expected {} GIF files, found {}".format(mode_count, len(paths)))
    paths_by_stage: Dict[int, Path] = {}
    outputs: List[Dict[str, Any]] = []
    for number, path in enumerate(paths, start=1):
        result = validate_all_frames(path, expected_mode_count=mode_count)
        stage = int(result["stage_index"])
        if stage in paths_by_stage:
            raise ValueError("duplicate stage {}".format(stage))
        paths_by_stage[stage] = path
        outputs.append(result)
        print("validated GIF {}/{}: {}".format(number, mode_count, path.name), flush=True)
    if set(paths_by_stage) != set(range(1, mode_count + 1)):
        raise ValueError("GIF stage set is incomplete")

    contact_stages = tuple(sorted(set((1, max(1, int(math.ceil(mode_count / 2.0))), mode_count))))
    contact_path = context.output_dir / "09_qa" / "gif_contact_sheet.png"
    write_contact_sheet(
        extract_contact_frames(paths_by_stage, contact_stages),
        contact_path,
        contact_stages,
    )
    flow = RVMD3DFlow(context.output_dir / "02_rvmd")
    try:
        staticity = mean_only_staticity(flow, context.hub_world_m)
    finally:
        flow.close()
    if not staticity["pass"]:
        raise ValueError("mean-only RVMD stage varies with time")

    result = {
        "schema_version": 1,
        "status": "pass",
        "independent_full_decode": True,
        "gif_count": len(outputs),
        "frames_per_gif": 200,
        "decoded_frame_count_total": int(sum(row["frame_count"] for row in outputs)),
        "common_dimensions_px": [864, 496],
        "stages": outputs,
        "mean_only_stage_staticity": staticity,
        "contact_sheet": str(contact_path),
        "contact_sheet_rows_are_stages": list(contact_stages),
        "contact_sheet_columns_are_gif_frame_indices": list(CONTACT_FRAME_INDICES),
    }
    output_path = context.output_dir / "09_qa" / "gif_validation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/nato_gd_rvmd_3d.json")
    parser.add_argument("--allow-provisional", action="store_true")
    args = parser.parse_args(argv)
    result = audit(args.config, args.allow_provisional)
    print(
        json.dumps(
            {
                "status": result["status"],
                "gif_count": result["gif_count"],
                "decoded_frame_count_total": result["decoded_frame_count_total"],
                "mean_only_stage_staticity": result["mean_only_stage_staticity"],
                "contact_sheet": result["contact_sheet"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["audit", "mean_only_staticity", "validate_all_frames", "write_contact_sheet"]
