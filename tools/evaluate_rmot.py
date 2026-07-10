#!/usr/bin/env python3
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "TrackEval"))

from trackeval.metrics import CLEAR, HOTA  # noqa: E402


def parse_mot_file(path):
    frames = defaultdict(list)
    if not path.exists() or path.stat().st_size == 0:
        return frames

    with path.open(newline="") as handle:
        for line_number, row in enumerate(csv.reader(handle), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 6:
                raise ValueError(f"{path}:{line_number}: expected at least 6 columns")
            frame_id = int(float(row[0]))
            track_id = int(float(row[1]))
            if track_id < 0:
                continue
            x, y, width, height = (float(value) for value in row[2:6])
            if width <= 0 or height <= 0:
                continue
            frames[frame_id].append((track_id, x, y, width, height))
    return frames


def box_iou_xywh(gt_boxes, tracker_boxes):
    if len(gt_boxes) == 0 or len(tracker_boxes) == 0:
        return np.zeros((len(gt_boxes), len(tracker_boxes)), dtype=np.float64)

    gt = np.asarray(gt_boxes, dtype=np.float64)
    tracker = np.asarray(tracker_boxes, dtype=np.float64)
    gt_xyxy = np.column_stack((gt[:, :2], gt[:, :2] + gt[:, 2:]))
    tracker_xyxy = np.column_stack(
        (tracker[:, :2], tracker[:, :2] + tracker[:, 2:])
    )
    intersection_min = np.maximum(gt_xyxy[:, None, :2], tracker_xyxy[None, :, :2])
    intersection_max = np.minimum(gt_xyxy[:, None, 2:], tracker_xyxy[None, :, 2:])
    intersection_wh = np.maximum(0.0, intersection_max - intersection_min)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    gt_area = gt[:, 2] * gt[:, 3]
    tracker_area = tracker[:, 2] * tracker[:, 3]
    union = gt_area[:, None] + tracker_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > np.finfo(np.float64).eps,
    )


def contiguous_id_map(frames):
    ids = sorted({row[0] for rows in frames.values() for row in rows})
    return {track_id: index for index, track_id in enumerate(ids)}


def build_sequence_data(gt_path, prediction_path):
    gt_frames = parse_mot_file(gt_path)
    tracker_frames = parse_mot_file(prediction_path)
    gt_id_map = contiguous_id_map(gt_frames)
    tracker_id_map = contiguous_id_map(tracker_frames)
    max_frame = max([0, *gt_frames.keys(), *tracker_frames.keys()])

    gt_ids = []
    tracker_ids = []
    similarities = []
    num_gt_dets = 0
    num_tracker_dets = 0
    for frame_id in range(1, max_frame + 1):
        gt_rows = gt_frames.get(frame_id, [])
        tracker_rows = tracker_frames.get(frame_id, [])
        gt_ids_t = np.asarray(
            [gt_id_map[row[0]] for row in gt_rows], dtype=np.int64
        )
        tracker_ids_t = np.asarray(
            [tracker_id_map[row[0]] for row in tracker_rows], dtype=np.int64
        )
        if len(np.unique(gt_ids_t)) != len(gt_ids_t):
            raise ValueError(f"{gt_path}: duplicate GT id in frame {frame_id}")
        if len(np.unique(tracker_ids_t)) != len(tracker_ids_t):
            raise ValueError(
                f"{prediction_path}: duplicate tracker id in frame {frame_id}"
            )
        gt_boxes = [row[1:] for row in gt_rows]
        tracker_boxes = [row[1:] for row in tracker_rows]
        gt_ids.append(gt_ids_t)
        tracker_ids.append(tracker_ids_t)
        similarities.append(box_iou_xywh(gt_boxes, tracker_boxes))
        num_gt_dets += len(gt_rows)
        num_tracker_dets += len(tracker_rows)

    return {
        "num_timesteps": max_frame,
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(tracker_id_map),
        "num_gt_dets": num_gt_dets,
        "num_tracker_dets": num_tracker_dets,
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
    }


def metric_summary(hota_result, clear_result):
    return {
        "HOTA": float(np.mean(hota_result["HOTA"]) * 100.0),
        "DetA": float(np.mean(hota_result["DetA"]) * 100.0),
        "AssA": float(np.mean(hota_result["AssA"]) * 100.0),
        "DetRe": float(np.mean(hota_result["DetRe"]) * 100.0),
        "DetPr": float(np.mean(hota_result["DetPr"]) * 100.0),
        "AssRe": float(np.mean(hota_result["AssRe"]) * 100.0),
        "AssPr": float(np.mean(hota_result["AssPr"]) * 100.0),
        "LocA": float(np.mean(hota_result["LocA"]) * 100.0),
        "IDSW": int(clear_result["IDSW"]),
    }


def evaluate(results_root, gt_name="gt.txt", prediction_name="predict.txt"):
    results_root = Path(results_root)
    gt_paths = sorted(results_root.rglob(gt_name))
    if not gt_paths:
        raise FileNotFoundError(f"no {gt_name} files found under {results_root}")

    hota_metric = HOTA()
    clear_metric = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False})
    hota_results = {}
    clear_results = {}
    per_sequence = {}
    for gt_path in gt_paths:
        sequence = str(gt_path.parent.relative_to(results_root))
        prediction_path = gt_path.with_name(prediction_name)
        data = build_sequence_data(gt_path, prediction_path)
        hota_result = hota_metric.eval_sequence(data)
        clear_result = clear_metric.eval_sequence(data)
        hota_results[sequence] = hota_result
        clear_results[sequence] = clear_result
        per_sequence[sequence] = metric_summary(hota_result, clear_result)

    combined_hota = hota_metric.combine_sequences(hota_results)
    combined_clear = clear_metric.combine_sequences(clear_results)
    summary = metric_summary(combined_hota, combined_clear)
    summary["num_sequences"] = len(gt_paths)
    summary["per_sequence"] = per_sequence
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Refer-KITTI expression tracks with HOTA and CLEAR."
    )
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--gt-name", default="gt.txt")
    parser.add_argument("--prediction-name", default="predict.txt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summary = evaluate(args.results_root, args.gt_name, args.prediction_name)
    compact = {key: value for key, value in summary.items() if key != "per_sequence"}
    print(json.dumps(compact, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
