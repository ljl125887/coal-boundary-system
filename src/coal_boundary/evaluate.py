from __future__ import annotations

import numpy as np

from .geometry import PileResult
from .reference import ReferencePile


def evaluate_results(
    results: list[PileResult],
    references: list[ReferencePile],
    true_total_volume_m3: float,
    target_total_volume_error_percent: float = 2.0,
) -> dict:
    if not results or not references:
        return {"matched_count": 0, "reference_count": len(references), "detected_count": len(results)}
    iou = np.zeros((len(results), len(references)), dtype=np.float64)
    for i, result in enumerate(results):
        for j, ref in enumerate(references):
            if not result.polygon.intersects(ref.polygon_projected):
                continue
            intersection = result.polygon.intersection(ref.polygon_projected).area
            union = result.polygon.union(ref.polygon_projected).area
            iou[i, j] = intersection / union if union else 0.0
    # Greedy maximum-IoU matching is deterministic and avoids a heavy linear algebra dependency.
    auto_idx, ref_idx = [], []
    available_auto, available_ref = set(range(len(results))), set(range(len(references)))
    while available_auto and available_ref:
        best = max(((iou[i, j], i, j) for i in available_auto for j in available_ref), default=(0, -1, -1))
        if best[0] <= 0:
            break
        _, i, j = best
        auto_idx.append(i); ref_idx.append(j)
        available_auto.remove(i); available_ref.remove(j)
    matches = []
    for i, j in zip(auto_idx, ref_idx):
        if iou[i, j] <= 0:
            continue
        matches.append({
            "detected_id": results[i].pile_id, "reference_id": references[j].pile_id,
            "iou": float(iou[i, j]),
            "boundary_hausdorff_m": float(results[i].polygon.boundary.hausdorff_distance(references[j].polygon_projected.boundary)),
            "calculated_volume_m3": results[i].volume_m3,
            "reference_volume_m3": references[j].true_volume_m3,
        })
    calculated_total = float(sum(r.volume_m3 for r in results))
    volume_error_pct = abs(calculated_total - true_total_volume_m3) / true_total_volume_m3 * 100.0
    return {
        "reference_count": len(references), "detected_count": len(results), "matched_count": len(matches),
        "mean_iou": float(np.mean([m["iou"] for m in matches])) if matches else 0.0,
        "median_iou": float(np.median([m["iou"] for m in matches])) if matches else 0.0,
        "calculated_total_volume_m3": calculated_total,
        "true_total_volume_m3": float(true_total_volume_m3),
        "total_volume_error_percent": float(volume_error_pct),
        "target_total_volume_error_percent": float(target_total_volume_error_percent),
        "volume_acceptance_passed": bool(volume_error_pct <= target_total_volume_error_percent),
        "matches": matches,
    }
