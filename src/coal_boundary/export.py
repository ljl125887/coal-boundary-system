from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os

from shapely.geometry import mapping

from .crs import projected_to_lonlat
from .geometry import PileResult


def _polygon_to_lonlat_mapping(polygon, central_meridian, false_easting, false_northing):
    def ring(coords):
        xs, ys = zip(*coords)
        lon, lat = projected_to_lonlat(xs, ys, central_meridian, false_easting, false_northing)
        return [[float(a), float(b)] for a, b in zip(lon, lat)]
    return {
        "type": "Polygon",
        "coordinates": [ring(polygon.exterior.coords)] + [ring(interior.coords) for interior in polygon.interiors],
    }


def make_result_documents(results: list[PileResult], metadata: dict, crs_config, evaluation: dict | None = None):
    piles = []
    features = []
    for result in results:
        geometry = _polygon_to_lonlat_mapping(
            result.polygon, crs_config.central_meridian_deg,
            crs_config.false_easting_m, crs_config.false_northing_m,
        )
        properties = {
            "pile_id": result.pile_id, "area_m2": result.area_m2,
            "volume_m3": result.volume_m3, "max_relative_height_m": result.max_relative_height_m,
            "mean_relative_height_m": result.mean_relative_height_m,
            "quality_score": result.quality_score, "quality_grade": result.quality_grade,
            "boundary_confidence": result.boundary_confidence,
            "pixel_count": result.pixel_count,
        }
        pile = {**properties, "geometry": geometry}
        feature = {"type": "Feature", "id": result.pile_id, "properties": properties, "geometry": geometry}
        if result.display_polygon is not None:
            display_geometry = _polygon_to_lonlat_mapping(
                result.display_polygon, crs_config.central_meridian_deg,
                crs_config.false_easting_m, crs_config.false_northing_m,
            )
            pile["display_geometry"] = display_geometry
            feature["display_geometry"] = display_geometry
            if result.display_geometry_metrics is not None:
                pile["display_geometry_metrics"] = result.display_geometry_metrics
        piles.append(pile)
        features.append(feature)
    created = datetime.now(timezone.utc).isoformat()
    result_doc = {
        "schema_version": "1.0", "created_at": created,
        "metadata": metadata, "crs": {"source_epsg": crs_config.source_epsg, "output_epsg": crs_config.output_epsg},
        "summary": {
            "pile_count": len(results), "total_area_m2": float(sum(r.area_m2 for r in results)),
            "total_volume_m3": float(sum(r.volume_m3 for r in results)),
            "quality_grades": {g: sum(r.quality_grade == g for r in results) for g in ("A", "B", "C")},
        },
        "evaluation": evaluation, "piles": piles,
    }
    geojson = {
        "type": "FeatureCollection", "name": metadata.get("project_name", "coal_piles"),
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "features": features,
    }
    return result_doc, geojson


def make_service_fence_document(results: list[PileResult], crs_config) -> list[dict]:
    """Build the legacy fence array consumed by the Volume3D Cesium service."""
    fences = []
    for result in results:
        geometry = _polygon_to_lonlat_mapping(
            result.polygon, crs_config.central_meridian_deg,
            crs_config.false_easting_m, crs_config.false_northing_m,
        )
        exterior = geometry["coordinates"][0]
        if len(exterior) > 1 and exterior[0] == exterior[-1]:
            exterior = exterior[:-1]
        points = [{"lon": lon, "lat": lat} for lon, lat in exterior]
        longitudes = [point["lon"] for point in points]
        latitudes = [point["lat"] for point in points]
        fences.append({
            "num": result.pile_id,
            "volume_m3": result.volume_m3,
            "maxlon": max(longitudes),
            "minlon": min(longitudes),
            "maxlat": max(latitudes),
            "minlat": min(latitudes),
            "points": points,
        })
        if result.display_polygon is not None:
            display_geometry = _polygon_to_lonlat_mapping(
                result.display_polygon, crs_config.central_meridian_deg,
                crs_config.false_easting_m, crs_config.false_northing_m,
            )
            display_exterior = display_geometry["coordinates"][0]
            if len(display_exterior) > 1 and display_exterior[0] == display_exterior[-1]:
                display_exterior = display_exterior[:-1]
            fences[-1]["display_points"] = [
                {"lon": lon, "lat": lat} for lon, lat in display_exterior
            ]
    return fences


def atomic_write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
