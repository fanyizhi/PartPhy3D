from __future__ import annotations

import gzip
import importlib.util
import io
import json
import tarfile
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "catalog_dimensions", ROOT / "scripts/prepare_catalog_dimensions.py"
    )
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_catalog_scale_input_excludes_mass_and_package_dimensions(
    tmp_path: Path,
) -> None:
    record = {
        "item_id": "TEST",
        "domain_name": "amazon.com",
        "item_weight": [{"value": 10000, "unit": "pounds"}],
        "item_dimensions": {
            axis: {"unit": "inches", "value": length}
            for axis, length in zip(
                ("length", "width", "height"), (10, 5, 1), strict=True
            )
        },
        "package_dimensions": {"length": {"value": 100, "unit": "inches"}},
    }
    archive = tmp_path / "listings.tar"
    data = gzip.compress((json.dumps(record) + "\n").encode())
    with tarfile.open(archive, "w") as bundle:
        info = tarfile.TarInfo("listings/metadata/listings_0.json.gz")
        info.size = len(data)
        bundle.addfile(info, io.BytesIO(data))
    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"object_ids": ["TEST"]}))
    output = tmp_path / "dimensions.json"
    module().prepare(archive, cohort, output)
    result = json.loads(output.read_text())
    entry = result["objects"]["TEST"]
    assert entry["status"] == "available"
    assert entry["max_dimension_m"] == pytest.approx(0.254)
    assert all("item_weight" not in candidate for candidate in entry["candidates"])
    assert all(
        "package_dimensions" not in candidate for candidate in entry["candidates"]
    )
    assert result["item_weight_passed_to_inference"] is False
    assert result["mass_labels_included"] is False


def test_missing_or_unknown_units_are_not_treated_as_meters() -> None:
    assert (
        module().metric_dimensions(
            {
                "item_dimensions": {
                    axis: {"value": 1, "unit": "unknown"}
                    for axis in ("length", "width", "height")
                }
            }
        )
        is None
    )
