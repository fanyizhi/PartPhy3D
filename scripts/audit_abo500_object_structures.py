"""Create evidence and follow-up records for every ABO-500 structure estimate."""

import json
import math
from collections import Counter

import numpy as np

from evaluate_structure_cues import ROOT, WORK, errors, read, sealed


def main():
    geometry = read(WORK / "geometry_predictions.json")
    old = read(WORK / "bounded_components.json")["predictions"]
    new = read(WORK / "balanced_components.json")
    scale = read(WORK / "scale_geometry_study.json")["scale_inputs"]
    weights = read(ROOT / "weights.json")
    panels = read(WORK / "component_evidence.json")["scenes"]
    for directory in [*ROOT.glob("train_???/panels"), *ROOT.glob("val_???/panels")]:
        for panel in sealed(directory):
            key = panel["scene_id"]
            if (
                key in geometry
                and geometry[key]["split"] == directory.parent.name.split("_")[0]
            ):
                panels[key] = panel

    records = {}
    for key, geo in geometry.items():
        target = float(weights.get(key, weights.get(key.split("_")[0])))
        previous = old[key]
        current = new["predictions"][key]
        assert target > 0 and geo["split"] == current["split"] == previous["split"]
        assigned = current["scored_parts"]
        estimate = current["mass_kg"]
        assert math.isclose(sum(p["mass_kg"] for p in assigned), estimate)
        item = scale[key]
        flags = []
        source = geo["recovered_panel"] if geo["recovered"] else panels.get(key, {})
        source_parts = source.get("parts", [])
        localized = geo["component_evidence"]
        if not localized:
            flags.append("component_evidence_missing")
            source_parts = []
        if item["status"] == "missing":
            flags.append("scale_evidence_missing")
        elif abs(math.log(item["ratio"])) >= math.log(1.5):
            flags.append("catalog_length_differs_by_factor_1_5")
        if target < 1 and estimate > target * 2:
            flags.append("light_object_overestimated_by_factor_2")
        largest = max(assigned, key=lambda p: p["mass_kg"])
        if localized and largest["mass_kg"] / estimate >= 0.7:
            flags.append("single_part_has_70_percent_of_mass")
        if localized and any(
            p["construction"] == "shell"
            and p["outer_volume_m3"] > 0
            and p["mass_kg"] / p["density_kg_m3"] >= 0.95 * p["outer_volume_m3"]
            for p in source_parts
        ):
            flags.append("shell_occupies_at_least_95_percent_of_allocated_volume")
        for entry in assigned:
            if math.isclose(
                entry["mass_kg"] / entry["raw_kg"],
                new["maximum_multiplier"],
                rel_tol=1e-6,
            ):
                flags.append("part_hits_upper_calibration_limit")
                break
        names = [p["name"].lower().replace("_", " ") for p in source_parts]
        if any("drawer front" in n for n in names) and not any(
            "drawer" in n and any(word in n for word in ("body", "bottom", "box"))
            for n in names
        ):
            flags.append("drawer_fronts_without_explicit_internal_drawer_volume")
        if any(n == "mattress" for n in names) and any("slat" in n for n in names):
            flags.append("mattress_and_frame_share_object_envelope")
        if not flags:
            flags.append("no_threshold_triggered")
        records[key] = dict(
            split=geo["split"],
            target_kg=target,
            raw_kg=geo["raw"],
            previous_kg=previous["mass_kg"],
            estimate_kg=estimate,
            error_kg=estimate - target,
            previous_error_kg=previous["mass_kg"] - target,
            catalog_to_predicted_length=item["ratio"],
            localized=localized,
            largest_part=largest if localized else None,
            assigned_parts=assigned if localized else [],
            geometry_parts=[
                {
                    f: p.get(f)
                    for f in (
                        "name",
                        "material",
                        "construction",
                        "area_share",
                        "outer_volume_m3",
                        "thickness_mm",
                        "density_kg_m3",
                        "mass_kg",
                    )
                }
                for p in source_parts
            ],
            flags=flags,
        )

    labels = {
        "component_evidence_missing": "Recover the mesh and component record before attributing part errors.",
        "scale_evidence_missing": "Recover the original length estimate and compare it with a known scale reference.",
        "catalog_length_differs_by_factor_1_5": "Check units and match all three dimensions to the observed mesh.",
        "light_object_overestimated_by_factor_2": "Check for unsupported solid volumes and use a section or a thin-layer calculation where geometrically supported.",
        "single_part_has_70_percent_of_mass": "Inspect the dominant part's mask, volume, and material. Concentration may be correct.",
        "shell_occupies_at_least_95_percent_of_allocated_volume": "Check shell thickness and the allocated cross-section. A filled shell may also reflect an undersized envelope.",
        "part_hits_upper_calibration_limit": "Check which geometry or material assumption requires the calibration increase.",
        "drawer_fronts_without_explicit_internal_drawer_volume": "Check if sides, bottoms and runners are already included in another part. Model them individually if missing.",
        "mattress_and_frame_share_object_envelope": "Check if rail and bedding volumes use physically disjoint geometry, and whether internal mattress voids are already represented.",
        "no_threshold_triggered": "Compare part geometry and material with independent measurements if error remains.",
    }
    overview = {}
    for split in ("train", "val", "test", "all"):
        available = {k: r for k, r in records.items() if split in {"all", r["split"]}}
        groups = {}
        for name, predicate in (
            ("under_1kg", lambda y: y < 1),
            ("1_to_10kg", lambda y: 1 <= y < 10),
            ("at_least_10kg", lambda y: y >= 10),
        ):
            subset = [r for r in available.values() if predicate(r["target_kg"])]
            truth = np.array([r["target_kg"] for r in subset])
            baseline = errors(truth, np.array([r["previous_kg"] for r in subset]))
            revised = errors(truth, np.array([r["estimate_kg"] for r in subset]))
            groups[name] = dict(
                n=len(subset),
                previous=baseline.mean(0).tolist(),
                revised=revised.mean(0).tolist(),
                totals=revised.sum(0).tolist(),
            )
        ranked = sorted(available, key=lambda k: -abs(available[k]["error_kg"]))
        overview[split] = dict(
            n=len(available),
            mass_groups=groups,
            worst_ade_ids=ranked[:10],
            counts=dict(Counter(f for r in available.values() for f in r["flags"])),
            top10_ade_share=sum(abs(available[k]["error_kg"]) for k in ranked[:10])
            / sum(abs(r["error_kg"]) for r in available.values()),
        )
    assert len(records) == 500 and [
        overview[s]["n"] for s in ("train", "val", "test")
    ] == [300, 100, 100]
    result = dict(flags_and_checks=labels, overview=overview, records=records)
    (WORK / "object_structure_diagnostics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps(overview["test"], indent=2))
    for key in overview["test"]["worst_ade_ids"][:3]:
        r = records[key]
        print(
            key,
            {k: r[k] for k in ("target_kg", "estimate_kg", "largest_part", "flags")},
        )


if __name__ == "__main__":
    main()
