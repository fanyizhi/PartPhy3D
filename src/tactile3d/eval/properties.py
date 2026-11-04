"""Matched scalar physical-property evaluation, separated by reference provenance.

Input CSV rows refer to the same measured location/part and physical protocol.
References must be supplied independently. Empty references never become zero.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from tactile3d.utils.io import write_json

PROPERTIES = {
    "density": ("kg/m^3", 0, np.inf),
    "youngs_modulus": ("Pa", 0, np.inf),
    "poisson_ratio": ("1", -1, .5),
    "friction_coefficient": ("1", 0, np.inf),
    "contact_stiffness": ("N/m", 0, np.inf),
    "hardness_shore_a": ("Shore A", 0, 100),
    "hardness_shore_d": ("Shore D", 0, 100),
    "hardness_n2p_encoded": ("N2P A/D code", 0, 200),
}
REFERENCE_KINDS = {"measured", "known_synthetic", "pseudo_label"}
REQUIRED = {"method", "object_id", "sample_id", "property", "unit", "protocol",
            "predicted", "target", "reference_kind", "reference_source"}


def _number(value):
    if value is None or str(value).strip().lower() in {"", "na", "n/a", "none", "null"}:
        return None
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("Physical values must be finite, or explicitly missing")
    return result


def _check_domain(value, prop):
    _, low, high = PROPERTIES[prop]
    valid = low <= value <= high
    if prop == "poisson_ratio":
        valid = low < value < high
    if not valid:
        raise ValueError(f"{prop} value {value} is outside its supported domain")


def _scalar_errors(pred, target, prop):
    errors = {"mae": abs(pred-target), "mse": (pred-target)**2}
    # APE is defined at a positive target; log and ratio metrics require both.
    if prop != "poisson_ratio":
        if target > 0:
            errors["ape"] = abs(pred-target)/target
        if target > 0 and pred > 0:
            errors["alde"] = abs(float(np.log(pred)-np.log(target)))
            errors["mnre"] = min(pred/target, target/pred)
    return errors


def evaluate_property_rows(rows):
    """Macro-average per-object errors; report missing labels and predictions."""
    groups, seen = defaultdict(list), set()
    for row in rows:
        if not REQUIRED.issubset(row):
            raise ValueError(f"Missing CSV columns: {sorted(REQUIRED-set(row))}")
        prop = row["property"]
        if prop not in PROPERTIES:
            raise ValueError(f"Unknown property {prop!r}")
        if row["unit"] != PROPERTIES[prop][0]:
            raise ValueError(f"Expected unit {PROPERTIES[prop][0]!r} for {prop}; convert explicitly")
        if any(not row[key].strip() for key in ("method", "object_id", "sample_id", "protocol")):
            raise ValueError("Method, object, sample and protocol identifiers are required")
        key = tuple(row[k] for k in ("method", "object_id", "sample_id", "property", "protocol"))
        if key in seen:
            raise ValueError(f"Duplicate matched sample: {key}")
        seen.add(key)
        pred, target = _number(row["predicted"]), _number(row["target"])
        if pred is not None:
            _check_domain(pred, prop)
        if target is not None:
            _check_domain(target, prop)
            if row["reference_kind"] not in REFERENCE_KINDS or not row["reference_source"].strip():
                raise ValueError("A target requires an explicit reference kind and source")
        elif row["reference_kind"] and row["reference_kind"] not in REFERENCE_KINDS:
            raise ValueError("Unsupported reference kind")
        kind = row["reference_kind"] or "unavailable"
        group = (row["method"], prop, row["unit"], row["protocol"], kind)
        groups[group].append({**row, "pred": pred, "truth": target})
    results = []
    for (method, prop, unit, protocol, kind), entries in sorted(groups.items()):
        labelled = [r for r in entries if r["truth"] is not None]
        valid = [r for r in labelled if r["pred"] is not None]
        errors_by_object = defaultdict(list)
        for row in valid:
            errors_by_object[row["object_id"]].append(_scalar_errors(row["pred"], row["truth"], prop))
        metrics, eligible_points = {}, {}
        for metric in ("mae", "mse", "ape", "alde", "mnre"):
            object_values, count = [], 0
            for errors in errors_by_object.values():
                values = [error[metric] for error in errors if metric in error]
                if values:
                    object_values.append(float(np.mean(values)))
                    count += len(values)
            value = float(np.mean(object_values)) if object_values else None
            if metric == "mse":
                metrics["rmse"] = float(np.sqrt(value)) if value is not None else None
            else:
                metrics[metric] = value
            eligible_points["rmse" if metric == "mse" else metric] = count
        results.append({"method": method, "property": prop, "unit": unit,
            "protocol": protocol, "reference_kind": kind,
            "status": "reference_unavailable" if not labelled else
                      ("complete" if len(valid) == len(labelled) == len(entries) else "incomplete"),
            "num_rows": len(entries), "num_labelled_rows": len(labelled),
            "num_scored_rows": len(valid), "num_scored_objects": len(errors_by_object),
            "missing_reference_count": len(entries)-len(labelled),
            "missing_prediction_count_on_labelled_rows": len(labelled)-len(valid),
            "prediction_coverage_on_labelled_rows": len(valid)/len(labelled) if labelled else None,
            "object_macro_metrics": metrics, "eligible_points_by_metric": eligible_points})
    return {"aggregation": "per-object mean errors, then equal object weight; RMSE=sqrt(macro MSE)",
            "reference_groups_kept_separate": True, "ape_is_fraction_not_percent": True,
            "poisson_uses_absolute_metrics_only": True,
            "ratios_exclude_undefined_values_and_report_counts": True, "groups": results}


def make_pilot_reference_template(pilot_output, output_path):
    """Export a matching template without inventing local reference values.

    Density here is part-envelope effective density, not intrinsic phase density.
    Voigt/Reuss bounds are deliberately not exported as a measured-modulus prediction.
    """
    from tactile3d.utils.io import read_json

    folder = Path(pilot_output)
    scene_id = read_json(folder/"prediction.json")["scene_id"]
    with (folder/"part_physical_parameters.csv").open(encoding="utf-8") as stream:
        parts = list(csv.DictReader(stream))
    rows = []
    for part in parts:
        for prop, (unit, _, _) in PROPERTIES.items():
            if prop == "hardness_n2p_encoded":
                continue
            rows.append({"method": "untrained_construction_prior", "object_id": scene_id,
                "sample_id": f"part_{part['part_id']}", "property": prop, "unit": unit,
                "protocol": "part_envelope_effective_density" if prop == "density"
                            else "measurement_protocol_to_be_defined",
                "predicted": part["density_mean_kg_m3"] if prop == "density" else "",
                "target": "", "reference_kind": "", "reference_source": ""})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="matched scalar measurements CSV")
    parser.add_argument("--pilot-output", type=Path, help="create an unfilled pilot reference template")
    parser.add_argument("--template", type=Path, help="destination CSV for --pilot-output")
    parser.add_argument("--output", type=Path, required=True, help="evaluation JSON")
    args = parser.parse_args()
    if args.pilot_output:
        if args.input or not args.template:
            parser.error("Use --pilot-output with --template, without --input")
        rows = make_pilot_reference_template(args.pilot_output, args.template)
    elif args.input:
        with args.input.open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    else:
        parser.error("Provide --input or --pilot-output")
    report = evaluate_property_rows(rows)
    write_json(args.output, report)
    print(f"Saved {len(report['groups'])} property groups to {args.output}")


if __name__ == "__main__":
    main()
