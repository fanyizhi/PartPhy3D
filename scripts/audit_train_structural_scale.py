"""Audit fixed-wall vs proportional thickness scale substitutions on training objects."""

import json
import numpy as np
from evaluate_structure_cues import ROOT, WORK, read, sealed, errors

records = read(WORK / "structural_ablation.json")["records"]
catalog = read(WORK / "abo500_catalog_dimensions.json")["objects"]
rows = []
for folder in sorted(ROOT.glob("train_???")):
    for panel in sealed(folder / "panels"):
        key = panel["scene_id"]
        if key not in records or records[key]["split"] != "train":
            continue
        row = records[key]
        length = catalog.get(key, {}).get("max_dimension_m")
        est = panel.get("predicted_max_length_m")
        area = panel.get("area_m2")
        parts = panel.get("parts", [])
        if not (
            length
            and est
            and area
            and parts
            and np.isclose(sum(p["mass_kg"] for p in parts), row["raw"])
        ):
            continue
        if any(
            p.get("construction") not in ("solid", "shell", "partial") for p in parts
        ):
            continue
        k = length / est
        fixed = 0
        for part in parts:
            m = part["mass_kg"] * k**3
            if part["construction"] == "shell":
                m = part["density_kg_m3"] * min(
                    part["outer_volume_m3"] * k**3,
                    area * part["area_share"] * part["thickness_mm"] / 1000 * k**2,
                )
            fixed += m
        rows.append(
            dict(
                id=key,
                ratio=k,
                gt=row["gt"],
                raw=row["raw"],
                scaled=row["raw"] * k**3,
                fixed=fixed,
            )
        )
assert len({r["id"] for r in rows}) == len(rows)
y = np.array([r["gt"] for r in rows])
summary = {
    m: errors(y, np.array([r[m] for r in rows])).mean(0).tolist()
    for m in ("raw", "scaled", "fixed")
}
(WORK / "scale_diagnostic.json").write_text(
    json.dumps(dict(rows=rows, metrics=summary), indent=2)
)
print(len(rows), summary)
