"""Audit explicit furniture role constraints without using labels in prediction."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from run_reference_mesh_study import active_states, load_prepared

from tactile3d.mesh.contacts import closest_surface_pair
from tactile3d.physics.part_io import write_json
from tactile3d.vision.role_constraints import furniture_role_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    mesh, _labels, pieces, _texture = load_prepared(args.source)
    state = active_states(args.source)["progressive"]
    materials = {c["component_id"]: c["surface_material"] for c in state["components"]}
    records = []
    for i, part in enumerate(pieces):
        proposal = furniture_role_candidates(
            part.bounds, mesh.bounds, materials[i], part.is_watertight
        )
        records.append({"component_id": i, "material": materials[i], **proposal})
    parts = []
    for record, original, part in zip(records, state["components"], pieces):
        role = record["allowed_roles"][0]
        parts.append(
            {
                **original,
                "part_id": f"part_{record['component_id']:03d}",
                "instance_version": 2,
                "functional_role": role,
                "part_name": f"{role}_{record['component_id']}",
                "vlm_original_part_name": original.get("part_name"),
                "role_evidence": record["evidence"],
                "role_source": "category-conditioned geometric constraints",
                "anchor_m": part.bounds.mean(0).tolist(),
                "material_parameters": state["materials"].get(record["material"], {}),
            }
        )
    connections = []
    legs = [r["component_id"] for r in records if r["allowed_roles"] == ["leg"]]
    rails = [r["component_id"] for r in records if r["allowed_roles"] == ["frame_rail"]]
    for leg in legs:
        for rail in rails:
            pair = closest_surface_pair(pieces[leg], pieces[rail])
            if pair["distance_m"] <= 0.001:
                connections.append(
                    {
                        "part_ids": [f"part_{leg:03d}", f"part_{rail:03d}"],
                        "relation": "geometric interface within 1 mm",
                        "surface_pair": pair,
                        "mechanical_connection_verified": False,
                    }
                )
    write_json(
        args.source / "part_state_geometry_constrained.json",
        {
            "schema": "object/part/construction_region/mesh_support",
            "version": 2,
            "geometry_sha256": hashlib.sha256(
                (args.source / "reference_arrays.npz").read_bytes()
            ).hexdigest(),
            "parts": parts,
            "connections": connections,
            "role_predictions_use_evaluation_annotations": False,
        },
    )
    # Annotation access occurs after every prediction has been formed.
    review = json.loads(
        (args.source / "views_contacts" / "role_review.json").read_text()
    )["roles"]
    correct = sum(
        r["allowed_roles"][0] == review[str(r["component_id"])] for r in records
    )
    report = {
        "scope": "single development reference with category-conditioned furniture rules",
        "role_accuracy": correct / len(records),
        "correct": correct,
        "components": len(records),
        "rules_have_not_been_validated_on_independent_objects": True,
        "records": records,
    }
    write_json(args.source / "views_contacts" / "geometry_role_results.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
