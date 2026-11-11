"""Bounded follow-up questions driven by previous VLM answers and mesh evidence."""

from __future__ import annotations

import json
from typing import Any


OUTPUT_SCHEMA = {
    "object": "object name",
    "components": [
        {
            "component_id": 0,
            "part_name": "name",
            "surface_material": "wood|fabric|metal|plastic|unknown",
            "construction": "solid|shell|upholstered|unknown",
            "contains_macroscopic_cavity": "yes|no|unknown",
            "evidence": "visible evidence or explicit assumption",
        }
    ],
    "materials": [
        {
            "name": "surface material",
            "density_kg_m3": None,
            "youngs_modulus_pa": None,
            "friction_coefficient": None,
            "hardness_value": None,
            "hardness_scale": None,
            "parameter_source": "inferred typical prior, not measured",
        }
    ],
    "unresolved": ["uncertain question"],
}


def followup_focus(previous: dict[str, Any]) -> str:
    """Select the next diagnostic without access to evaluation labels."""
    components = previous.get("components", [])
    if any(
        c.get("surface_material") == "fabric" and c.get("construction") == "solid"
        for c in components
        if isinstance(c, dict)
    ):
        return (
            "Resolve whether solid describes the textile fibers or the entire upholstered "
            "part. Separate fabric skin, possible foam and hidden support. Do not assume "
            "a closed exterior is homogeneous solid fabric."
        )
    if any(
        c.get("construction") in ("unknown", "shell")
        for c in components
        if isinstance(c, dict)
    ):
        return (
            "Inspect solid versus shell candidates. Distinguish a watertight digital "
            "surface from evidence of physical interior. State what remains unknown."
        )
    return (
        "Check material boundaries, duplicated or missing component IDs and physical "
        "parameter units. Reconsider low-evidence claims using the provided views."
    )


def make_prompt(
    evidence: list[dict], previous: dict | None = None, stage: str = "baseline"
) -> str:
    task = (
        "Inspect all original product images and the numbered reference-mesh component "
        "sheet. Image views are observations; the component sheet is an artist-model "
        "reference, not a physical interior measurement. Assign every listed component "
        "exactly once. Component numbers follow their row-major position in the sheet. "
        "A cloth surface does not imply homogeneous textile volume. Give typical material "
        "property priors only when meaningful; use null for unsupported hardness or modulus. "
        "Friction depends on a contact pair: use null unless you explicitly name a pair. "
        "Hardness values must name their scale. No target mass is provided. "
        "Return strict JSON in this schema: "
        + json.dumps(OUTPUT_SCHEMA)
        + " Mesh component observations: "
        + json.dumps(evidence)
    )
    if previous is not None:
        task += (
            " Previous answer to critique, not ground truth: "
            + json.dumps(previous)
            + " Next diagnostic: "
            + followup_focus(previous)
        )
    if stage == "overview":
        task += " First establish functional parts, surface materials and candidate constructions."
    elif stage == "geometry_check":
        task += " Check each candidate against mesh connectivity, dimensions and visible context."
    elif stage == "final":
        task += " Produce the revised complete state, retaining genuinely unresolved alternatives."
    return task


def validate_answer(answer: dict, count: int) -> dict:
    components = answer.get("components")
    if not isinstance(components, list) or len(components) != count:
        raise ValueError("One assignment per mesh component is required")
    ids = [c.get("component_id") for c in components]
    if any(type(i) is not int for i in ids) or sorted(ids) != list(range(count)):
        raise ValueError("Component IDs must be unique and complete")
    allowed = {"wood", "fabric", "metal", "plastic", "unknown"}
    for item in components:
        if item.get("surface_material") not in allowed:
            raise ValueError("Unsupported material label")
        if item.get("construction") not in {"solid", "shell", "upholstered", "unknown"}:
            raise ValueError("Unsupported construction label")
    if not isinstance(answer.get("materials"), list):
        raise ValueError("Material prior records required")
    return answer
