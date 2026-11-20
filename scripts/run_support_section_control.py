"""Check the physical mass of capped rectangular tubes using section dimensions.

The input dimensions are specified independently of the target mass. The hollow
interval begins after the first end cap and ends before the second end cap.
"""

import json
from pathlib import Path

import numpy as np


def section_mass(length_m, width_m, height_m, wall_m, density_kg_m3):
    """Return material mass for a solid prism or a capped hollow support."""
    values = np.array([length_m, width_m, height_m, wall_m, density_kg_m3])
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("Dimensions and density must be positive and finite")
    if 2 * wall_m >= min(length_m, width_m, height_m):
        raise ValueError("Wall leaves no valid inner cavity")
    outer = length_m * width_m * height_m
    inner = (length_m - 2 * wall_m) * (width_m - 2 * wall_m) * (height_m - 2 * wall_m)
    # Partition into disjoint side walls and two end caps.
    side_area = 2 * wall_m * (width_m + height_m - 2 * wall_m)
    material = side_area * (length_m - 2 * wall_m) + 2 * wall_m * width_m * height_m
    assert np.isclose(material, outer - inner, rtol=1e-10, atol=0)
    return {
        "solid_kg": outer * density_kg_m3,
        "section_kg": material * density_kg_m3,
        "volume_fraction": material / outer,
    }


def main():
    cases = {}
    for size in (0.02, 0.04):
        for wall in (0.001, 0.002, 0.004):
            cases[f"{size*1000:g}mm_section_{wall*1000:g}mm_wall"] = section_mass(
                1.0, size, size, wall, 7850
            )
    output = {
        "case": "Synthetic steel support with a square section, 1 m long and capped ends",
        "density_kg_m3": 7850,
        "results": cases,
    }
    path = (
        Path(__file__).resolve().parents[1]
        / "paper_assets/abo500_20261120/section_mass_control.json"
    )
    path.write_bytes((json.dumps(output, indent=2) + "\n").encode())


if __name__ == "__main__":
    main()
