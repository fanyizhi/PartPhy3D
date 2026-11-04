from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tactile3d.experiments.edit_commands import (
    DEFAULT_COMMANDS,
    check_reference,
    export_result,
    synthetic_reference,
)
from tactile3d.experiments.part_edit_demo import make_example
from tactile3d.physics.edit_program import (
    MAX_EDIT_STEPS,
    EditProgram,
    execute_program,
    parse_instruction,
    program_from_commands,
)
from tactile3d.physics.part_editing import EditSpec
from tactile3d.physics.part_io import file_sha256, load_object, save_object
from tactile3d.physics.part_state import Interface, PhysicalObject


@pytest.mark.parametrize(
    "unit",
    ["10 mm", "1 cm", "0.01 m", "10 毫米", "1 厘米", "0.01 米"],
)
def test_units_and_explicit_part_alias(unit: str) -> None:
    spec = parse_instruction(
        f"将 左前腿 设为空心盒，材料 steel，壁厚 {unit}。", {"左前腿": "leg"}
    )
    assert spec.target_part_id == "leg"
    assert spec.wall_thickness_m == pytest.approx(0.01)


@pytest.mark.parametrize(
    "command",
    [
        "把椅子弄轻一点",
        "将 leg 的材料 wood 改为 steel",
        "将 leg 的材料 wood 改为 steel，密度 0 kg/m3",
        "将 leg 的材料 wood 改为 steel，密度 1e309 kg/m3",
        "将 leg 设为空心盒，材料 wood，壁厚 -1 mm",
        "将 leg 设为空心盒，材料 wood，壁厚 10 mm，然后删除其他部件",
    ],
)
def test_ambiguous_missing_or_trailing_parameters_are_rejected(command: str) -> None:
    with pytest.raises(ValueError):
        parse_instruction(command)


def test_three_edits_match_independent_analytic_mass_com_and_inertia() -> None:
    source, donor = make_example()
    donor_key = str(Path("synthetic_donor_for_test").resolve())
    program = program_from_commands(
        source, DEFAULT_COMMANDS, donors={"aluminum_leg": donor_key}
    )
    result = execute_program(source, program, {donor_key: donor})
    errors = check_reference(result, synthetic_reference())
    np.testing.assert_allclose(
        [state.total.mass_kg for state in result.states], [0.784, 3.088, 2.5228, 0.8568]
    )
    assert max(max(row.values()) for row in errors) < 1e-12
    assert [step.expected_part_version for step in result.program.steps] == [0, 1, 2]
    for state in result.states[1:]:
        assert state.parts[0] is source.parts[0]
        assert state._cache[0] is source._cache[0]
    assert result.final_state.part("leg").instance_id == donor.instance_id


def test_late_failure_rolls_back_entire_sequence_without_mutating_source() -> None:
    source, _ = make_example()
    before = source.total.properties()
    original_fractions = source.part("leg").fractions.copy()
    program = program_from_commands(
        source,
        [
            DEFAULT_COMMANDS[0],
            "将 leg 设为空心盒，材料 steel，壁厚 2 mm",
        ],
    )
    with pytest.raises(ValueError, match="第2步失败，整组未提交.*整数体素"):
        execute_program(source, program)
    assert source.total.properties() == before
    assert source.version == source.part("leg").version == 0
    np.testing.assert_array_equal(source.part("leg").fractions, original_fractions)


def test_incompatible_donor_is_rejected_after_valid_material_change() -> None:
    source, donor = make_example()
    interface = Interface(
        "mount", donor.interfaces[0].frame_local, np.array([0.03, 0.04])
    )
    bad = replace(donor, interfaces=(interface,))
    program = EditProgram(
        source.object_id,
        0,
        (
            parse_instruction(DEFAULT_COMMANDS[0]),
            EditSpec("leg", "replace_part", donor_state_ref="bad"),
        ),
    )
    with pytest.raises(ValueError, match="第2步失败.*不兼容"):
        execute_program(source, program, {"bad": bad})
    assert source.total.mass_kg == pytest.approx(0.784)


def test_identity_versions_unknown_parts_and_alias_collisions() -> None:
    source, _ = make_example()
    spec = parse_instruction(DEFAULT_COMMANDS[0])
    for program in (
        EditProgram("another_object", 0, (spec,)),
        EditProgram(source.object_id, 9, (spec,)),
        EditProgram(source.object_id, 0, (replace(spec, expected_part_version=9),)),
    ):
        with pytest.raises(ValueError):
            execute_program(source, program)
    with pytest.raises(ValueError, match="冲突"):
        program_from_commands(source, [DEFAULT_COMMANDS[0]], {"body": "leg"})
    with pytest.raises(ValueError, match="不存在部件"):
        program_from_commands(source, ["将 missing 设为实心，材料 wood"])
    with pytest.raises(ValueError, match="指令列表"):
        program_from_commands(source, DEFAULT_COMMANDS[0])
    with pytest.raises(ValueError):
        EditProgram(source.object_id, 0, (spec,) * (MAX_EDIT_STEPS + 1))


def test_strict_json_program_and_relative_donor_round_trip(tmp_path: Path) -> None:
    source, donor = make_example()
    save_object(
        PhysicalObject("donor", (donor,), source.grid_origin_m), tmp_path / "donor"
    )
    program = EditProgram(
        source.object_id, 0, (EditSpec("leg", "replace_part", donor_state_ref="donor"),)
    )
    file = tmp_path / "program.json"
    file.write_text(json.dumps(program.to_dict()), encoding="utf-8-sig")
    parsed = EditProgram.from_json(file)
    assert parsed.steps[0].donor_state_ref == str(tmp_path / "donor")
    result = execute_program(source, parsed)
    assert result.final_state.total.mass_kg == pytest.approx(0.8568)
    payload = program.to_dict()
    payload["ignore_this"] = True
    file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="字段"):
        EditProgram.from_json(file)
    del payload["ignore_this"]
    payload["schema_version"] = True
    file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        EditProgram.from_json(file)


def test_portable_bundle_freezes_donor_used_and_replays_without_original(
    tmp_path: Path,
) -> None:
    source, donor = make_example()
    donor_root = tmp_path / "original_donor"
    save_object(PhysicalObject("donor", (donor,), source.grid_origin_m), donor_root)
    program = EditProgram(
        source.object_id,
        0,
        (EditSpec("leg", "replace_part", donor_state_ref=str(donor_root)),),
    )
    result = execute_program(source, program)
    (donor_root / "state.json").write_text(
        "corrupted after execution", encoding="utf-8"
    )
    folder = tmp_path / "bundle"
    report = export_result(result, folder, None)
    replay_source = load_object(folder / "state_000")
    replay = execute_program(
        replay_source, EditProgram.from_json(folder / "program.json")
    )
    assert (
        replay.final_state.total.properties() == result.final_state.total.properties()
    )
    assert report["reference_type"] == "none"
    assert report["real_object_accuracy_evaluated"] is False
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected_hash in manifest["files"].items():
        assert file_sha256(folder / relative) == expected_hash
    for stem in ("geometry_updates", "physical_updates", "density_slices"):
        for suffix in ("png", "svg", "pdf"):
            assert (folder / f"{stem}.{suffix}").stat().st_size > 100
    with pytest.raises(FileExistsError):
        export_result(result, folder, None)


def test_cli_failure_creates_no_output_directory(tmp_path: Path) -> None:
    project = Path(__file__).resolve().parents[1]
    folder = tmp_path / "must_not_exist"
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    process = subprocess.run(
        [
            sys.executable,
            str(project / "scripts/edit_parts.py"),
            "--command",
            DEFAULT_COMMANDS[0],
            "--command",
            "将 leg 设为空心盒，材料 steel，壁厚 2 mm",
            "--output",
            str(folder),
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert process.returncode == 1
    assert "第2步失败，整组未提交" in process.stderr
    assert not folder.exists()
