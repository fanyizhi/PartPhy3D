import json

from PIL import Image

from tactile3d.training.dataset import prepare_sft_dataset
from tactile3d.training.psychophysics import fit_bradley_terry
from tactile3d.utils.io import read_jsonl


def test_prepare_sft_groups_objects_and_builds_multimodal_messages(tmp_path):
    image = tmp_path / "view.png"
    Image.new("RGB", (16, 16), "white").save(image)
    annotations = tmp_path / "annotations.jsonl"
    rows = [
        {
            "sample_id": f"part_{part}",
            "scene_id": "same_object",
            "part_id": part,
            "images": ["view.png"],
            "geometry": {"volume_m3": 0.001},
            "professional_evidence": [{"source_id": "lab-1", "kind": "measurement"}],
            "expert_target": {"object_category": "chair", "interior_hypotheses": []},
        }
        for part in range(2)
    ]
    annotations.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = prepare_sft_dataset(annotations, tmp_path / "sft", train_fraction=0.6, val_fraction=0.2)
    assert report["num_rejected"] == 0
    nonempty = [split for split, count in report["counts"].items() if count]
    assert len(nonempty) == 1
    samples = read_jsonl(tmp_path / "sft" / f"{nonempty[0]}.jsonl")
    assert len(samples) == 2
    assert samples[0]["messages"][1]["content"][0]["type"] == "image"


def test_bradley_terry_keeps_held_out_comparisons_for_evaluation(tmp_path):
    comparisons = tmp_path / "comparisons.jsonl"
    rows = [
        {"item_a": "steel", "item_b": "wood", "choice": "a", "split": "train"},
        {"item_a": "wood", "item_b": "foam", "choice": "a", "split": "train"},
        {"item_a": "steel", "item_b": "foam", "choice": "a", "split": "validation"},
    ]
    comparisons.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    result = fit_bradley_terry(comparisons, tmp_path / "scores.json", l2=0.1)
    assert result["scores"]["steel"] > result["scores"]["wood"]
    assert result["scores"]["wood"] > result["scores"]["foam"]
    assert result["metrics"]["validation"]["accuracy_without_ties"] == 1.0
