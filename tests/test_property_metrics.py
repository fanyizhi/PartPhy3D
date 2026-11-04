import pytest

from tactile3d.eval.properties import evaluate_property_rows


def row(*, obj="one", sample="a", pred=2, target=1, prop="density", unit="kg/m^3",
        kind="measured"):
    return {"method": "model", "object_id": obj, "sample_id": sample,
            "property": prop, "unit": unit, "protocol": "fixed_reference_protocol",
            "predicted": str(pred), "target": str(target), "reference_kind": kind,
            "reference_source": "independent_measurement"}


def test_object_macro_avoids_weighting_densely_sampled_object_more():
    rows = [row(sample=str(i), pred=2, target=1) for i in range(10)]
    rows.append(row(obj="two", pred=5, target=1))
    group = evaluate_property_rows(rows)["groups"][0]
    assert group["object_macro_metrics"]["mae"] == 2.5
    assert group["object_macro_metrics"]["rmse"] == pytest.approx((8.5)**.5)
    assert group["num_scored_objects"] == 2


def test_missing_references_are_unscored_and_zero_is_not_replaced_by_epsilon():
    missing = row(target="", kind="")
    group = evaluate_property_rows([missing])["groups"][0]
    assert group["status"] == "reference_unavailable"
    assert all(value is None for value in group["object_macro_metrics"].values())
    group = evaluate_property_rows([row(pred=0, target=0)])["groups"][0]
    assert group["object_macro_metrics"]["mae"] == 0
    assert group["object_macro_metrics"]["alde"] is None
    assert group["eligible_points_by_metric"]["alde"] == 0


def test_poisson_can_be_negative_and_is_not_log_scored():
    group = evaluate_property_rows([row(pred=-.2, target=-.1,
                                      prop="poisson_ratio", unit="1")])["groups"][0]
    assert group["object_macro_metrics"]["mae"] == pytest.approx(.1)
    assert group["object_macro_metrics"]["mnre"] is None


def test_measured_and_pseudo_labels_remain_separate():
    groups = evaluate_property_rows([row(), row(obj="two", kind="pseudo_label")])["groups"]
    assert {g["reference_kind"] for g in groups} == {"measured", "pseudo_label"}
    assert len(groups) == 2


def test_units_duplicates_invalid_values_and_missing_prediction():
    with pytest.raises(ValueError, match="unit"):
        evaluate_property_rows([row(prop="youngs_modulus", unit="GPa")])
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_property_rows([row(), row()])
    with pytest.raises(ValueError, match="finite"):
        evaluate_property_rows([row(pred="nan")])
    group = evaluate_property_rows([row(pred="")])["groups"][0]
    assert group["status"] == "incomplete"
    assert group["prediction_coverage_on_labelled_rows"] == 0
