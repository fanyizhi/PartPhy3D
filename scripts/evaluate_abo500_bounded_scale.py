"""Evaluate bounded changes to part volumes after a frozen structural calibration."""

import json
from itertools import product

import numpy as np

from evaluate_structure_cues import ROOT, WORK, errors, read


def main():
    old = read(WORK / "bounded_components.json")["predictions"]
    scale = read(WORK / "scale_geometry_study.json")
    geometry = read(WORK / "geometry_predictions.json")
    splits = read(ROOT / "splits.json")
    labels = read(ROOT / "weights.json")
    cohorts = {k: splits[k] for k in ("train", "val", "test")}
    ids = [k for group in cohorts.values() for k in group]
    assert len(ids) == len(set(ids)) == 500 and set(ids) == set(old)
    assert tuple(map(len, cohorts.values())) == (300, 100, 100)
    index = {key: i for i, key in enumerate(ids)}
    truth = np.array([labels.get(k, labels.get(k.split("_")[0])) for k in ids])
    base_masses, log_changes = [], []
    for key in ids:
        scored = old[key]["scored_parts"]
        initial = np.array([p["mass_kg"] for p in scored])
        raw = np.array([p["raw_kg"] for p in scored])
        change = np.zeros(len(scored))
        if scale["scale_inputs"][key]["status"] == "eligible":
            panel = scale["results"]["catalog_assisted"]["objects"][key]
            names = panel["component_names"]
            scaled = np.array(panel["before_calibration_component_kg"])
            reference = np.array(
                scale["results"]["structure"]["objects"][key][
                    "before_calibration_component_kg"
                ]
            )
            occupied = reference > 0
            if [p["name"] for p in scored] != [
                n for n, keep in zip(names, occupied, strict=True) if keep
            ]:
                raise ValueError(f"Part ordering mismatch for {key}")
            np.testing.assert_allclose(reference[occupied], raw)
            change = np.log(scaled[occupied] / raw)
        np.testing.assert_allclose(initial.sum(), old[key]["mass_kg"])
        base_masses.append(initial)
        log_changes.append(change)

    def allocate(weight, max_change):
        limit = np.log(max_change)
        return [
            m * np.exp(np.clip(weight * d, -limit, limit))
            for m, d in zip(base_masses, log_changes, strict=True)
        ]

    def calculate(parts, group):
        idx = np.array([index[k] for k in group])
        values = np.array([parts[i].sum() for i in idx])
        return errors(truth[idx], values).mean(0)

    # The frozen base already uses fold-specific weights for all training objects.
    # Give equal weight to the three error rates relative to their train baseline.
    train = cohorts["train"]
    base_score = calculate(base_masses, train)
    trials = []
    for weight, bound in [(0.0, 1.0)] + list(
        product((0.125, 0.25, 0.5, 1.0), (1.25, 1.5, 2.0))
    ):
        score = calculate(allocate(weight, bound), train)
        objective = np.mean(score[:3] / base_score[:3])
        trials.append(
            dict(
                weight=weight,
                bound=bound,
                metrics=score.tolist(),
                objective=float(objective),
            )
        )

    winner = min(trials, key=lambda t: t["objective"])
    chosen = allocate(winner["weight"], winner["bound"])
    raw = [np.array([geometry[k]["raw"]]) for k in ids]
    methods = {"raw": raw, "bounded_components": base_masses, "bounded_scale": chosen}
    paired = {}
    generator = np.random.default_rng(25)
    for split in ("val", "test"):
        idx = np.array([index[k] for k in cohorts[split]])
        current = np.array([chosen[i].sum() for i in idx])
        previous = np.array([base_masses[i].sum() for i in idx])
        delta = errors(truth[idx], current) - errors(truth[idx], previous)
        draws = generator.integers(0, len(idx), (10000, len(idx)))
        worst = np.argsort(delta[:, 0])[-3:][::-1]
        paired[split] = dict(
            changed=int(np.count_nonzero(np.abs(current - previous) > 1e-10)),
            mean_delta=delta.mean(0).tolist(),
            ci95=np.quantile(delta[draws].mean(1), [0.025, 0.975], axis=0).tolist(),
            most_increased=[
                dict(
                    id=ids[idx[j]],
                    target=float(truth[idx[j]]),
                    previous=float(previous[j]),
                    estimate=float(current[j]),
                )
                for j in worst
            ],
        )
    result = dict(
        train_selection=winner,
        paired=paired,
        candidates=trials,
        metrics={
            split: {
                name: calculate(values, group).tolist()
                for name, values in methods.items()
            }
            for split, group in {**cohorts, "all": ids}.items()
        },
        records={
            key: dict(
                split=old[key]["split"],
                target_kg=float(truth[i]),
                initial_kg=float(base_masses[i].sum()),
                predicted_kg=float(chosen[i].sum()),
                localized=old[key]["localized"],
                part_names=[p["name"] for p in old[key]["scored_parts"]],
                part_mass_kg=chosen[i].tolist(),
            )
            for i, key in enumerate(ids)
        },
    )
    (WORK / "allocation_full500.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps({k: result[k] for k in ("train_selection", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
