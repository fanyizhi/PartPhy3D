# Research milestones

These milestones are pending research work, not promised performance.

Current scope, 2026-11-05: **five development objects**, using the first five IDs
of the previously frozen ten-object cohort. Check the first object before the
remaining four. Preserve failures and historical ten-object outputs. The current
selection is `configs/five_object_pilot.json`; use `scripts/run_five_objects.py`.

| Milestone | Acceptance | Stop or revise when |
| --- | --- | --- |
| Real frontend, one object | Actual generated geometry, scale/alignment and logs | Geometry or missing scale dominates |
| Parts and Qwen hypotheses | Auditable boundaries, interfaces and structured candidates | Hidden guesses lack support |
| Fixed five-object pilot | Same IDs/budgets, sealed predictions, failures retained | Inputs leak labels or failures disappear |
| Joint construction module | Training/validation supervision and mechanism ablations | Density/occupancy compensation is unexamined |
| Adaptive final construction | Coarse proposal, local budgets, valid transitions and actual cost | Full dense processing remains necessary |
| Synchronized editing | Independent edited references and fair full-recompute comparison | Only color or cache agreement is measured |
| Final benchmark/interaction | Held-out protocol, actual baselines and fixed reference dynamics | Development data are called unseen |

Freeze large frontends while establishing the baseline. Train small fusion and
construction modules before deciding whether QLoRA is needed. Physical measurements
and perceptual ratings have different targets. Select robot/rigid/deformable tests
according to the claims supported by independent references.
