# Chaise reference-geometry study

Artist reference geometry is an additional input, not reconstructed geometry. Component assignments and surface property fields are VLM predictions/priors. Density maps describe surface material and do not assert homogeneous upholstery. Missing VLM density/modulus values bind to the existing configured material library; per-parameter provenance is recorded in source_data.json. Unknown properties remain gray. Hardness is not pooled across incompatible scales.

QEM comparison uses common requested budgets and reports actual face counts. Surface distances are evaluated in both directions using 12000 samples per surface. Physical errors concern only VLM-assigned closed wooden regions with fixed nominal density. Invalid domains have no physical-error value. Times include reduction and component mapping, exclude VLM, and summarize three warmed runs.

Local edit changes one closed wooden component to specified aluminum density. Mass and inertia refer to four closed wooden legs, excluding open rails, upholstery and hidden supports.
RGB and reference mesh: Amazon Berkeley Objects, Collins et al., CVPR 2022, CC BY 4.0. These renders, overlays and simplifications are derived analyses.

Contact panels ghost upholstery to show supports. Red circles mark exact surface pairs; colored samples are projected interface neighborhoods. Distances refer to the digital reference.
Grounding ablation uses identical VLM outputs for VLM and fusion; geometry-only removes appearance gating. The selected-view control also supplies per-region sampled texture RGB as visual evidence.
Matched timing uses one warmup and 15 alternating repeats per pair; bars show IQR. Physical metrics concern four closed legs. Missing closed-domain values remain unavailable.
