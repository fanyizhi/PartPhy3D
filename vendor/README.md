# Pinned SAM2 runtime

`wheels/sam_2-1.0-py3-none-any.whl` contains the upstream Python runtime,
configuration files and license notices from
[facebookresearch/sam2](https://github.com/facebookresearch/sam2/tree/2b90b9f5ceec907a1c18123530e92e794ad901a4).
It was built on Linux with `SAM2_BUILD_CUDA=0`; it contains no native CUDA
extension and no model weights. Our predictor disables postprocessing that
would require that optional extension. Neural inference still uses PyTorch CUDA.

The wheel is included so that `pip install -r requirements.txt` does not build
SAM2 in an isolated environment and unexpectedly download another PyTorch.
Install from the repository root. This source redistribution retains Apache-2.0
and the upstream BSD notice; see the two license files and `SOURCES.json`.
This is an upstream component, not a proposed research contribution.

Rebuild against the pinned source, after installing the project's PyTorch:

```bash
SAM2_BUILD_CUDA=0 python -m pip wheel --no-deps --no-build-isolation \
  /path/to/sam2-2b90b9f5ceec907a1c18123530e92e794ad901a4 -w /tmp/sam2-wheel
```

Wheel archive metadata can vary between builds; verify the shipped wheel against
the published SHA256 instead of assuming bitwise reproducibility of a rebuild.

`InstantMesh_LICENSE.txt` retains the Apache-2.0 license of the frozen front end
whose decoder interface is used by the inference-only sparse weight adapter.
