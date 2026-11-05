# Third-party resources

Upstream data, code and weights retain their own licenses. The repository includes
the fixed five-object ABO-500 subset under `data/abo500_five/`, with its
`ATTRIBUTION.md` and license texts. No pretrained checkpoints or full dataset are
bundled. The project MIT license applies to our code, not a relicensing of the data.

| Resource | Role |
| --- | --- |
| [ABO](https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | Product data and geometry |
| [NeRF2Physics](https://github.com/ajzhai/NeRF2Physics) | Evaluation/data conventions and comparison |
| [PUGS](https://github.com/EverNorif/PUGS) | Physical-grounding comparison |
| [InstantMesh](https://github.com/TencentARC/InstantMesh) | Planned fixed reconstruction frontend |
| [PartField](https://research.nvidia.com/labs/toronto-ai/partfield-release/) | Planned initial part features |
| [Gen2Physics](https://arxiv.org/abs/2608.23869) | Closely related mesh/material/construction method |
| [VoMP](https://research.nvidia.com/labs/sil/projects/vomp/) | Related volumetric mechanical fields |
| [FlexiCubes](https://research.nvidia.com/labs/toronto-ai/flexicubes/) | Related controllable mesh extraction |
| [QEM](https://www.mgarland.org/research/quadrics.html) | Standard simplification foundation |
| [Manifold](https://github.com/elalish/manifold) | Apache-2.0 upstream Mesh boolean and section operations used in the native physical path |

Optional Qwen, CLIP and SAM-family models require model/version and license records.
Their methods are not PartPhy3D's original contributions. Describing a related
method does not mean it has been executed or reproduced here. Project licensing
does not grant rights to third-party assets.
