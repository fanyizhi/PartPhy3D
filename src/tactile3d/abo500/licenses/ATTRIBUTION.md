# ABO-500 five-object subset: attribution and terms

This development subset contains material from **Amazon Berkeley Objects (ABO)**,
by Amazon and the University of California, Berkeley, and the **ABO-500 derivative
release used by NeRF2Physics**, by Albert J. Zhai and collaborators.

- ABO dataset and original asset source:
  https://amazon-berkeley-objects.s3.amazonaws.com/index.html
- Current ABO license: Creative Commons Attribution 4.0 International (CC BY 4.0).
  The AWS Open Data registry records that the original noncommercial restriction
  was removed in 2023: https://registry.opendata.aws/amazon-berkeley-objects/
- Original license text:
  https://amazon-berkeley-objects.s3.amazonaws.com/LICENSE-CC-BY-4.0.txt
- NeRF2Physics release and its MIT license:
  https://github.com/ajzhai/NeRF2Physics

ABO reference: Collins et al., *ABO: Dataset and Benchmarks for Real-World 3D
Object Understanding*, CVPR 2022, https://arxiv.org/abs/2110.06199.
See the NeRF2Physics repository for the authors and citation of the derivative.

Changes made by PartPhy3D: selected the five fixed scene IDs listed in
`selected_objects.json`; copied original images, cameras and cached annotations;
added separately obtained native ABO reference GLBs; mapped product-ID weight
keys to scene IDs; created a representative-view contact sheet and integrity
manifests. Original observation and GLB bytes were preserved. The generated
contact sheet composites transparent backgrounds onto white and resizes images.

The project MIT license applies to PartPhy3D code, not a relicensing of ABO
assets. No endorsement by the original dataset creators is implied. Product
names and trademarks remain with their respective owners. The subset does not
include model weights, credentials, or newly measured internal-material labels.

The included material follows the upstream licenses, without warranties. Preserve
this attribution, the license texts and source/provenance records when sharing.
