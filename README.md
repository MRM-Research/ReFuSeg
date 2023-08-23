# ReFuSeg
Pytorch code for our paper titled "ReFuSeg: Regularized Multi-Modal Fusion for Precise Brain Tumour Segmentation"
### Architecture
<div align="center">
  <img width="70%" alt="" src="media/model.png">
</div>
 Semantic segmentation of brain tumours is a fundamental
task in medical image analysis that can help clinicians in diagnosing the
patient and tracking the progression of any malignant entities. Accu-
rate segmentation of brain lesions is essential for medical diagnosis and
treatment planning. However, failure to acquire specific MRI imaging
modalities can prevent applications from operating in critical situations,
raising concerns about their reliability and overall trustworthiness. This
paper presents a novel multi-modal approach for brain lesion segmen-
tation that leverages information from four distinct imaging modalities
while being robust to real-world scenarios of missing modalities, such
as T1, T1c, T2, and FLAIR MRI of brains. Our proposed method can
help address the challenges posed by artifacts in medical imagery due to
data acquisition errors (such as patient motion) or a reconstruction algo-
rithm’s inability to represent the anatomy while ensuring a trade-off in
accuracy. Our proposed regularization module makes it robust to these
scenarios and ensures the reliability of lesion segmentation.
<!-- 

## Get Started
```
$ git clone https://github.com/Kasliwal17/ThermalSuperResolution.git
$ cd ThermalSuperResolution
```
## Dependencies 
- Pytorch 1.11.0
- Segmentation-models-pytorch
- wandb
## Train & Eval
```
$ python -m src.train
```
## Citation

If you find this method and/or code useful, please consider citing

```bibtex
@article{kasliwal2023corefusion,
  title={CoReFusion: Contrastive Regularized Fusion for Guided Thermal Super-Resolution},
  author={Kasliwal, Aditya and Seth, Pratinav and Rallabandi, Sriya and Singhal, Sanchit},
  journal={arXiv preprint arXiv:2304.01243},
  year={2023}
} -->
```
