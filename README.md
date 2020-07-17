# ODMS
__O__bject __D__epth via __M__otion and __S__egmentation Dataset<br />
ODMS is the first dataset for motion- and segmentation-based depth estimation. ODMS training data is configurable and extensible, with each training example consisting of a series of object segmentation masks, camera movement distances, and ground truth object depth. As a benchmark evaluation, we also provide four ODMS validation and test sets with over 15,650 examples in multiple domains, including robotics and driving. Using ODMS, we are able to train a network to perform object depth estimation in real-time robot grasping experiments, demonstrating how our approach to depth estimation is a viable tool for 3D perception from a single RGB camera.

Contact: Brent Griffin (griffb at umich dot edu)

## Publication
Please cite our [paper](https://arxiv.org/abs/2007.05676 "ECCV Paper") if you find it useful for your research.
```
@inproceedings{GrCoECCV20,
  author = {Griffin, Brent A. and Corso, Jason J.},
  booktitle={The European Conference on Computer Vision (ECCV)},
  title = {Learning Object Depth from Camera Motion and Video Object Segmentation},
  year = {2020}
}
```

## Quick Introduction

__ECCV 2020 Supplementary Video:__ https://youtu.be/c90Fg_whjpI

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/c90Fg_whjpI/0.jpg)](https://youtu.be/c90Fg_whjpI)

## Using ODMS

__Run__ ``./demo/demo_datagen.py`` to generate random ODMS data to train your model. <br />
Example training data configurations are provided in the ``./config/`` folder. Has the option to save a static dataset. <br />
[native Python, has scipy dependency]

__Run__ ``./demo/demo_dataset_eval.py`` to evaluate your model on the ODMS validation and test sets. <br />
Provides an example evaluation for the VOS-DE baseline. Results are saved in the ``./results/`` folder. <br />
[native Python, VOS-DE baseline has skimage dependency]

## Benchmark

| Method | Robot | Driving | Normal | Perturb | All |
| --------------- | --------------- | --------------- | --------------- | --------------- | --------------- |
| [ODN<sub>*lr*</sub>](https://arxiv.org/abs/2007.05676 "ECCV Paper") | 13.1 | 31.7 | 8.6 | 17.9 | 17.8 |
| [VOS-DE](https://arxiv.org/abs/1903.08336) | 32.6 | 36.0 | 7.9 | 33.6 | 27.5 |

Is your technique missing although it's published and the code is public? Let us know and we'll add it.

## Use

This code is available for non-commercial research purposes only.