# Audio-Visual Batvision Dataset

This repository contains the official codebase for "The Audio-Visual BatVision Dataset for Research on Sight and Sound" (IROS 2023).

[Project Page](https://amandinebtto.github.io/Batvision-Dataset/) | [Dataset](https://cloud.minesparis.psl.eu/index.php/s/qurl3oySgTmT85M) | [Paper](https://ieeexplore.ieee.org/abstract/document/10341715)

For help contact amandine.brunetto [a.t] minesparis.psl.eu or open an issue.

## Dataset
The BatVision dataset is separated in two parts: BatVision V1, recorded at UC Berkely and BatVision V2, recorded at Ecole des Mines de Paris. While BV1 contains more data, BV2 contains more complex scenes featuring a wide variety of material, room shapes and objects (including a few outdoor data).

Binaural echoes are 0.5s long and sampled at 44,1kHz. They are synchronized with corresponding RGB-D images.

To get more information about the data and data collection, please check out our [project page](https://amandinebtto.github.io/Batvision-Dataset/) and [paper](https://ieeexplore.ieee.org/abstract/document/10341715).

## Usage

### Data
Batvision V1 and BatVision V2 have csv files necessary to split data in train, val and test.

All data of BatVision V1 are listed in `BatvisionV1/train.csv`, `val.csv` and `test.csv`.
In BatVision V2, each location is stored in separate folders containing `train.csv`, `val.csv` and `test.csv`.

Examples of dataloader are located in `UNetSoundOnly/dataloader`. 

### U-Net Baseline
We provide the code of the baseline presented in the paper. It consists in a U-Net architecture taking recorded binaural echoes as input to predict depth in the robot's field-of-view. 

Hydra is used for config file management. Examples of config files are given in `UNetSoundOnly/conf`. 
The code support TensorBoard for training visualization. 


## Citation
If you find this repository or the dataset useful, please cite:
```
@INPROCEEDINGS{10341715,
  author={Brunetto, Amandine and Hornauer, Sascha and Yu, Stella X. and Moutarde, Fabien},
  booktitle={2023 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={The Audio-Visual BatVision Dataset for Research on Sight and Sound}, 
  year={2023},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IROS55552.2023.10341715}}
```

## License
The Audio-Visual BatVision Dataset is CC-BY-SA-4.0 licensed, as found in the LICENSE file.
