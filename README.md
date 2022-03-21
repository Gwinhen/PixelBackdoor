# PixelBackdoor

This is the implementation for CVPR 2022 paper "Better Trigger Inversion Optimization in Backdoor Scanning".

## Prerequisite

* Keras 2.3.0
* Tensorflow 1.14.0
* Tensorpack 0.9.8 (if load data using data generator)

## Usage

The main functions are located in `src/main.py` file. For a test drive, please use the following command:

   ```bash
   python3 src/main.py --phase evaluate
   ```

This will generate a backdoor trigger for the label pair from loggerhead turtle (source) to kangaroo (target). The result might be slightly different as in the paper due to randomness and GPU architectures.

The `src/main.py` script provides different options:

   * `--gpu`: GPU id for running the experiment, default is `0`
   * `--phase`: `evaluate` for trigger inversion, `test` for testing model performance
   * `--model`: model architecture, default is `resnet50`
   * `--pair`: label pair for generating the backdoor trigger, default is `33-104` (souce label: loggerhead turtle, target label: kangaroo)
   * `--seed`: random seed
   * `--batch_size`: batch size in trigger inversion and model testing
   * `--num_classes`: number of classes of the subject model
   * `--attack_size`: number of samples for inversing the backdoor trigger

## Reference

```
@inproceedings{taog2022better,
  title={Better Trigger Inversion Optimization in Backdoor Scanning},
  author={Tao, Guanhong and Shen, Guangyu and Liu, Yingqi and An, Shengwei and Xu, Qiuling and Ma, Shiqing and Li, Pan and Zhang, Xiangyu},
  booktitle={2022 Conference on Computer Vision and Pattern Recognition (CVPR 2022)},
  year={2022}
}
```
