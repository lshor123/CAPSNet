<div align="center">

# CAPSNet

### Reliability Calibration for Proposal-Level Weakly-Supervised Temporal Action Localization

Weakly-supervised temporal action localization with class-aware action-core calibration and cross-video proposal support.

<p>
  <img alt="Task" src="https://img.shields.io/badge/task-WTAL-4051b5">
  <img alt="Framework" src="https://img.shields.io/badge/framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Datasets" src="https://img.shields.io/badge/datasets-THUMOS14%20%7C%20ActivityNet%20v1.3-1f8a70">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20code-555555">
</p>

</div>

<p align="center">
  <img src="https://anonymous.4open.science/api/repo/CAPSNet-1592/file/assets/framework.svg?v=3f3c3e74" alt="CAPSNet framework" width="96%">
</p>

## Overview

CAPSNet is a proposal-level weakly-supervised temporal action localization (WTAL) framework. It starts from temporal proposals and calibrates proposal scores without changing the proposal-level prediction interface. The key idea is that a high proposal score is not always reliable: it may be driven by context, a short discriminative fragment, or an absent class that correlates with the current scene.

CAPSNet calibrates proposal reliability from two complementary views:

- **Class-Aware Action-Core Proportion (CAP)** separates action-core evidence from contextual responses inside each ordered proposal RoI.
- **Cross-Video Proposal Support (CPS)** retrieves same-class positive supports from other videos and edits proposal logits through present-versus-absent support competition.
- **Proposal-level calibration** keeps the P-MIL-style proposal scorer as the detection interface and improves the reliability of the final proposal-class ranking.

## Abstract

Weakly-supervised temporal action localization (WTAL) aims to localize action instances using only video-level labels. Proposal-level MIL alleviates the train-test mismatch of snippet-based WTAL by scoring temporal proposals during both training and inference. However, proposal scores are not necessarily reliable: a high score may be dominated by contextual evidence, a short discriminative fragment, or absent classes correlated with the scene of present actions. We propose CAPSNet, a dual reliability calibration framework for proposal-level WTAL. Without changing the proposal-level prediction interface, CAPSNet calibrates proposal scores from two complementary perspectives. First, Class-Aware Action-Core Proportion (CAP) models ordered RoI bins inside each proposal to estimate class-specific action-core dominance, separating action-core evidence from contextual responses and selectively correcting under-confident proposals supported by strong action-core evidence. Second, Cross-Video Proposal Support (CPS) retrieves same-class positive supports from other videos and uses present-versus-absent support competition to suppress context-driven and class-confused logits. Together, CAP and CPS provide internal and external reliability cues for calibrated proposal scoring. Extensive experiments on THUMOS14 and ActivityNet v1.3 show that CAPSNet consistently improves over the proposal-level MIL baseline and achieves competitive performance against state-of-the-art WTAL methods.

## Results

### THUMOS14

Our experimental results on the THUMOS14 dataset are as follows.

| Method | mAP@0.1 | mAP@0.2 | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | AVG |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CAPSNet | 73.6 | 68.7 | 60.4 | 50.5 | 41.2 | 28.4 | 16.0 | 48.4 |

### ActivityNet v1.3

Our experimental results on the ActivityNet v1.3 dataset are as follows.

| Method | mAP@0.5 | mAP@0.75 | mAP@0.95 | AVG |
| --- | ---: | ---: | ---: | ---: |
| CAPSNet | 43.3 | 26.8 | 6.2 | 27.0 |

## Repository Layout

```text
.
|-- assets/
|   `-- framework.svg
|-- THUMOS14/
|   `-- run_pcm_stage1_v3_simplified/
|       |-- main.py
|       |-- model.py
|       |-- dataset.py
|       |-- options.py
|       |-- utils.py
|       |-- eval_detection.py
|       |-- opt.txt
|       `-- mAP-results.log
`-- paper/
    `-- versions/00_current_CAPSNet/
```

## Requirements

The code is implemented with PyTorch and expects a CUDA-capable environment.

```bash
conda create -n capsnet python=3.9 -y
conda activate capsnet

# Install the PyTorch build that matches your CUDA version:
# https://pytorch.org/get-started/locally/
pip install torch torchvision

pip install numpy pandas joblib tensorboard
```

The experiments use two-stream I3D features with 2048 dimensions. The code calls `torchvision.ops.roi_align`, so `torch` and `torchvision` must be ABI-compatible.

## Data Preparation

### THUMOS14

Place THUMOS14 features and annotations under `--dataset_root` with the following layout:

```text
Thumos14reduced/
|-- Thumos14reduced-I3D-JOINTFeatures.npy
`-- Thumos14reduced-Annotations/
    |-- labels_all.npy
    |-- classlist.npy
    |-- subset.npy
    |-- videoname.npy
    `-- Ambiguous_test.txt
```

Place proposal JSON files under the running directory:

```text
proposals/
|-- detection_result_base_train.json
`-- detection_result_base_test.json
```

Each proposal item should contain at least:

```json
{
  "segment": [12.5, 18.3],
  "label": "ActionClass",
  "score": 0.91
}
```

For THUMOS14, CAPSNet uses proposal boundaries from the external proposal source. The proposal labels and scores are not required by the core THUMOS14 proposal scorer.

## Usage

### Train on THUMOS14

```bash
cd THUMOS14/run_pcm_stage1_v3_simplified

python main.py \
  --run_type train \
  --dataset_name Thumos14reduced \
  --dataset_root /path/to/Thumos14reduced \
  --exp_dir run_capsnet_thumos \
  --max_epoch 500 \
  --interval 2
```

For staged CPS tuning from a pretrained CAP or base checkpoint:

```bash
python main.py \
  --run_type train \
  --dataset_name Thumos14reduced \
  --dataset_root /path/to/Thumos14reduced \
  --exp_dir run_capsnet_thumos_cps \
  --pretrained_ckpt /path/to/best_model.pkl \
  --freeze_pcm_branches 1 \
  --pcm_warmup_epoch 1 \
  --max_epoch 500 \
  --interval 2
```

### Evaluate on THUMOS14

```bash
cd THUMOS14/run_pcm_stage1_v3_simplified

python main.py \
  --run_type test \
  --dataset_name Thumos14reduced \
  --dataset_root /path/to/Thumos14reduced \
  --pretrained_ckpt /path/to/best_model.pkl
```

## Reproducibility Notes

- Training writes a full argument snapshot to `opt.txt`.
- Detection results are appended to `mAP-results.log`.
- Training automatically copies the current Python source files into `code_backup/` inside the experiment directory.
- The controlled baseline in the paper is the non-fused P-MIL-style proposal scorer under the same proposal source; literature comparison rows may use the original fused results reported by prior work.
- Results can vary with proposal quality, feature preprocessing, CUDA/cuDNN versions, and staged checkpoint selection.

## Paper

The paper source is included under:

```text
paper/versions/00_current_CAPSNet/
```

Current title:

```text
CAPSNet: Reliability Calibration for Proposal-Level Weakly-Supervised Temporal Action Localization
```

If you use this repository, please cite the final paper once the official bibliographic record is available.

## Acknowledgements

This project builds on the proposal-level WTAL line of work, especially proposal-level MIL for weakly-supervised temporal action localization and external S-MIL proposal generation. We thank the authors of the public WTAL datasets, feature releases, and evaluation protocols used by the community.

## License

No license file is included in this snapshot. Please contact the repository owner before redistributing or using the code outside research evaluation.
