<div align="center">

# CAPSNet

### Reliability Calibration for Proposal-Level Weakly-Supervised Temporal Action Localization

Weakly-supervised temporal action localization with class-aware action-core calibration and cross-video proposal support.

<p>
  <img alt="Task" src="https://img.shields.io/badge/task-WTAL-4051b5">
  <img alt="Framework" src="https://img.shields.io/badge/framework-PyTorch-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Datasets" src="https://img.shields.io/badge/datasets-THUMOS14%20%7C%20ActivityNet1.3-1f8a70">
  <img alt="Status" src="https://img.shields.io/badge/status-research%20code-555555">
</p>

</div>

<p align="center">
  <img src="assets/framework.svg" alt="CAPSNet framework" width="96%">
</p>

## Overview

CAPSNet is a proposal-level weakly-supervised temporal action localization (WTAL) framework. It starts from temporal proposals and calibrates proposal scores without changing the proposal-level prediction interface. The key idea is that a high proposal score is not always reliable: it may be driven by context, a short discriminative fragment, or an absent class that correlates with the current scene.

CAPSNet calibrates proposal reliability from two complementary views:

- **Class-Aware Action-Core Proportion (CAP)** separates action-core evidence from contextual responses inside each ordered proposal RoI.
- **Cross-Video Proposal Support (CPS)** retrieves same-class positive supports from other videos and edits proposal logits through present-versus-absent support competition.
- **Proposal-level calibration** keeps the P-MIL-style proposal scorer as the detection interface and improves the reliability of the final proposal-class ranking.

## Abstract

Weakly-supervised temporal action localization (WTAL) aims to localize action instances using only video-level labels. Proposal-level MIL alleviates the train-test mismatch of snippet-based WTAL by scoring temporal proposals during both training and inference. However, proposal scores are not necessarily reliable: a high score may be dominated by contextual evidence, a short discriminative fragment, or absent classes correlated with the scene of present actions. We propose CAPSNet, a dual reliability calibration framework for proposal-level WTAL. Without changing the proposal-level prediction interface, CAPSNet calibrates proposal scores from two complementary perspectives. First, Class-Aware Action-Core Proportion (CAP) models ordered RoI bins inside each proposal to estimate class-specific action-core dominance, separating action-core evidence from contextual responses and selectively correcting under-confident proposals supported by strong action-core evidence. Second, Cross-Video Proposal Support (CPS) retrieves same-class positive supports from other videos and uses present-versus-absent support competition to suppress context-driven and class-confused logits. Together, CAP and CPS provide internal and external reliability cues for calibrated proposal scoring. Extensive experiments on THUMOS14 and ActivityNet1.3 show that CAPSNet consistently improves over the proposal-level MIL baseline and achieves competitive performance against state-of-the-art WTAL methods.

## Results

### THUMOS14

Controlled ablations are reported under the same proposal source and non-fused proposal-level baseline. AVG is computed over IoU thresholds 0.1:0.1:0.7.

| Method | mAP@0.1 | mAP@0.3 | mAP@0.5 | mAP@0.7 | AVG |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 70.9 | 57.8 | 39.8 | 14.4 | 46.5 |
| Base + CAP | 71.9 | 59.4 | 40.9 | 15.8 | 47.6 |
| Base + CPS | 72.2 | 59.5 | 40.0 | 15.3 | 47.4 |
| Base + CAP w/o rescue + CPS | 72.6 | 60.0 | 40.5 | 15.5 | 47.8 |
| **CAPSNet** | **73.6** | **60.4** | **41.2** | **16.0** | **48.4** |

### ActivityNet1.3

The controlled Base row is the non-fused baseline used for gain computation. The P-MIL row is the fused literature entry.

| Method | mAP@0.5 | mAP@0.75 | mAP@0.95 | AVG |
| --- | ---: | ---: | ---: | ---: |
| Base (non-fused) | - | - | - | 23.9 |
| P-MIL (fused literature) | 41.8 | 25.4 | 5.2 | 25.5 |
| **CAPSNet** | 43.3 | **26.8** | 6.2 | **27.0** |

## Repository Layout

```text
.
鈹溾攢鈹€ assets/
鈹?  鈹斺攢鈹€ framework.svg
鈹溾攢鈹€ THUMOS14/
鈹?  鈹斺攢鈹€ run_pcm_stage1_v3_simplified/
鈹?      鈹溾攢鈹€ main.py
鈹?      鈹溾攢鈹€ model.py
鈹?      鈹溾攢鈹€ dataset.py
鈹?      鈹溾攢鈹€ options.py
鈹?      鈹溾攢鈹€ utils.py
鈹?      鈹溾攢鈹€ eval_detection.py
鈹?      鈹溾攢鈹€ opt.txt
鈹?      鈹斺攢鈹€ mAP-results.log
鈹溾攢鈹€ ActivityNet1.3/
鈹?  鈹斺攢鈹€ run_activitynet_pcm/
鈹?      鈹溾攢鈹€ code_backup/
鈹?      鈹?  鈹溾攢鈹€ main.py
鈹?      鈹?  鈹溾攢鈹€ model.py
鈹?      鈹?  鈹溾攢鈹€ dataset.py
鈹?      鈹?  鈹溾攢鈹€ options.py
鈹?      鈹?  鈹溾攢鈹€ utils.py
鈹?      鈹?  鈹斺攢鈹€ eval_detection.py
鈹?      鈹溾攢鈹€ opt.txt
鈹?      鈹溾攢鈹€ final_result.txt
鈹?      鈹斺攢鈹€ mAP-results.log
鈹斺攢鈹€ paper/
    鈹斺攢鈹€ versions/00_current_CAPSNet/
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
鈹溾攢鈹€ Thumos14reduced-I3D-JOINTFeatures.npy
鈹斺攢鈹€ Thumos14reduced-Annotations/
    鈹溾攢鈹€ labels_all.npy
    鈹溾攢鈹€ classlist.npy
    鈹溾攢鈹€ subset.npy
    鈹溾攢鈹€ videoname.npy
    鈹斺攢鈹€ Ambiguous_test.txt
```

Place proposal JSON files under the running directory:

```text
proposals/
鈹溾攢鈹€ detection_result_base_train.json
鈹斺攢鈹€ detection_result_base_test.json
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

### ActivityNet1.3

Place ActivityNet features, splits, and annotations under `--dataset_root`:

```text
ActivityNet13/
鈹溾攢鈹€ gt.json
鈹溾攢鈹€ split_train.txt
鈹溾攢鈹€ split_test.txt
鈹溾攢鈹€ train/
鈹?  鈹斺攢鈹€ <video_id>.npy
鈹斺攢鈹€ test/
    鈹斺攢鈹€ <video_id>.npy
```

ActivityNet also supports explicit proposal paths:

```bash
--proposal_train_path /path/to/pseudo_proposals_step0.json
--proposal_test_path  /path/to/final_test_ActivityNet_result.json
```

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

### Train on ActivityNet1.3

```bash
cd ActivityNet1.3/run_activitynet_pcm/code_backup

python main.py \
  --run_type train \
  --dataset_name ActivityNet13 \
  --dataset_root /path/to/ActivityNet13 \
  --num_class 200 \
  --exp_dir ../run_capsnet_activitynet \
  --proposal_train_path /path/to/pseudo_proposals_step0.json \
  --proposal_test_path /path/to/final_test_ActivityNet_result.json \
  --filter_train_proposal_by_label 1 \
  --activitynet_eval_mode calibrated \
  --max_epoch 50 \
  --interval 5
```

### Evaluate on ActivityNet1.3

```bash
cd ActivityNet1.3/run_activitynet_pcm/code_backup

python main.py \
  --run_type test \
  --dataset_name ActivityNet13 \
  --dataset_root /path/to/ActivityNet13 \
  --num_class 200 \
  --pretrained_ckpt /path/to/best_model.pkl \
  --proposal_test_path /path/to/final_test_ActivityNet_result.json \
  --activitynet_eval_mode calibrated
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
