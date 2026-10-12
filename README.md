# CU-TSQ-MoE
A Counterfactual Utility-Guided Mixture-of-Experts Framework for Multitask Quality Assessment of Electrocardiogram, Photoplethysmogram, and Bone-Conducted Phonocardiogram
# CU-TSQ-MoE: Multimodal Signal Quality Assessment

## Overview

This repository accompanies our research on joint signal quality assessment for electrocardiogram (ECG), photoplethysmogram (PPG), and phonocardiogram (PCG) signals.

The overall framework combines raw-signal representation learning, feature-based learning, and residual expert fusion with counterfactual utility-guided routing. It jointly predicts the quality of all three signal modalities. The routing mechanism estimates whether modality-specific expert corrections are beneficial and selectively applies them to the shared base predictions.

## Currently Released Code

### `1.train_raw_3signal_resnet1d_mtl.py`

This script implements the raw-signal multitask learning baseline and provides a pretrained backbone for subsequent stages of the framework.

Aligned ECG, PPG, and PCG waveform segments are stacked into a three-channel input and processed by a shared one-dimensional ResNet backbone. Three separate classification heads predict the signal quality of ECG, PPG, and PCG, respectively, with three classes for each task.

The script includes:

- **Data alignment:** Matches samples across modalities using `(basekey, segment)` and retains their intersection.
- **Multitask training:** Optimizes the sum of three weighted cross-entropy losses, with class weights estimated from the training set.
- **Model selection:** Selects the best checkpoint and performs early stopping using the mean validation macro-F1 across the three tasks.
- **Evaluation and feature export:** Saves evaluation metrics, predictions, confusion matrices, classification reports, and shared deep features for the validation and test sets.

The current partitioning is performed at the recording level.

## Full Code Release

This repository currently provides the raw-signal multitask baseline as an initial release. The remaining source code, including the feature-based learning, expert fusion, counterfactual utility-guided routing, and associated training and evaluation scripts, will be fully open-sourced in this repository upon acceptance of the manuscript.
