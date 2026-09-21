# Explainable Detection of Chagas Disease
## Background
This project started from the 2025 PhysioNet challenge: [“Detection of Chagas Disease from the ECG”](https://moody-challenge.physionet.org/2025/). The original challenge goal is to have the highest true positive rate ranked by probability @ 5%. For this project I will attempt to design a model which performs well and then assess my model's decision making through Grad-CAM for interpretable results.

## Current State
I've trained a 1D ResNet model with Squeeze-Excitation blocks using RunPod GPUs and tested it using a held out set from the CODE-15%, SaMi-Trop, PTB-XL datasets. Completed POC for model explainability using Grad-CAM. Current Test AUROC: 0.83, TPR@top-5%: 0.3875

## Next Steps
Continue iterating on model hyperparameters and architecture to imporve performance while documenting results from an explainability perspective with Grad-CAM.

## Use of AI
Development for this project involves Claude Code (as seen in [contributor log](https://github.com/L-Sandler/chagas-ecg-explainability/graphs/contributors)). I've included my [.claude folder for this project](https://github.com/L-Sandler/chagas-ecg-explainability/tree/main/.claude) in this repository to demonstrate how I've been using it to make progress on this project while leaving room for independent learning and problem solving. 

