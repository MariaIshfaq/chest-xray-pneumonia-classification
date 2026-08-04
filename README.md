# Chest X-ray Pneumonia Classification

A comparative study of CNNs, Vision Transformers, and LoRA-based 
fine-tuning for automated pneumonia detection from chest X-rays.

## Results Summary

| Model                | Accuracy | Trainable Params | Macro F1 |
|----------------------|----------|------------------|----------|
| ResNet18 CNN         | 85%      | 1,026            | 0.83     |
| ViT Frozen           | 86%      | 1,538            | 0.83     |
| ViT Full Fine-tuning | 86%      | 85,800,194       | 0.83     |
| **LoRA ViT ⭐**      | **90%**  | **296,450**      | **0.88** |

**Key finding:** LoRA achieves the best accuracy while training 
only 0.34% of total model parameters.

## Notebooks

| Notebook            | Description 
|---------------------|----------------
| 01_setup_and_data   | Dataset loading and preprocessing 
| 02_resnet18_cnn     | ResNet18 CNN baseline 
| 03_vit_finetuning   | ViT frozen and full fine-tuning 
| 04_lora_finetuning  | LoRA parameter-efficient fine-tuning 
| 05_final_comparison | Final comparison and W&B logging 

## Requirements

pip install torch torchvision timm peft wandb matplotlib scikit-learn seaborn

## Dataset

Chest X-ray Pneumonia dataset by Kermany et al. (2018)
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

## Experiment Tracking

W&B Dashboard: https://wandb.ai/maria-ishfaq/chest-xray-pneumonia
