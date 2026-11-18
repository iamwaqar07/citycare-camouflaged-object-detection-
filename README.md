# CityCare Hospital – Camouflaged Object Detection System

## Project Overview
This project implements an attention-based Camouflaged Object Detection (COD) system using a Res2Net-50 backbone and CBAM attention module. It is trained and evaluated on the COD10K dataset.

## Features
- Res2Net-50 backbone
- CBAM attention mechanism
- Baseline and attention model comparison
- Different-backbone ablation experiment
- Evaluation using MAE, S-measure, Weighted F-measure, Max E-measure, and Adaptive F-measure
- Gradio interface for image segmentation

## Repository Structure
- `task3_attention_cod_res2net.ipynb` – Training and evaluation notebook
- `app.py` – Gradio demo application
- `best_model.pth` – Trained attention model
- `models/` – Model checkpoints
- `results/task3_model_comparison.csv` – Model comparison results
- `evaluation_report.pdf` – Evaluation report
- `requirements.txt` – Project dependencies

## Results
The fine-tuned Res2Net + CBAM model achieved better results than the fine-tuned baseline on MAE, S-measure, and Max E-measure. The baseline performed better on Weighted F-measure and Adaptive F-measure.

See `results/task3_model_comparison.csv` and `evaluation_report.pdf` for the detailed results.

## Run the Demo
Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python app.py
```

## Dataset
COD10K dataset was used for training, validation, and evaluation.

## Limitations
The ResNet-50 ablation model was trained for fewer epochs than the fine-tuned Res2Net models. More training and experimentation may improve the results.

## Author
Waqar Ahmad
