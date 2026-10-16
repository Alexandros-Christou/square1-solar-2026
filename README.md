# Estimate the sun: reproducible forecasting entry

AlexCChristou, Square1 2026. Autonomous AI-agent entry.

Use only the organiser's supplied train.csv and test.csv. No dataset or prediction CSV is included in this repository. Do not obtain withheld test targets or introduce external data.

Python 3.12.14. Install the dependencies from requirements.txt, then run:

```
python -m pip install -r requirements.txt
python forecast.py train.csv test.csv results
```

The script uses a chronological July 2024 to June 2025 validation year and compares two gradient-boosting configurations. It then trains the selected model on the full training file. Local validation MAE: 150.47602849088827 MW, calendar capacity-factor baseline: 606.9734427498794 MW. These are historical validation scores, not official test scores.

Outputs are generated locally: predictions.csv, validation_predictions.csv, metrics.json, error_analysis.json and model.joblib. Keep predictions and datasets private. The model file must only be loaded if you trust its source.

Data attribution: PV_Live by Sheffield Solar, University of Sheffield, CC BY 4.0; NASA Langley Research Center POWER Project. Obtain the authorised files and full data card from https://www.square1ai.com/competitions/solar-britain.

No claim of human authorship or independent human review is made. An autonomous AI agent developed this workflow, evaluated it locally and wrote the report. The code contains no user contact details, credentials or private application evidence.
