# DCIFM — Fresh V2 Research Prototype

Dynamic Consumer Identity Forecasting Model (DCIFM)

## What this version fixes

### 1. Current vs emerging identity
The previous prototype could infer the same identity for both fields because it used essentially the same aggregate domain evidence.

This version uses **two separately trained identity classifiers**:
- Current identity → `identity_orientation_ground_truth`
- Emerging identity → `emerging_identity_ground_truth`

The two outputs are compared. A shift is shown only when the predictions differ. The application does not force a transition when the evidence does not support one.

### 2. Identity Forecast
The forecast page explicitly shows:

Current identity
↓
Detected behavioral direction
↓
Emerging identity
↓
Potential future-need category
↓
Potential future needs

The transition is an analytical forecast, not a psychological diagnosis.

### 3. YOLO visual analytics
The previous implementation used generic YOLOv8n classes. Generic COCO models cannot reliably detect specialized objects such as yoga mats, running shoes or protein containers.

This version **prefers YOLO-World** with a consumer-relevant vocabulary. It can attempt open-vocabulary detection for:
- bicycle
- sports equipment
- running shoe
- sneaker
- yoga mat
- dumbbell
- exercise equipment
- water bottle
- protein powder container
- laptop
- smartphone
- smartwatch
- backpack
- suitcase
- book
- and other common objects.

If YOLO-World is unavailable, the app falls back to YOLOv8n.

The visual branch remains a **contextual evidence source**, not an identity classifier.

### 4. Dashboard explanations
Every major page begins with a concise explanation of:
- what the section contains
- what its graphs mean
- why the analysis matters
- how the output relates to DCIFM

The UI also distinguishes classifier confidence from validated research accuracy.

## Dataset

`data/DCIFM_Master_Consumer_Dataset_10000.csv`

- 10,000 synthetic consumers
- one row per consumer
- cross-source behavioral aggregates
- behavioral-domain scores
- identity and future-behavior ground truth for controlled evaluation

The following columns are evaluation-only and are **excluded from model inputs**:

- identity_orientation_ground_truth
- identity_strength_ground_truth
- emerging_identity_ground_truth
- identity_change_score_ground_truth
- future_need_category_ground_truth
- future_purchase_probability_ground_truth
- future_purchase_30d_ground_truth
- future_purchase_value_ground_truth

This prevents target leakage.

## How to run in VS Code

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Demo accounts

- executive / executive123
- analyst / analyst123
- researcher / researcher123
- user / user123

These are demonstration credentials only.

## YOLO first run

The preferred visual detector is `yolov8s-worldv2.pt`. Ultralytics may download the model on first use. Internet access may therefore be required the first time the visual page is opened.

The fallback is `yolov8n.pt`.

## Research interpretation

This is a synthetic-data prototype. Its validation metrics demonstrate whether the implementation can recover the synthetic labels under a controlled train/test split. They must not be reported as evidence of real-world consumer prediction accuracy.

For a publishable empirical study, use a real longitudinal dataset with:
1. a validated cross-source identity mapping,
2. timestamps,
3. historical/current windows,
4. a future observation window,
5. explicit future behavior labels,
6. a separate labeled visual dataset if visual accuracy is claimed.

## Project structure

```text
DCIFM_Fresh_V2/
├── app.py
├── model.py
├── requirements.txt
├── README.md
├── SOURCES.md
├── .gitignore
├── data/
│   └── DCIFM_Master_Consumer_Dataset_10000.csv
├── models/
│   └── README.md
└── outputs/
    └── README.md
```
