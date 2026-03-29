# Manufacturing Defect Prediction — ML Task Documentation

## Use Case

A manufacturing line produces components under varying process conditions
(temperature, pressure, vibration, humidity, cycle time). Undetected defects
lead to scrap, rework, warranty claims, and unplanned downtime. This task
demonstrates how a machine-learning model can predict defects in real time and
generate prioritised operator alerts with recommended corrective actions.

---

## Business Objectives

| Objective | Target |
|---|---|
| Reduce defect escape rate | Catch defects before end-of-line inspection |
| Lower scrap and rework costs | Prioritise high-risk batches for intervention |
| Improve operator response time | Provide actionable alerts with root-cause hints |
| Enable data-driven continuous improvement | Track feature importance over time |

---

## ML Workflow

```
1. Data Generation
   Synthetic process telemetry (temperature, pressure, vibration,
   humidity, cycle time) with a probabilistic defect label.

2. Pre-processing
   • Train / test split (80 / 20)
   • Z-score standardisation (fitted on training set only)

3. Model Training
   Logistic regression trained via batch gradient descent.
   Pure Python implementation — no external dependencies.

4. Evaluation
   Accuracy, precision, recall, F1 score, and confusion matrix
   computed on the held-out test set.

5. Operator Alerts
   The top-N highest-probability defect predictions are surfaced
   with the primary contributing feature and a recommended action.
```

---

## Integration Points

| System | Integration |
|---|---|
| **SCADA / Historian** | Ingests real-time sensor readings as model input |
| **MES** | Receives defect predictions to flag lots for inspection |
| **HMI / Dashboard** | Displays operator alerts and recommended actions |
| **Quality System** | Logs predictions for traceability and audit |
| **Maintenance (CMMS)** | Auto-creates work orders when specific factors recur |

---

## Why This Matters for Industry 4.0

* **Predictive over reactive** — shifting from end-of-line inspection to
  in-process prediction reduces waste and accelerates feedback loops.
* **Operator empowerment** — alerts include root-cause context (which sensor
  is abnormal) and a suggested action, reducing mean time to respond.
* **Portability** — the example runs on the Python standard library alone,
  making it deployable on edge devices, locked-down OT networks, and CI/CD
  pipelines without dependency management overhead.
* **Foundation for scaling** — the same workflow generalises to more complex
  models (random forests, neural networks) when external libraries are
  available, while the evaluation and alerting framework remains unchanged.

---

## Running the Example

```bash
python manufacturing_ml_task.py
```

No additional packages are required — Python 3.6+ is sufficient.

Expected output includes:

* Dataset summary (sample count, defect rate)
* Training progress (loss at every 50 epochs)
* Test-set metrics (accuracy, precision, recall, F1, confusion matrix)
* Top-5 priority operator alerts with recommended actions
