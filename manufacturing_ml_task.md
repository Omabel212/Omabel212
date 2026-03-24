# Machine Learning Task for a Manufacturing System

## Use Case
This task demonstrates how machine learning can be used in a manufacturing system to **predict product defects before they happen**.

## Business Objective
- Reduce scrap and rework
- Improve first-pass yield
- Alert operators before a quality failure becomes a batch issue
- Support maintenance and process optimisation decisions

## ML Workflow
1. Collect process data from machines, sensors, and operators.
2. Build a dataset with features such as:
   - Temperature
   - Vibration
   - Pressure
   - Humidity
   - Line speed
   - Tool wear
   - Shift
   - Product family
3. Train a classification model to predict whether a produced unit or batch is likely to be defective.
4. Score live or recent production data.
5. Trigger recommendations such as slowing the line, inspecting tooling, or holding a batch for review.

## Included Example
The Python script in `manufacturing_ml_task.py`:
- Creates synthetic manufacturing data
- Trains a logistic-regression-style classifier with gradient descent
- Evaluates model performance using classification metrics such as accuracy, precision, recall, and F1-score
- Produces high-priority operator alerts for the riskiest production records
- Runs with the Python standard library only, so it is easy to execute in restricted environments

## Example Integration in Industry
This task can be connected to:
- PLC/SCADA historians
- MES quality records
- Maintenance systems
- Industrial dashboards for operator decision support

## Why It Matters
This is a practical Industry 4.0 example because it combines:
- Industrial process understanding
- Machine learning
- Real-time operational decision support
- Manufacturing performance improvement
