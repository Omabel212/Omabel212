#!/usr/bin/env python3
"""
Manufacturing Defect Prediction — Self-Contained ML Example
============================================================
Generates synthetic process data, trains a logistic-regression-style
classifier using gradient descent (pure Python / standard library only),
evaluates classification metrics, and prints priority operator alerts
with recommended actions.

No external packages required — runs on any Python 3.6+ installation.
"""

import csv
import io
import math
import random

# ---------------------------------------------------------------------------
# 1. Synthetic data generation
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    "temperature_c",
    "pressure_bar",
    "vibration_mm_s",
    "humidity_pct",
    "cycle_time_s",
]

# Reasonable ranges for each feature
FEATURE_RANGES = {
    "temperature_c": (60.0, 120.0),
    "pressure_bar": (1.0, 10.0),
    "vibration_mm_s": (0.1, 5.0),
    "humidity_pct": (20.0, 90.0),
    "cycle_time_s": (10.0, 60.0),
}

# Weights used to create a realistic defect signal (hidden from the model)
TRUE_WEIGHTS = [0.03, 0.15, 0.40, 0.01, -0.02]
TRUE_BIAS = -5.0


def _sigmoid(z):
    """Numerically stable sigmoid function."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def generate_dataset(n_samples=1000, seed=42):
    """Return (rows, labels) where each row is a list of floats."""
    rng = random.Random(seed)
    rows, labels = [], []
    for _ in range(n_samples):
        row = []
        for feat in FEATURE_NAMES:
            lo, hi = FEATURE_RANGES[feat]
            row.append(rng.uniform(lo, hi))
        z = TRUE_BIAS + sum(w * x for w, x in zip(TRUE_WEIGHTS, row))
        prob = _sigmoid(z)
        label = 1 if rng.random() < prob else 0
        rows.append(row)
        labels.append(label)
    return rows, labels


# ---------------------------------------------------------------------------
# 2. Pre-processing helpers
# ---------------------------------------------------------------------------


def train_test_split(rows, labels, test_ratio=0.2, seed=42):
    """Shuffle and split data into training and test sets."""
    combined = list(zip(rows, labels))
    random.Random(seed).shuffle(combined)
    split = int(len(combined) * (1 - test_ratio))
    train = combined[:split]
    test = combined[split:]
    X_train = [r for r, _ in train]
    y_train = [l for _, l in train]
    X_test = [r for r, _ in test]
    y_test = [l for _, l in test]
    return X_train, y_train, X_test, y_test


def _col_stats(X, col):
    vals = [row[col] for row in X]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = math.sqrt(var) if var > 0 else 1.0
    return mean, std


def standardise(X_train, X_test):
    """Z-score standardisation fitted on training data."""
    n_features = len(X_train[0])
    stats = [_col_stats(X_train, i) for i in range(n_features)]
    def _transform(X):
        return [
            [(row[i] - stats[i][0]) / stats[i][1] for i in range(n_features)]
            for row in X
        ]
    return _transform(X_train), _transform(X_test), stats


# ---------------------------------------------------------------------------
# 3. Logistic regression via gradient descent
# ---------------------------------------------------------------------------


class LogisticRegression:
    """Minimal logistic regression trained with batch gradient descent."""

    def __init__(self, n_features, lr=0.1, epochs=200):
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.lr = lr
        self.epochs = epochs

    def _predict_prob(self, row):
        z = self.bias + sum(w * x for w, x in zip(self.weights, row))
        return _sigmoid(z)

    def fit(self, X, y):
        n = len(X)
        for epoch in range(self.epochs):
            grad_w = [0.0] * len(self.weights)
            grad_b = 0.0
            total_loss = 0.0
            for row, label in zip(X, y):
                pred = self._predict_prob(row)
                error = pred - label
                for j in range(len(self.weights)):
                    grad_w[j] += error * row[j]
                grad_b += error
                # log-loss for monitoring
                eps = 1e-15
                total_loss -= label * math.log(pred + eps) + (1 - label) * math.log(1 - pred + eps)
            for j in range(len(self.weights)):
                self.weights[j] -= self.lr * grad_w[j] / n
            self.bias -= self.lr * grad_b / n
            if (epoch + 1) % 50 == 0:
                print(f"  Epoch {epoch + 1:>4d}/{self.epochs}  —  loss: {total_loss / n:.4f}")

    def predict_proba(self, X):
        return [self._predict_prob(row) for row in X]

    def predict(self, X, threshold=0.5):
        return [1 if p >= threshold else 0 for p in self.predict_proba(X)]


# ---------------------------------------------------------------------------
# 4. Evaluation metrics
# ---------------------------------------------------------------------------


def confusion_counts(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return tp, fp, fn, tn


def evaluate(y_true, y_pred):
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    accuracy = (tp + tn) / len(y_true) if y_true else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


# ---------------------------------------------------------------------------
# 5. Operator alert system
# ---------------------------------------------------------------------------

ACTION_MAP = {
    "temperature_c": "Check cooling system and recalibrate thermal sensors.",
    "pressure_bar": "Inspect pressure relief valves and seals.",
    "vibration_mm_s": "Schedule bearing inspection and alignment check.",
    "humidity_pct": "Review environmental controls and desiccant packs.",
    "cycle_time_s": "Investigate conveyor speed and actuator response times.",
}


def generate_alerts(X_test, y_prob, stats, top_n=5):
    """
    For the highest-probability defect samples, identify the feature
    contributing most to the prediction and recommend an action.
    """
    indexed = sorted(enumerate(y_prob), key=lambda x: -x[1])
    alerts = []
    for rank, (idx, prob) in enumerate(indexed[:top_n], start=1):
        row = X_test[idx]
        # un-standardise and find the feature with the largest absolute
        # z-score (biggest deviation from normal)
        max_z, worst_feat = 0.0, 0
        for j, val in enumerate(row):
            if abs(val) > max_z:
                max_z = abs(val)
                worst_feat = j
        feat_name = FEATURE_NAMES[worst_feat]
        mean, std = stats[worst_feat]
        raw_value = row[worst_feat] * std + mean
        alerts.append({
            "rank": rank,
            "sample_idx": idx,
            "defect_probability": prob,
            "primary_factor": feat_name,
            "raw_value": raw_value,
            "recommended_action": ACTION_MAP[feat_name],
        })
    return alerts


def print_alerts(alerts):
    print("\n" + "=" * 72)
    print("  OPERATOR ALERTS — Priority Defect Predictions")
    print("=" * 72)
    for a in alerts:
        print(
            f"\n  [{a['rank']}] Sample #{a['sample_idx']:<4d}  "
            f"Defect probability: {a['defect_probability']:.2%}"
        )
        print(f"      Primary factor : {a['primary_factor']} = {a['raw_value']:.2f}")
        print(f"      Action         : {a['recommended_action']}")
    print("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# 6. Main pipeline
# ---------------------------------------------------------------------------


def main():
    print("Manufacturing Defect Prediction — ML Pipeline")
    print("-" * 50)

    # Generate data
    rows, labels = generate_dataset(n_samples=1000)
    defect_rate = sum(labels) / len(labels)
    print(f"Generated {len(rows)} samples  (defect rate: {defect_rate:.1%})")

    # Split
    X_train, y_train, X_test, y_test = train_test_split(rows, labels)
    print(f"Training samples: {len(X_train)}  |  Test samples: {len(X_test)}")

    # Standardise
    X_train_s, X_test_s, stats = standardise(X_train, X_test)

    # Train
    print("\nTraining logistic regression (gradient descent)…")
    model = LogisticRegression(n_features=len(FEATURE_NAMES), lr=0.1, epochs=200)
    model.fit(X_train_s, y_train)

    # Evaluate
    y_pred = model.predict(X_test_s)
    metrics = evaluate(y_test, y_pred)
    print("\nTest-set evaluation:")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1 Score  : {metrics['f1']:.4f}")
    print(f"  Confusion : TP={metrics['tp']}  FP={metrics['fp']}  "
          f"FN={metrics['fn']}  TN={metrics['tn']}")

    # Operator alerts
    y_prob = model.predict_proba(X_test_s)
    alerts = generate_alerts(X_test_s, y_prob, stats, top_n=5)
    print_alerts(alerts)

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
