"""Machine-learning task for a manufacturing system using only the standard library.

The script generates manufacturing process data, trains a logistic-regression-style
classifier with gradient descent, evaluates performance, and prints operator
recommendations for the highest-risk production records.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, List, Sequence, Tuple


RANDOM_SEED = 42


@dataclass(frozen=True)
class ManufacturingRecord:
    machine_id: str
    product_family: str
    shift: str
    temperature_c: float
    vibration_mm_s: float
    pressure_bar: float
    humidity_pct: float
    line_speed_units_min: float
    tool_wear_pct: float
    defect_flag: int


@dataclass(frozen=True)
class ManufacturingAlert:
    machine_id: str
    defect_probability: float
    action: str


def sigmoid(value: float) -> float:
    if value >= 0:
        exp_value = math.exp(-value)
        return 1.0 / (1.0 + exp_value)
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def build_synthetic_dataset(samples: int = 1500) -> List[ManufacturingRecord]:
    rng = random.Random(RANDOM_SEED)
    machines = ["Mixer-01", "Mixer-02", "Filler-01", "Press-01"]
    product_families = ["Capsule", "Tablet", "Powder"]
    shifts = ["Day", "Night"]

    records: List[ManufacturingRecord] = []
    for _ in range(samples):
        machine_id = rng.choice(machines)
        product_family = rng.choice(product_families)
        shift = rng.choice(shifts)

        temperature_c = rng.gauss(78, 7)
        vibration_mm_s = rng.gauss(3.4, 1.1)
        pressure_bar = rng.gauss(5.5, 0.9)
        humidity_pct = rng.gauss(46, 9)
        line_speed_units_min = rng.gauss(120, 18)
        tool_wear_pct = max(0.0, min(100.0, rng.gauss(42, 20)))

        risk_score = (
            0.06 * (temperature_c - 78)
            + 0.75 * (vibration_mm_s - 3.4)
            + 0.42 * (pressure_bar - 5.5)
            + 0.03 * (humidity_pct - 46)
            + 0.025 * (line_speed_units_min - 120)
            + 0.05 * (tool_wear_pct - 42)
            + (0.45 if shift == "Night" else 0.0)
            + (0.35 if product_family == "Powder" else 0.0)
            + (0.25 if machine_id == "Press-01" else 0.0)
        )

        defect_probability = sigmoid(risk_score - 1.8)
        defect_flag = 1 if rng.random() < defect_probability else 0

        records.append(
            ManufacturingRecord(
                machine_id=machine_id,
                product_family=product_family,
                shift=shift,
                temperature_c=round(temperature_c, 2),
                vibration_mm_s=round(vibration_mm_s, 2),
                pressure_bar=round(pressure_bar, 2),
                humidity_pct=round(humidity_pct, 2),
                line_speed_units_min=round(line_speed_units_min, 2),
                tool_wear_pct=round(tool_wear_pct, 2),
                defect_flag=defect_flag,
            )
        )
    return records


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def std_dev(values: Sequence[float], avg: float) -> float:
    variance = sum((value - avg) ** 2 for value in values) / len(values)
    return math.sqrt(variance) or 1.0


def fit_normalization(records: Sequence[ManufacturingRecord]) -> Dict[str, Tuple[float, float]]:
    numeric_fields = [
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "humidity_pct",
        "line_speed_units_min",
        "tool_wear_pct",
    ]
    stats: Dict[str, Tuple[float, float]] = {}
    for field in numeric_fields:
        values = [getattr(record, field) for record in records]
        avg = mean(values)
        stats[field] = (avg, std_dev(values, avg))
    return stats


def record_to_features(record: ManufacturingRecord, stats: Dict[str, Tuple[float, float]]) -> List[float]:
    features = [1.0]
    for field in [
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "humidity_pct",
        "line_speed_units_min",
        "tool_wear_pct",
    ]:
        avg, deviation = stats[field]
        features.append((getattr(record, field) - avg) / deviation)

    for machine in ["Mixer-01", "Mixer-02", "Filler-01", "Press-01"]:
        features.append(1.0 if record.machine_id == machine else 0.0)
    for family in ["Capsule", "Tablet", "Powder"]:
        features.append(1.0 if record.product_family == family else 0.0)
    features.append(1.0 if record.shift == "Night" else 0.0)
    return features


def train_logistic_regression(
    train_records: Sequence[ManufacturingRecord],
    stats: Dict[str, Tuple[float, float]],
    learning_rate: float = 0.08,
    epochs: int = 300,
) -> List[float]:
    feature_count = len(record_to_features(train_records[0], stats))
    weights = [0.0] * feature_count

    for _ in range(epochs):
        gradients = [0.0] * feature_count
        for record in train_records:
            features = record_to_features(record, stats)
            prediction = sigmoid(sum(weight * feature for weight, feature in zip(weights, features)))
            error = prediction - record.defect_flag
            for index, feature in enumerate(features):
                gradients[index] += error * feature

        scale = learning_rate / len(train_records)
        for index in range(feature_count):
            weights[index] -= scale * gradients[index]
    return weights


def predict_probability(record: ManufacturingRecord, stats: Dict[str, Tuple[float, float]], weights: Sequence[float]) -> float:
    features = record_to_features(record, stats)
    score = sum(weight * feature for weight, feature in zip(weights, features))
    return sigmoid(score)


def evaluate(test_records: Sequence[ManufacturingRecord], stats: Dict[str, Tuple[float, float]], weights: Sequence[float]) -> Dict[str, float]:
    predictions = []
    probabilities = []
    for record in test_records:
        probability = predict_probability(record, stats, weights)
        probabilities.append(probability)
        predictions.append(1 if probability >= 0.5 else 0)

    actual = [record.defect_flag for record in test_records]
    true_positive = sum(1 for y, p in zip(actual, predictions) if y == 1 and p == 1)
    true_negative = sum(1 for y, p in zip(actual, predictions) if y == 0 and p == 0)
    false_positive = sum(1 for y, p in zip(actual, predictions) if y == 0 and p == 1)
    false_negative = sum(1 for y, p in zip(actual, predictions) if y == 1 and p == 0)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    accuracy = (true_positive + true_negative) / len(test_records)
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_probability = sum(probabilities) / len(probabilities)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "avg_probability": avg_probability,
    }


def recommend_action(record: ManufacturingRecord, defect_probability: float) -> ManufacturingAlert:
    if defect_probability >= 0.80:
        action = "Hold batch, inspect tooling, and reduce line speed immediately"
    elif defect_probability >= 0.60:
        action = "Schedule operator inspection and verify process setpoints"
    elif record.tool_wear_pct > 75:
        action = "Plan maintenance at next changeover due to high tool wear"
    else:
        action = "Continue production with standard monitoring"

    return ManufacturingAlert(
        machine_id=record.machine_id,
        defect_probability=defect_probability,
        action=action,
    )


def split_dataset(records: Sequence[ManufacturingRecord], test_ratio: float = 0.25) -> Tuple[List[ManufacturingRecord], List[ManufacturingRecord]]:
    shuffled = list(records)
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(shuffled)
    split_index = int(len(shuffled) * (1 - test_ratio))
    return shuffled[:split_index], shuffled[split_index:]


def main() -> None:
    records = build_synthetic_dataset()
    train_records, test_records = split_dataset(records)
    stats = fit_normalization(train_records)
    weights = train_logistic_regression(train_records, stats)
    metrics = evaluate(test_records, stats, weights)

    print("=== Manufacturing Quality Prediction Report ===")
    print(f"Training samples: {len(train_records)}")
    print(f"Test samples: {len(test_records)}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-score:  {metrics['f1_score']:.3f}")
    print(f"Average predicted defect probability: {metrics['avg_probability']:.3f}")

    ranked_records = sorted(
        test_records,
        key=lambda record: predict_probability(record, stats, weights),
        reverse=True,
    )[:5]

    print("\n=== Priority Manufacturing Alerts ===")
    for record in ranked_records:
        probability = predict_probability(record, stats, weights)
        alert = recommend_action(record, probability)
        print(
            f"{alert.machine_id}: risk={alert.defect_probability:.2%} | "
            f"product={record.product_family} | shift={record.shift} | "
            f"temp={record.temperature_c}C | vibration={record.vibration_mm_s} mm/s | "
            f"tool_wear={record.tool_wear_pct}% | action={alert.action}"
        )


if __name__ == "__main__":
    main()
