import json
from pathlib import Path


def load_time_series_file(data_path):
    raw_text = data_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    payload = json.loads(raw_text)
    return _parse_json_payload(payload)


def load_training_series(base_path):
    training_dir = Path(base_path) / "training_data"
    series_list = []

    for data_path in sorted(training_dir.glob("*.json")):
        try:
            points = load_time_series_file(data_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

        if points:
            series_list.append((data_path.name, points))

    return series_list


def _parse_json_payload(payload):
    if not isinstance(payload, list):
        raise ValueError("JSON payload must be an array.")

    points = []
    for item in payload:
        if not isinstance(item, dict) or "close" not in item:
            continue
        try:
            close_value = float(item["close"])
        except (TypeError, ValueError):
            continue
        label = str(item.get("reported_at", len(points)))
        points.append((label, close_value))

    points.sort(key=lambda point: point[0])
    return points