import json
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_LOOKBACK_DAYS = 365 * 10
TIINGO_API_BASE = "https://api.tiingo.com/tiingo/daily"
DEFAULT_SETTINGS_PATH = Path(__file__).with_name("settings.json")


def load_time_series_file(data_path):
    raw_text = data_path.read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    payload = json.loads(raw_text)
    return _parse_json_payload(payload)


def _ticker_from_data_path(data_path):
    stem = Path(data_path).stem
    if stem.startswith("data_"):
        stem = stem[5:]

    normalized = stem.strip().upper()
    if not normalized or not normalized.isalnum() or len(normalized) > 6:
        return None

    return normalized


def load_training_series(base_path):
    training_dir = Path(base_path) / "training_data"
    series_list = []

    for data_path in sorted(training_dir.glob("*.json")):
        ticker = _ticker_from_data_path(data_path)
        if ticker is None:
            continue
        try:
            points = load_time_series_file(data_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

        if points:
            series_list.append((data_path.name, points))

    return series_list


def load_dip_labels(base_path, labels_path=None):
    base_path = Path(base_path)
    resolved_labels_path = Path(labels_path) if labels_path else base_path / "dip_labels.json"

    try:
        payload = json.loads(resolved_labels_path.read_text(encoding="utf-8"))
    except OSError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"Labels file {resolved_labels_path} is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Labels file {resolved_labels_path} must contain a JSON object.")

    tickers_payload = payload.get("tickers", payload)
    if not isinstance(tickers_payload, dict):
        raise ValueError(f"Labels file {resolved_labels_path} must contain a ticker mapping.")

    labels_by_ticker = {}
    for ticker, value in tickers_payload.items():
        normalized_ticker = str(ticker).strip().upper()
        if not normalized_ticker:
            continue

        if isinstance(value, dict):
            raw_dates = value.get("sample_dips", [])
        else:
            raw_dates = value

        if not isinstance(raw_dates, list):
            raise ValueError(f"Label list for {normalized_ticker} must be an array of dates.")

        cleaned_dates = sorted({str(date_value).strip()[:10] for date_value in raw_dates if str(date_value).strip()})
        labels_by_ticker[normalized_ticker] = cleaned_dates

    return labels_by_ticker


def load_training_samples(base_path, labels_path=None):
    base_path = Path(base_path)
    labels_by_ticker = load_dip_labels(base_path, labels_path=labels_path)
    samples = []
    seen_tickers = set()

    for file_name, points in load_training_series(base_path):
        ticker = _ticker_from_data_path(file_name)
        if ticker is None:
            continue
        samples.append(
            {
                "ticker": ticker,
                "file_name": file_name,
                "points": points,
                "dip_dates": labels_by_ticker.get(ticker, []),
            }
        )
        seen_tickers.add(ticker)

    for data_path in sorted(base_path.glob("*.json")):
        ticker = _ticker_from_data_path(data_path)
        if ticker is None:
            continue
        if ticker in seen_tickers:
            continue
        try:
            points = load_time_series_file(data_path)
        except (OSError, json.JSONDecodeError, ValueError):
            continue

        if not points:
            continue

        samples.append(
            {
                "ticker": ticker,
                "file_name": data_path.name,
                "points": points,
                "dip_dates": labels_by_ticker.get(ticker, []),
            }
        )

    return samples


def load_tiingo_api_key(settings_path=DEFAULT_SETTINGS_PATH):
    try:
        payload = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Could not read settings file {settings_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Settings file {settings_path} is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Settings file {settings_path} must contain a JSON object.")

    api_key = str(payload.get("tiingo_api_key", "")).strip()
    if not api_key or api_key == "REPLACE_WITH_YOUR_TIINGO_API_KEY":
        raise ValueError(f"Set tiingo_api_key in {settings_path} before continuing.")

    return api_key


def fetch_tiingo_daily_prices(
    ticker,
    start_date=None,
    end_date=None,
    settings_path=DEFAULT_SETTINGS_PATH,
    api_key=None,
):
    normalized_ticker = str(ticker).strip().upper()
    if not normalized_ticker:
        raise ValueError("Provide a ticker symbol.")

    if start_date is None:
        start_date = (date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)).isoformat()
    if end_date is None:
        end_date = date.today().isoformat()
    if api_key is None:
        api_key = load_tiingo_api_key(settings_path)

    query = urlencode(
        {
            "startDate": start_date,
            "endDate": end_date,
            "resampleFreq": "daily",
        }
    )
    url = f"{TIINGO_API_BASE}/{normalized_ticker}/prices?{query}"
    request = Request(
        url,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise ValueError(f"Network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError("Tiingo returned invalid JSON") from exc

    if not isinstance(payload, list):
        raise ValueError(f"Unexpected response: {payload!r}")

    points = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        reported_at = str(item.get("date", ""))[:10]
        close_value = item.get("adjClose", item.get("close"))
        if not reported_at:
            continue
        try:
            close_number = float(close_value)
        except (TypeError, ValueError):
            continue
        points.append(
            {
                "reported_at": reported_at,
                "close": round(close_number, 4),
            }
        )

    points.sort(key=lambda point: point["reported_at"])
    return points


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