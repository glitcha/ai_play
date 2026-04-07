import argparse
import json
import shutil
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_TICKERS = ["MSFT", "NVDA", "TSLA", "V"]
DEFAULT_LOOKBACK_DAYS = 365 * 10
TIINGO_API_BASE = "https://api.tiingo.com/tiingo/daily"
DEFAULT_SETTINGS_PATH = Path(__file__).with_name("tiingo_settings.json")


def main():
    parser = argparse.ArgumentParser(
        description="Rebuild training_data/*.json from Tiingo daily price history."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=DEFAULT_TICKERS,
        help="Tickers to fetch. Defaults to the built-in training ticker list.",
    )
    parser.add_argument(
        "--start-date",
        default=(date.today() - timedelta(days=DEFAULT_LOOKBACK_DAYS)).isoformat(),
        help="Inclusive start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--end-date",
        default=date.today().isoformat(),
        help="Inclusive end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).with_name("training_data"),
        type=Path,
        help="Directory where training JSON files will be written.",
    )
    parser.add_argument(
        "--settings-path",
        default=DEFAULT_SETTINGS_PATH,
        type=Path,
        help="JSON settings file containing the Tiingo API key.",
    )
    args = parser.parse_args()

    api_key = load_api_key(args.settings_path)

    tickers = [ticker.strip().upper() for ticker in args.tickers if ticker.strip()]
    if not tickers:
        raise SystemExit("Provide at least one ticker.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_training_directory(output_dir)

    failures = []
    written_files = []
    for ticker in tickers:
        try:
            rows = fetch_tiingo_prices(ticker, args.start_date, args.end_date, api_key)
        except RuntimeError as exc:
            failures.append(f"{ticker}: {exc}")
            continue

        if not rows:
            failures.append(f"{ticker}: no price rows returned")
            continue

        output_path = output_dir / f"data_{ticker.lower()}.json"
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        written_files.append(output_path.name)

    print(f"Wrote {len(written_files)} training files to {output_dir}")
    for file_name in written_files:
        print(f"  - {file_name}")

    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)


def clear_training_directory(training_dir):
    for child in training_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def load_api_key(settings_path):
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Could not read settings file {settings_path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Settings file {settings_path} is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise SystemExit(f"Settings file {settings_path} must contain a JSON object.")

    api_key = str(payload.get("tiingo_api_key", "")).strip()
    if not api_key or api_key == "REPLACE_WITH_YOUR_TIINGO_API_KEY":
        raise SystemExit(
            f"Set tiingo_api_key in {settings_path} before running this script."
        )

    return api_key


def fetch_tiingo_prices(ticker, start_date, end_date, api_key):
    query = urlencode(
        {
            "startDate": start_date,
            "endDate": end_date,
            "resampleFreq": "daily",
        }
    )
    url = f"{TIINGO_API_BASE}/{ticker}/prices?{query}"
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
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("Tiingo returned invalid JSON") from exc

    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected response: {payload!r}")

    rows = []
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
        rows.append(
            {
                "reported_at": reported_at,
                "close": round(close_number, 4),
            }
        )

    rows.sort(key=lambda row: row["reported_at"])
    return rows


if __name__ == "__main__":
    main()