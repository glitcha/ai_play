import argparse
import json
import shutil
from datetime import date, timedelta
from pathlib import Path

from data_sources import fetch_tiingo_daily_prices, load_tiingo_api_key


DEFAULT_TICKERS = ["MSFT", "NVDA", "TSLA", "V", "AAPL", "AMZN", "GOOGL", "META", "JPM", "UNH"]
DEFAULT_LOOKBACK_DAYS = 365 * 10
DEFAULT_SETTINGS_PATH = Path(__file__).with_name("settings.json")


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
            rows = fetch_tiingo_daily_prices(
                ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                api_key=api_key,
            )
        except ValueError as exc:
            failures.append(f"{ticker}: {exc}")
            continue

        if not rows:
            failures.append(f"{ticker}: no price rows returned")
            continue

        output_path = output_dir / f"{ticker.lower()}.json"
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
        return load_tiingo_api_key(settings_path)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()