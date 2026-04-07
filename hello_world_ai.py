import gi
import sys
import argparse
from pathlib import Path

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from aichart import AIChart
from aichart_app import AIChartApp
from data_sources import load_training_series

def main():
    parser = argparse.ArgumentParser(description="AIChart Stock Dip Predictor")
    parser.add_argument('--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    base_path = Path(__file__).parent
    model_path = base_path / "trained_model.pt"
    ai = AIChart()
    if args.retrain:
        training_series = [
            (file_name, [close for _, close in points])
            for file_name, points in load_training_series(base_path)
        ]
        if not training_series:
            print("No training data found in training_data/*.json")
            return

        total_samples = 0
        total_dips = 0
        training_closes = [closes for _, closes in training_series]
        training_summary = ai.fit(training_closes)
        ai.save_model(model_path)
        for file_name, closes in training_series:
            analysis = ai.predict(closes)
            total_samples += analysis["samples"]
            total_dips += len(analysis["dip_indices"])
            print(f"{file_name}: {analysis['samples']} samples, {len(analysis['dip_indices'])} major dips")

        print(
            f"Model retrained on {len(training_series)} files with {total_samples} total samples and {total_dips} detected dips."
        )
        print(
            f"Training loss: {training_summary['loss']:.4f} across {training_summary['training_points']} labeled points with {training_summary['positive_labels']} positives."
        )
        print(f"Saved trained model to {model_path}")
        return

    try:
        ai.load_model(model_path)
    except (OSError, ValueError, RuntimeError):
        pass

    app = AIChartApp()
    app.run(sys.argv)

if __name__ == "__main__":
    main()
