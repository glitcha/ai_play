import gi
import sys
import argparse
from pathlib import Path

gi.require_version('Gtk', '4.0')
from gi.repository import Gtk

from aichart import AIChart
from aichart_app import AIChartApp
from data_sources import load_training_samples


def log_retrain_progress(message):
    print(f"[retrain] {message}", flush=True)

def main():
    parser = argparse.ArgumentParser(description="Cronse Stock Dip Predictor")
    parser.add_argument('--retrain', action='store_true', help='Retrain the model')
    args = parser.parse_args()

    base_path = Path(__file__).parent
    model_path = base_path / "trained_model.pt"
    ai = AIChart()
    if args.retrain:
        log_retrain_progress(f"Loading training samples from {base_path}")
        training_samples = load_training_samples(base_path)
        if not training_samples:
            print("No training data found in training_data/*.json or ticker JSON files in the project root")
            return

        total_samples = 0
        total_dips = 0
        log_retrain_progress(f"Loaded {len(training_samples)} training files. Starting fit.")
        training_summary = ai.fit(training_samples, progress_callback=log_retrain_progress)
        log_retrain_progress(f"Saving trained model to {model_path}")
        ai.save_model(model_path)
        log_retrain_progress("Scoring each training file with the retrained model")
        for sample in training_samples:
            file_name = sample["file_name"]
            closes = [close for _, close in sample["points"]]
            analysis = ai.predict(closes)
            total_samples += analysis["samples"]
            total_dips += len(analysis["dip_indices"])
            print(f"{file_name}: {analysis['samples']} samples, {len(analysis['dip_indices'])} major dips")

        print(
            f"Model retrained on {len(training_samples)} files with {total_samples} total samples and {total_dips} detected dips."
        )
        print(
            f"Training loss: {training_summary['loss']:.4f} across {training_summary['training_points']} labeled points with {training_summary['positive_labels']} positives."
        )
        print(
            f"Used explicit dip labels for {training_summary['explicitly_labeled_series_count']} training files."
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
