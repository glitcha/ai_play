from pathlib import Path

import torch


class AIChart:
    MODEL_VERSION = 2
    WINDOW_RADIUS = 10
    MIN_SERIES_LENGTH = 32
    MIN_SCORE_FOR_DIP = 0.35

    def __init__(self):
        self.analysis = None
        self.model = None
        self.model_metadata = None
        self.training_summary = None
        self.feature_mean = None
        self.feature_std = None

    def train(self, closes, training_series=None, epochs=2400):
        if training_series is not None:
            self.fit(training_series, epochs=epochs)
        elif self.model is None:
            self.fit([closes], epochs=epochs)

        self.analysis = self._analyze(closes)
        return self.analysis

    def predict(self, closes):
        if self.model is None:
            return self.train(closes)
        self.analysis = self._analyze(closes)
        return self.analysis

    def fit(self, training_series, epochs=2400):
        prepared_series = [
            [float(value) for value in series]
            for series in training_series
            if len(series) >= self.MIN_SERIES_LENGTH
        ]
        if not prepared_series:
            raise ValueError("Need at least one training series with 32 or more closing prices.")

        feature_rows = []
        label_rows = []
        total_points = 0
        total_positive_labels = 0

        for closes in prepared_series:
            labels = self._label_dips(closes)
            total_positive_labels += sum(labels)
            total_points += len(closes)
            for index in range(len(closes)):
                feature_rows.append(self._extract_features(closes, index))
                label_rows.append(float(labels[index]))

        if total_positive_labels == 0:
            raise ValueError("Training data did not produce any dip labels.")

        torch.manual_seed(7)
        features = torch.tensor(feature_rows, dtype=torch.float32)
        labels = torch.tensor(label_rows, dtype=torch.float32)

        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0).clamp_min(1e-6)
        normalized_features = (features - self.feature_mean) / self.feature_std

        self.model = _DipClassifier(normalized_features.shape[1])
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        positive_weight = max((len(label_rows) - total_positive_labels) / max(total_positive_labels, 1), 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], dtype=torch.float32))
        training_epochs = min(600, max(80, epochs // 6))

        self.model.train()
        final_loss = None
        for _ in range(training_epochs):
            optimizer.zero_grad()
            logits = self.model(normalized_features)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().item())

        self.model.eval()
        self.model_metadata = self._build_model_metadata(prepared_series)
        self.training_summary = {
            "epochs": training_epochs,
            "loss": final_loss,
            "positive_labels": total_positive_labels,
            "training_points": total_points,
            "series_count": len(prepared_series),
            "window_radius": self.WINDOW_RADIUS,
            "feature_count": normalized_features.shape[1],
        }
        return self.training_summary

    def save_model(self, model_path):
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise ValueError("Train the model before saving it.")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "version": self.MODEL_VERSION,
                "state_dict": self.model.state_dict(),
                "input_size": int(self.feature_mean.numel()),
                "feature_mean": self.feature_mean,
                "feature_std": self.feature_std,
                "model_metadata": self.model_metadata,
                "training_summary": self.training_summary,
            },
            path,
        )

    def load_model(self, model_path):
        path = Path(model_path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        input_size = int(checkpoint["input_size"])
        model = _DipClassifier(input_size)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        self.model = model
        self.feature_mean = checkpoint["feature_mean"].detach().clone().float()
        self.feature_std = checkpoint["feature_std"].detach().clone().float().clamp_min(1e-6)
        self.model_metadata = checkpoint.get("model_metadata") or self._build_model_metadata([])
        self.training_summary = checkpoint.get("training_summary") or {}
        return self.training_summary

    def _analyze(self, closes):
        if len(closes) < 5:
            return {
                "scores": [0.0 for _ in closes],
                "dip_indices": [],
                "loss": self.training_summary.get("loss") if self.training_summary else None,
                "samples": 0,
                "model_kind": "torch",
            }

        scores = self._score_series(closes)
        dip_candidates = self._build_dip_candidates(closes, scores)
        dip_indices = self._select_top_dips(dip_candidates, len(closes))

        return {
            "scores": scores,
            "dip_indices": dip_indices,
            "projected_dip": self._project_next_dip(closes, dip_indices),
            "loss": self.training_summary.get("loss") if self.training_summary else None,
            "samples": len(closes),
            "model_kind": "torch",
        }

    def _score_series(self, closes):
        if self.model is None or self.feature_mean is None or self.feature_std is None:
            raise ValueError("Model is not loaded or trained.")

        features = torch.tensor([self._extract_features(closes, index) for index in range(len(closes))], dtype=torch.float32)
        normalized_features = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            logits = self.model(normalized_features)
            probabilities = torch.sigmoid(logits)
        return probabilities.tolist()

    def _build_dip_candidates(self, closes, scores):
        radius = max(2, self.WINDOW_RADIUS // 2)
        candidates = []
        for index, score in enumerate(scores):
            left = max(0, index - radius)
            right = min(len(closes), index + radius + 1)
            local_prices = closes[left:right]
            if closes[index] > min(local_prices):
                continue

            local_scores = scores[left:right]
            if score < max(local_scores):
                continue

            if score < self.MIN_SCORE_FOR_DIP:
                continue

            local_peak = max(local_prices)
            prominence = (local_peak - closes[index]) / max(abs(local_peak), 1e-6)
            candidates.append((score + prominence, index))

        if not candidates:
            candidates = [(score, index) for index, score in enumerate(scores)]

        candidates.sort(reverse=True)
        return candidates

    def _build_model_metadata(self, training_series):
        if not training_series:
            return {
                "median_length": 120,
                "median_volatility": 1.0,
                "series_count": 0,
            }

        lengths = sorted(len(series) for series in training_series)
        volatilities = sorted(self._series_volatility(series) for series in training_series)
        median_index = len(lengths) // 2
        return {
            "median_length": lengths[median_index],
            "median_volatility": max(volatilities[median_index], 1e-6),
            "series_count": len(training_series),
        }

    def _extract_features(self, closes, index):
        window = self._centered_window(closes, index, self.WINDOW_RADIUS)
        current_price = max(abs(closes[index]), 1e-6)
        normalized_window = [(value / current_price) - 1.0 for value in window]

        daily_returns = []
        for offset in range(1, 6):
            previous_index = max(0, index - offset)
            previous_price = closes[previous_index]
            daily_returns.append((closes[index] - previous_price) / max(abs(previous_price), 1e-6))

        ma_5 = self._window_average(closes, index, 5)
        ma_10 = self._window_average(closes, index, 10)
        ma_20 = self._window_average(closes, index, 20)
        local_min = min(window)
        local_max = max(window)
        local_range = max(local_max - local_min, 1e-6)
        trailing_peak = max(closes[max(0, index - 10):index + 1])
        diff_window = [current - previous for previous, current in zip(window, window[1:])]
        local_volatility = self._std(diff_window) / current_price
        slope_5 = (closes[index] - closes[max(0, index - 5)]) / current_price
        slope_10 = (closes[index] - closes[max(0, index - 10)]) / current_price

        return normalized_window + daily_returns + [
            (ma_5 / current_price) - 1.0,
            (ma_10 / current_price) - 1.0,
            (ma_20 / current_price) - 1.0,
            (closes[index] - local_min) / local_range,
            (closes[index] - trailing_peak) / max(abs(trailing_peak), 1e-6),
            local_volatility,
            slope_5,
            slope_10,
        ]

    def _centered_window(self, closes, index, radius):
        window = []
        for offset in range(-radius, radius + 1):
            bounded_index = min(max(index + offset, 0), len(closes) - 1)
            window.append(closes[bounded_index])
        return window

    def _window_average(self, closes, index, width):
        start = max(0, index - width + 1)
        window = closes[start:index + 1]
        return sum(window) / len(window)

    def _label_dips(self, closes, lookback=8, lookahead=8, rebound_days=12):
        labels = [0] * len(closes)
        for index in range(lookback, len(closes) - lookahead):
            current_price = closes[index]
            left = closes[index - lookback:index]
            right = closes[index + 1:index + lookahead + 1]
            if current_price > min(left + right + [current_price]):
                continue

            prior_peak = max(left)
            future_window = closes[index + 1:min(len(closes), index + rebound_days + 1)]
            if not future_window:
                continue

            future_peak = max(future_window)
            drop_ratio = (prior_peak - current_price) / max(abs(prior_peak), 1e-6)
            rebound_ratio = (future_peak - current_price) / max(abs(current_price), 1e-6)
            if drop_ratio >= 0.04 and rebound_ratio >= 0.05:
                labels[index] = 1

        return labels

    def _series_volatility(self, series):
        if len(series) < 2:
            return 1.0
        diffs = [abs(current - previous) for previous, current in zip(series, series[1:])]
        return sum(diffs) / len(diffs)

    def _select_top_dips(self, dip_candidates, series_length):
        reference_length = self.model_metadata["median_length"] if self.model_metadata else series_length
        min_separation = max(40, round(max(series_length, reference_length) / 8))
        dip_indices = []

        for _, index in dip_candidates:
            if any(abs(index - existing_index) < min_separation for existing_index in dip_indices):
                continue
            dip_indices.append(index)
            if len(dip_indices) == 3:
                break

        dip_indices.sort()
        return dip_indices

    def _std(self, values):
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        return variance ** 0.5

    def _project_next_dip(self, closes, dip_indices):
        if len(dip_indices) < 2:
            return None

        index_gaps = [
            current_index - previous_index
            for previous_index, current_index in zip(dip_indices, dip_indices[1:])
        ]
        average_gap = max(5, round(sum(index_gaps) / len(index_gaps)))

        last_dip_index = dip_indices[-1]
        bars_since_last_dip = (len(closes) - 1) - last_dip_index
        bars_ahead = max(5, average_gap - bars_since_last_dip)

        dip_prices = [closes[index] for index in dip_indices]
        recent_price_changes = [
            current_price - previous_price
            for previous_price, current_price in zip(dip_prices, dip_prices[1:])
        ]
        average_price_change = sum(recent_price_changes) / len(recent_price_changes)
        projected_price = max(0.0, dip_prices[-1] + average_price_change)

        return {
            "bars_ahead": bars_ahead,
            "target_price": round(projected_price, 2),
        }


class _DipClassifier(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, 1),
        )

    def forward(self, features):
        return self.layers(features).squeeze(-1)
