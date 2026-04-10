from pathlib import Path
from statistics import median

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    raise ModuleNotFoundError(
        "PyTorch is not available in the active Python interpreter. "
        "Use /home/colin/src/ai_play/ml_env/bin/python or select the workspace interpreter in VS Code."
    ) from error


class AIChart:
    MODEL_VERSION = 5
    WINDOW_RADIUS = 10
    IMAGE_WINDOW_RADIUS = 24
    CHART_IMAGE_WIDTH = 32
    CHART_IMAGE_HEIGHT = 16
    MIN_SERIES_LENGTH = 32
    MIN_SCORE_FOR_DIP = 0.35
    MAX_DIPS = 8
    MIN_RELATIVE_DIP_STRENGTH = 0.55
    MIN_SHAPE_SCORE_FOR_DIP = 0.5
    MIN_DIP_SEPARATION = 14
    SHAPE_LOOKBACK = 12
    SHAPE_LOOKAHEAD = 12
    SOFT_LABEL_RADIUS = 3
    SMOOTH_DIP_LOSS_WEIGHT = 0.32
    SMOOTH_DIP_PRICE_TOLERANCE = 0.08
    CURVE_TARGET_WEIGHT = 0.88
    MAJOR_CYCLE_LOOKBACK = 220
    MAJOR_CYCLE_LOOKAHEAD = 160
    CYCLE_COLLAPSE_SEPARATION = 150
    CYCLE_COLLAPSE_PRICE_TOLERANCE = 0.12

    def __init__(self):
        self.analysis = None
        self.model = None
        self.model_metadata = None
        self.training_summary = None
        self.feature_mean = None
        self.feature_std = None

    def _report_progress(self, progress_callback, message):
        if progress_callback is not None:
            progress_callback(message)

    def train(self, closes, training_series=None, epochs=9400):
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

    def fit(
        self,
        training_series,
        epochs=18000,
        progress_callback=None,
        progress_interval=50,
        batch_size=512,
    ):
        prepared_samples = self._prepare_training_samples(training_series)
        if not prepared_samples:
            raise ValueError("Need at least one training series with 32 or more closing prices.")

        self._report_progress(
            progress_callback,
            f"Prepared {len(prepared_samples)} training series for model fitting.",
        )

        feature_rows = []
        image_rows = []
        label_rows = []
        smooth_target_rows = []
        total_points = 0
        total_positive_labels = 0
        labeled_series_count = 0

        sample_count = len(prepared_samples)
        for sample_index, sample in enumerate(prepared_samples, start=1):
            closes = sample["closes"]
            labels = sample["labels"]
            smooth_targets = sample["smooth_targets"]
            if sample["used_explicit_labels"]:
                labeled_series_count += 1
            total_positive_labels += len(sample["positive_indices"])
            total_points += len(closes)
            for index in range(len(closes)):
                feature_rows.append(self._extract_features(closes, index))
                image_rows.append(self._render_chart_image_window(closes, index))
                label_rows.append(float(labels[index]))
                smooth_target_rows.append(float(smooth_targets[index]))

            if progress_callback is not None and (
                sample_index == 1
                or sample_index == sample_count
                or sample_index % 5 == 0
            ):
                ticker = sample.get("ticker") or f"series {sample_index}"
                self._report_progress(
                    progress_callback,
                    f"Encoded {sample_index}/{sample_count} series ({ticker}) into training rows.",
                )

        if total_positive_labels == 0:
            raise ValueError("Training data did not produce any dip labels.")

        torch.manual_seed(7)
        self._report_progress(progress_callback, "Converting training rows to tensors.")
        features = torch.tensor(feature_rows, dtype=torch.float32)
        images = torch.tensor(image_rows, dtype=torch.float32).unsqueeze(1)
        labels = torch.tensor(label_rows, dtype=torch.float32)
        smooth_targets = torch.tensor(smooth_target_rows, dtype=torch.float32)

        self._report_progress(progress_callback, "Normalizing feature tensors.")
        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0).clamp_min(1e-6)
        normalized_features = (features - self.feature_mean) / self.feature_std

        self._report_progress(progress_callback, "Initializing dip classifier.")
        self.model = _DipClassifier(
            normalized_features.shape[1],
            self.CHART_IMAGE_HEIGHT,
            self.CHART_IMAGE_WIDTH,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.003)
        positive_weight = max((len(label_rows) - total_positive_labels) / max(total_positive_labels, 1), 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([positive_weight], dtype=torch.float32))
        training_epochs = min(1400, max(160, epochs // 6))

        self._report_progress(
            progress_callback,
            (
                f"Starting training for {training_epochs} epochs on {total_points} points "
                f"with {total_positive_labels} positive labels."
            ),
        )

        self.model.train()
        final_loss = None
        final_smooth_loss = 0.0
        report_interval = max(1, int(progress_interval))
        effective_batch_size = max(1, min(int(batch_size), len(labels)))
        batch_count = max(1, (len(labels) + effective_batch_size - 1) // effective_batch_size)
        self._report_progress(
            progress_callback,
            f"Training in {batch_count} batches per epoch (batch size {effective_batch_size}).",
        )
        for epoch_index in range(training_epochs):
            if progress_callback is not None and epoch_index == 0:
                self._report_progress(
                    progress_callback,
                    f"Epoch 1/{training_epochs} started.",
                )

            permutation = torch.randperm(len(labels))
            epoch_loss_total = 0.0
            epoch_smooth_loss_total = 0.0
            batch_samples_seen = 0

            for batch_index, start in enumerate(range(0, len(labels), effective_batch_size), start=1):
                batch_indices = permutation[start:start + effective_batch_size]
                batch_features = normalized_features[batch_indices]
                batch_images = images[batch_indices]
                batch_labels = labels[batch_indices]
                batch_smooth_targets = smooth_targets[batch_indices]
                batch_size_actual = int(batch_labels.shape[0])

                if progress_callback is not None and epoch_index == 0 and batch_index == 1:
                    self._report_progress(
                        progress_callback,
                        f"Epoch 1/{training_epochs}: running first batch of {batch_count}.",
                    )

                optimizer.zero_grad()
                logits = self.model(batch_features, batch_images)
                probabilities = torch.sigmoid(logits)
                classification_loss = loss_fn(logits, batch_labels)
                smooth_mask = (batch_smooth_targets > 0).float()
                smooth_loss = (((probabilities - batch_smooth_targets) ** 2) * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)
                loss = classification_loss + (smooth_loss * self.SMOOTH_DIP_LOSS_WEIGHT)
                loss.backward()
                optimizer.step()

                epoch_loss_total += float(loss.detach().item()) * batch_size_actual
                epoch_smooth_loss_total += float(smooth_loss.detach().item()) * batch_size_actual
                batch_samples_seen += batch_size_actual

            final_loss = epoch_loss_total / max(batch_samples_seen, 1)
            final_smooth_loss = epoch_smooth_loss_total / max(batch_samples_seen, 1)

            current_epoch = epoch_index + 1
            if progress_callback is not None and (
                current_epoch == 1
                or current_epoch == training_epochs
                or current_epoch % report_interval == 0
            ):
                self._report_progress(
                    progress_callback,
                    (
                        f"Epoch {current_epoch}/{training_epochs}: "
                        f"loss={final_loss:.4f}, smooth_loss={final_smooth_loss:.4f}"
                    ),
                )

        self.model.eval()
        self.model_metadata = self._build_model_metadata(prepared_samples)
        self.training_summary = {
            "epochs": training_epochs,
            "loss": final_loss,
            "smooth_loss": final_smooth_loss,
            "positive_labels": total_positive_labels,
            "training_points": total_points,
            "series_count": len(prepared_samples),
            "explicitly_labeled_series_count": labeled_series_count,
            "window_radius": self.WINDOW_RADIUS,
            "feature_count": normalized_features.shape[1],
            "chart_image_height": self.CHART_IMAGE_HEIGHT,
            "chart_image_width": self.CHART_IMAGE_WIDTH,
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
                "chart_image_height": self.CHART_IMAGE_HEIGHT,
                "chart_image_width": self.CHART_IMAGE_WIDTH,
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
        if int(checkpoint.get("version", 0)) != self.MODEL_VERSION:
            raise ValueError("Saved model version does not match the current dip model.")
        input_size = int(checkpoint["input_size"])
        chart_image_height = int(checkpoint.get("chart_image_height", self.CHART_IMAGE_HEIGHT))
        chart_image_width = int(checkpoint.get("chart_image_width", self.CHART_IMAGE_WIDTH))
        model = _DipClassifier(input_size, chart_image_height, chart_image_width)
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
        dip_indices = self._select_top_dips(closes, dip_candidates, len(closes))

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
        images = torch.tensor([self._render_chart_image_window(closes, index) for index in range(len(closes))], dtype=torch.float32).unsqueeze(1)
        normalized_features = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            logits = self.model(normalized_features, images)
            probabilities = torch.sigmoid(logits)
        return probabilities.tolist()

    def _build_dip_candidates(self, closes, scores):
        radius = max(2, self.WINDOW_RADIUS // 2)
        shape_scores = [self._shape_bottom_score(closes, index) for index in range(len(closes))]
        major_cycle_scores = [self._major_cycle_score(closes, index) for index in range(len(closes))]
        support_floor = self._broad_support_floor(closes)
        combined_scores = [
            (score * 0.24) + (shape_score * 0.36) + (major_cycle_score * 0.4)
            for score, shape_score, major_cycle_score in zip(scores, shape_scores, major_cycle_scores)
        ]
        candidates_by_index = {}

        for index, score in enumerate(scores):
            left = max(0, index - radius)
            right = min(len(closes), index + radius + 1)
            local_prices = closes[left:right]
            local_min = min(local_prices)
            local_max = max(local_prices)
            local_range = max(local_max - local_min, 1e-6)
            local_floor_tolerance = max(abs(closes[index]) * 0.0015, local_range * 0.05)
            if closes[index] > local_min + local_floor_tolerance:
                continue

            shape_score = shape_scores[index]
            combined_score = combined_scores[index]
            if shape_score < self.MIN_SHAPE_SCORE_FOR_DIP and score < self.MIN_SCORE_FOR_DIP:
                continue

            local_scores = combined_scores[left:right]
            if combined_score < max(local_scores):
                continue

            refined_index = self._refine_trough_index(closes, index, radius)
            local_peak = max(closes[left:refined_index + 1]) if refined_index >= left else local_max
            prominence = (local_peak - closes[refined_index]) / max(abs(local_peak), 1e-6)
            floor_gap = abs(closes[refined_index] - support_floor[refined_index]) / max(abs(support_floor[refined_index]), 1e-6)
            floor_closeness = max(0.0, 1.0 - (floor_gap / 0.1))
            candidate_strength = (
                (combined_score * 0.72)
                + (prominence * 0.18)
                + (major_cycle_scores[refined_index] * 0.22)
                + (floor_closeness * 0.12)
            )
            best_strength = candidates_by_index.get(refined_index)
            if best_strength is None or candidate_strength > best_strength:
                candidates_by_index[refined_index] = candidate_strength

        candidates = [(strength, index) for index, strength in candidates_by_index.items()]
        if not candidates:
            fallback_index = min(range(len(closes)), key=lambda index: closes[index])
            fallback_strength = max(scores[fallback_index], shape_scores[fallback_index])
            candidates = [(fallback_strength, fallback_index)]

        candidates.sort(reverse=True)
        return candidates

    def _build_model_metadata(self, training_series):
        if not training_series:
            return {
                "median_length": 120,
                "median_volatility": 1.0,
                "series_count": 0,
                "median_labeled_gap": 0,
                "median_positive_count": 0,
            }

        normalized_series = []
        positive_counts = []
        labeled_gaps = []
        for sample in training_series:
            if isinstance(sample, dict):
                closes = sample.get("closes", [])
                normalized_series.append(closes)
                positive_indices = list(sample.get("positive_indices", []))
                if positive_indices:
                    positive_counts.append(len(positive_indices))
                    labeled_gaps.extend(
                        current_index - previous_index
                        for previous_index, current_index in zip(positive_indices, positive_indices[1:])
                    )
            else:
                normalized_series.append(sample)

        lengths = sorted(len(series) for series in normalized_series)
        volatilities = sorted(self._series_volatility(series) for series in normalized_series)
        median_index = len(lengths) // 2
        return {
            "median_length": lengths[median_index],
            "median_volatility": max(volatilities[median_index], 1e-6),
            "series_count": len(normalized_series),
            "median_labeled_gap": int(round(median(labeled_gaps))) if labeled_gaps else 0,
            "median_positive_count": int(round(median(positive_counts))) if positive_counts else 0,
        }

    def _prepare_training_samples(self, training_series):
        prepared_samples = []
        for sample in training_series:
            if isinstance(sample, dict):
                points = sample.get("points") or []
                closes = [float(close) for _, close in points]
                if len(closes) < self.MIN_SERIES_LENGTH:
                    continue

                hard_labels = self._labels_from_dates(points, sample.get("dip_dates", []))
                used_explicit_labels = any(hard_labels)
                if used_explicit_labels:
                    labels = self._build_soft_dip_labels(closes, hard_labels)
                    smooth_targets = self._build_curved_dip_targets(closes, hard_labels)
                else:
                    hard_labels = self._label_dips(closes)
                    labels = [float(label) for label in hard_labels]
                    smooth_targets = self._build_curved_dip_targets(closes, hard_labels, exact_only=False)

                positive_indices = [index for index, label in enumerate(hard_labels) if label >= 1]
                prepared_samples.append(
                    {
                        "ticker": sample.get("ticker", ""),
                        "closes": closes,
                        "labels": labels,
                        "smooth_targets": smooth_targets,
                        "positive_indices": positive_indices,
                        "used_explicit_labels": used_explicit_labels,
                    }
                )
                continue

            closes = [float(value) for value in sample]
            if len(closes) < self.MIN_SERIES_LENGTH:
                continue
            hard_labels = self._label_dips(closes)
            prepared_samples.append(
                {
                    "ticker": "",
                    "closes": closes,
                    "labels": [float(label) for label in hard_labels],
                    "smooth_targets": self._build_curved_dip_targets(closes, hard_labels, exact_only=False),
                    "positive_indices": [index for index, label in enumerate(hard_labels) if label >= 1],
                    "used_explicit_labels": False,
                }
            )

        return prepared_samples

    def _labels_from_dates(self, points, dip_dates):
        normalized_dates = {str(dip_date).strip()[:10] for dip_date in dip_dates if str(dip_date).strip()}
        labels = []
        for label, _close in points:
            labels.append(1 if str(label)[:10] in normalized_dates else 0)
        return labels

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
        support_left = self._lower_envelope(window[: self.WINDOW_RADIUS + 1])
        support_center = self._lower_envelope(window)
        support_right = self._lower_envelope(window[self.WINDOW_RADIUS :])
        curve_bottom, curve_slope, curve_curvature = self._local_support_curve(window)

        return normalized_window + daily_returns + [
            (ma_5 / current_price) - 1.0,
            (ma_10 / current_price) - 1.0,
            (ma_20 / current_price) - 1.0,
            (closes[index] - local_min) / local_range,
            (closes[index] - trailing_peak) / max(abs(trailing_peak), 1e-6),
            local_volatility,
            slope_5,
            slope_10,
            (closes[index] - support_center) / current_price,
            (support_right - support_left) / current_price,
            ((support_left + support_right) - (2 * support_center)) / current_price,
            (closes[index] - curve_bottom) / current_price,
            curve_slope,
            curve_curvature,
        ]

    def _centered_window(self, closes, index, radius):
        window = []
        for offset in range(-radius, radius + 1):
            bounded_index = min(max(index + offset, 0), len(closes) - 1)
            window.append(closes[bounded_index])
        return window

    def _render_chart_image_window(self, closes, index):
        window = self._centered_window(closes, index, self.IMAGE_WINDOW_RADIUS)
        resampled_window = self._resample_series(window, self.CHART_IMAGE_WIDTH)
        if not resampled_window:
            return [[0.0 for _ in range(self.CHART_IMAGE_WIDTH)] for _ in range(self.CHART_IMAGE_HEIGHT)]

        min_value = min(resampled_window)
        max_value = max(resampled_window)
        value_range = max(max_value - min_value, max(abs(min_value), abs(max_value), 1.0) * 0.02, 1e-6)
        padded_min = min_value - (value_range * 0.08)
        padded_max = max_value + (value_range * 0.08)
        padded_range = max(padded_max - padded_min, 1e-6)

        image = [[0.0 for _ in range(self.CHART_IMAGE_WIDTH)] for _ in range(self.CHART_IMAGE_HEIGHT)]
        previous_row = None
        center_column = self.CHART_IMAGE_WIDTH // 2

        for column, value in enumerate(resampled_window):
            normalized_value = (value - padded_min) / padded_range
            normalized_value = min(max(normalized_value, 0.0), 1.0)
            row = int(round((self.CHART_IMAGE_HEIGHT - 1) * (1.0 - normalized_value)))
            row = min(max(row, 0), self.CHART_IMAGE_HEIGHT - 1)

            image[row][column] = 1.0
            for fill_row in range(row + 1, self.CHART_IMAGE_HEIGHT):
                image[fill_row][column] = max(image[fill_row][column], 0.14)

            if previous_row is not None:
                row_start = min(previous_row, row)
                row_end = max(previous_row, row)
                for draw_row in range(row_start, row_end + 1):
                    image[draw_row][column] = max(image[draw_row][column], 0.72)
            previous_row = row

            if abs(column - center_column) <= 1:
                image[row][column] = 1.0
                if row + 1 < self.CHART_IMAGE_HEIGHT:
                    image[row + 1][column] = max(image[row + 1][column], 0.85)

        return image

    def _resample_series(self, values, width):
        if not values:
            return []
        if len(values) == width:
            return list(values)
        if len(values) == 1:
            return [values[0] for _ in range(width)]

        result = []
        source_last_index = len(values) - 1
        for column in range(width):
            progress = (column * source_last_index) / max(width - 1, 1)
            left_index = int(progress)
            right_index = min(left_index + 1, source_last_index)
            blend = progress - left_index
            value = (values[left_index] * (1.0 - blend)) + (values[right_index] * blend)
            result.append(value)
        return result

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

             
            if self._shape_bottom_score(closes, index) < self.MIN_SHAPE_SCORE_FOR_DIP:
                continue

            prior_peak = max(left)
            future_window = closes[index + 1:min(len(closes), index + rebound_days + 1)]
            if not future_window:
                continue

            future_peak = max(future_window)
            drop_ratio = (prior_peak - current_price) / max(abs(prior_peak), 1e-6)
            rebound_ratio = (future_peak - current_price) / max(abs(current_price), 1e-6)
            if drop_ratio >= 0.03 and rebound_ratio >= 0.035:
                labels[index] = 1

        return labels

    def _build_soft_dip_labels(self, closes, hard_labels):
        labels = [0.0] * len(closes)
        weight_by_distance = {
            0: 1.0,
            1: 0.65,
            2: 0.35,
            3: 0.16,
        }

        for anchor_index, label in enumerate(hard_labels):
            if label < 1:
                continue

            for offset in range(-self.SOFT_LABEL_RADIUS, self.SOFT_LABEL_RADIUS + 1):
                candidate_index = anchor_index + offset
                if candidate_index < 0 or candidate_index >= len(closes):
                    continue

                distance_weight = weight_by_distance.get(abs(offset), 0.0)
                if distance_weight <= 0:
                    continue

                shape_score = self._shape_bottom_score(closes, candidate_index)
                trough_bonus = 1.0 if self._is_local_trough(closes, candidate_index) else 0.55
                soft_value = distance_weight * max(0.35, shape_score) * trough_bonus
                labels[candidate_index] = max(labels[candidate_index], min(1.0, soft_value))

        return labels

    def _build_curved_dip_targets(self, closes, hard_labels, exact_only=True):
        positive_indices = [index for index, label in enumerate(hard_labels) if label >= 1]
        targets = [0.0] * len(closes)
        if not positive_indices:
            return targets

        for positive_index in positive_indices:
            targets[positive_index] = 1.0

        if len(positive_indices) < 2:
            return targets

        tolerance = max(self.SMOOTH_DIP_PRICE_TOLERANCE, (self._series_volatility(closes) / max(abs(median(closes)), 1e-6)) * 5)
        for left_index, right_index in zip(positive_indices, positive_indices[1:]):
            span = right_index - left_index
            if span <= 1:
                continue

            left_price = closes[left_index]
            right_price = closes[right_index]
            left_slope = self._anchor_curve_slope(closes, positive_indices, positive_indices.index(left_index))
            right_slope = self._anchor_curve_slope(closes, positive_indices, positive_indices.index(right_index))
            for candidate_index in range(left_index + 1, right_index):
                if not self._is_local_trough(closes, candidate_index):
                    continue

                progress = (candidate_index - left_index) / span
                expected_price = self._hermite_interpolate(
                    left_price,
                    right_price,
                    left_slope * span,
                    right_slope * span,
                    progress,
                )
                relative_gap = abs(closes[candidate_index] - expected_price) / max(abs(expected_price), 1e-6)
                line_closeness = max(0.0, 1.0 - (relative_gap / tolerance))
                if line_closeness <= 0:
                    continue

                shape_score = self._shape_bottom_score(closes, candidate_index)
                local_curve_bottom, _curve_slope, curve_curvature = self._local_support_curve(
                    self._centered_window(closes, candidate_index, self.WINDOW_RADIUS)
                )
                local_curve_gap = abs(closes[candidate_index] - local_curve_bottom) / max(abs(local_curve_bottom), 1e-6)
                curve_fit_score = max(0.0, 1.0 - (local_curve_gap / max(tolerance, 1e-6)))
                curvature_bonus = min(1.0, max(0.0, curve_curvature * 120.0) + 0.35)
                base_score = (
                    (line_closeness * 0.5)
                    + (max(0.25, shape_score) * 0.3)
                    + (curve_fit_score * 0.15)
                    + (curvature_bonus * 0.05)
                )
                if exact_only:
                    base_score *= 0.75
                else:
                    base_score *= 0.9

                targets[candidate_index] = max(targets[candidate_index], min(0.92, base_score * self.CURVE_TARGET_WEIGHT))

        return targets

    def _build_smooth_dip_targets(self, closes, hard_labels, exact_only=True):
        return self._build_curved_dip_targets(closes, hard_labels, exact_only=exact_only)

    def _series_volatility(self, series):
        if len(series) < 2:
            return 1.0
        diffs = [abs(current - previous) for previous, current in zip(series, series[1:])]
        return sum(diffs) / len(diffs)

    def _major_cycle_score(self, closes, index):
        left = max(0, index - self.MAJOR_CYCLE_LOOKBACK)
        right = min(len(closes), index + self.MAJOR_CYCLE_LOOKAHEAD + 1)
        current_price = closes[index]
        past_window = closes[left:index + 1]
        future_window = closes[index:right]
        if len(past_window) < 5 or len(future_window) < 5:
            return 0.0

        past_peak = max(past_window)
        future_peak = max(future_window)
        drawdown = (past_peak - current_price) / max(abs(past_peak), 1e-6)
        rebound = (future_peak - current_price) / max(abs(current_price), 1e-6)

        broad_floor = self._lower_envelope(closes[left:right])
        floor_gap = abs(current_price - broad_floor) / max(abs(broad_floor), 1e-6)
        floor_closeness = max(0.0, 1.0 - (floor_gap / 0.12))

        drawdown_score = min(1.0, drawdown / 0.18)
        rebound_score = min(1.0, rebound / 0.22)
        return (drawdown_score * 0.4) + (rebound_score * 0.4) + (floor_closeness * 0.2)

    def _broad_support_floor(self, closes):
        if not closes:
            return []

        radius = min(max(90, len(closes) // 18), 220)
        floor = []
        for index in range(len(closes)):
            left = max(0, index - radius)
            right = min(len(closes), index + radius + 1)
            floor.append(self._lower_envelope(closes[left:right]))

        smooth_radius = max(3, radius // 6)
        smoothed_floor = []
        for index in range(len(floor)):
            left = max(0, index - smooth_radius)
            right = min(len(floor), index + smooth_radius + 1)
            smoothed_floor.append(sum(floor[left:right]) / (right - left))
        return smoothed_floor

    def _lower_envelope(self, values):
        if not values:
            return 0.0
        sorted_values = sorted(values)
        sample_size = min(3, len(sorted_values))
        return sum(sorted_values[:sample_size]) / sample_size

    def _local_support_curve(self, window):
        if not window:
            return 0.0, 0.0, 0.0

        center = len(window) // 2
        left_segment = window[:center + 1]
        right_segment = window[center:]
        left_bottom = self._lower_envelope(left_segment)
        center_bottom = self._lower_envelope(window[max(0, center - 2):min(len(window), center + 3)])
        right_bottom = self._lower_envelope(right_segment)

        left_x = -max(center, 1)
        center_x = 0
        right_x = max(len(window) - center - 1, 1)
        curve_bottom, slope, curvature = self._fit_parabola_through_points(
            (left_x, left_bottom),
            (center_x, center_bottom),
            (right_x, right_bottom),
        )
        scale = max(abs(center_bottom), 1e-6)
        return curve_bottom, slope / scale, curvature / scale

    def _fit_parabola_through_points(self, left_point, center_point, right_point):
        x1, y1 = left_point
        x2, y2 = center_point
        x3, y3 = right_point

        denominator = (x1 - x2) * (x1 - x3) * (x2 - x3)
        if abs(denominator) < 1e-6:
            baseline = y2
            slope = (y3 - y1) / max(x3 - x1, 1)
            return baseline, slope, 0.0

        a = (
            (x3 * (y2 - y1))
            + (x2 * (y1 - y3))
            + (x1 * (y3 - y2))
        ) / denominator
        b = (
            ((x3 ** 2) * (y1 - y2))
            + ((x2 ** 2) * (y3 - y1))
            + ((x1 ** 2) * (y2 - y3))
        ) / denominator
        c = (
            (x2 * x3 * (x2 - x3) * y1)
            + (x3 * x1 * (x3 - x1) * y2)
            + (x1 * x2 * (x1 - x2) * y3)
        ) / denominator
        return c, b, a

    def _anchor_curve_slope(self, closes, positive_indices, anchor_position):
        anchor_index = positive_indices[anchor_position]
        anchor_price = closes[anchor_index]
        if len(positive_indices) == 1:
            return 0.0

        if anchor_position == 0:
            next_index = positive_indices[1]
            return (closes[next_index] - anchor_price) / max(next_index - anchor_index, 1)

        if anchor_position == len(positive_indices) - 1:
            previous_index = positive_indices[-2]
            return (anchor_price - closes[previous_index]) / max(anchor_index - previous_index, 1)

        previous_index = positive_indices[anchor_position - 1]
        next_index = positive_indices[anchor_position + 1]
        return (closes[next_index] - closes[previous_index]) / max(next_index - previous_index, 1)

    def _hermite_interpolate(self, start_value, end_value, start_tangent, end_tangent, progress):
        t = min(max(progress, 0.0), 1.0)
        t2 = t * t
        t3 = t2 * t
        h00 = (2 * t3) - (3 * t2) + 1
        h10 = t3 - (2 * t2) + t
        h01 = (-2 * t3) + (3 * t2)
        h11 = t3 - t2
        return (h00 * start_value) + (h10 * start_tangent) + (h01 * end_value) + (h11 * end_tangent)

    def _select_top_dips(self, closes, dip_candidates, series_length):
        if not dip_candidates:
            return []

        strength_by_index = {index: strength for strength, index in dip_candidates}

        reference_length = self.model_metadata["median_length"] if self.model_metadata else series_length
        labeled_gap = (self.model_metadata or {}).get("median_labeled_gap", 0)
        if labeled_gap:
            min_separation = min(90, max(45, self.MIN_DIP_SEPARATION, round(labeled_gap / 20)))
        else:
            min_separation = min(90, max(45, self.MIN_DIP_SEPARATION, round(max(series_length, reference_length) / 30)))

        labeled_count = (self.model_metadata or {}).get("median_positive_count", 0)
        max_dips = min(self.MAX_DIPS, max(1, labeled_count + 2)) if labeled_count else self.MAX_DIPS
        strongest_candidate = dip_candidates[0][0]
        minimum_candidate_strength = strongest_candidate * self.MIN_RELATIVE_DIP_STRENGTH
        dip_indices = []
        for candidate_strength, index in dip_candidates:
            if candidate_strength < minimum_candidate_strength:
                continue
            if any(abs(index - existing_index) < min_separation for existing_index in dip_indices):
                continue
            dip_indices.append(index)
            if len(dip_indices) == max_dips:
                break

        if not dip_indices:
            dip_indices.append(dip_candidates[0][1])

        dip_indices = self._collapse_same_cycle_dips(closes, dip_indices, strength_by_index)
        dip_indices.sort()
        return dip_indices

    def _collapse_same_cycle_dips(self, closes, dip_indices, strength_by_index):
        if len(dip_indices) < 2:
            return dip_indices

        collapsed_indices = [dip_indices[0]]
        for candidate_index in dip_indices[1:]:
            previous_index = collapsed_indices[-1]
            gap = candidate_index - previous_index
            previous_price = closes[previous_index]
            candidate_price = closes[candidate_index]
            relative_price_gap = abs(candidate_price - previous_price) / max(abs(previous_price), 1e-6)

            if gap > self.CYCLE_COLLAPSE_SEPARATION or relative_price_gap > self.CYCLE_COLLAPSE_PRICE_TOLERANCE:
                collapsed_indices.append(candidate_index)
                continue

            segment = closes[previous_index:candidate_index + 1]
            segment_peak = max(segment) if segment else max(previous_price, candidate_price)
            rebound_from_previous = (segment_peak - previous_price) / max(abs(previous_price), 1e-6)
            rebound_from_candidate = (segment_peak - candidate_price) / max(abs(candidate_price), 1e-6)
            if max(rebound_from_previous, rebound_from_candidate) >= 0.22:
                collapsed_indices.append(candidate_index)
                continue

            previous_strength = strength_by_index.get(previous_index, 0.0)
            candidate_strength = strength_by_index.get(candidate_index, 0.0)
            if (candidate_strength > previous_strength + 0.04) or (candidate_price < previous_price * 0.985):
                collapsed_indices[-1] = candidate_index

        return collapsed_indices

    def _shape_bottom_score(self, closes, index):
        left_start = max(0, index - self.SHAPE_LOOKBACK)
        right_end = min(len(closes), index + self.SHAPE_LOOKAHEAD + 1)
        left_segment = closes[left_start:index + 1]
        right_segment = closes[index:right_end]
        if len(left_segment) < 3 or len(right_segment) < 3:
            return 0.0

        current_price = closes[index]
        local_window = left_segment[:-1] + right_segment
        local_min = min(local_window)
        local_max = max(local_window)
        local_range = max(local_max - local_min, 1e-6)
        valley_score = 1.0 - min(1.0, (current_price - local_min) / local_range)

        left_peak = max(left_segment[:-1])
        right_peak = max(right_segment[1:])
        drawdown = (left_peak - current_price) / max(abs(left_peak), 1e-6)
        rebound = (right_peak - current_price) / max(abs(current_price), 1e-6)
        drawdown_score = min(1.0, drawdown / 0.05)
        rebound_score = min(1.0, rebound / 0.06)

        left_slope = self._normalized_slope(left_segment)
        right_slope = self._normalized_slope(right_segment)
        left_slope_score = min(1.0, max(0.0, -left_slope) / 0.003)
        right_slope_score = min(1.0, max(0.0, right_slope) / 0.003)

        left_smoothness = self._trend_consistency(left_segment, direction=-1) * self._path_efficiency(left_segment)
        right_smoothness = self._trend_consistency(right_segment, direction=1) * self._path_efficiency(right_segment)

        return (
            (valley_score * 0.2)
            + (drawdown_score * 0.16)
            + (rebound_score * 0.16)
            + (left_slope_score * 0.16)
            + (right_slope_score * 0.16)
            + (left_smoothness * 0.08)
            + (right_smoothness * 0.08)
        )

    def _refine_trough_index(self, closes, index, radius):
        left = max(0, index - radius)
        right = min(len(closes), index + radius + 1)
        local_prices = closes[left:right]
        local_min = min(local_prices)
        local_range = max(max(local_prices) - local_min, 1e-6)
        trough_tolerance = max(abs(local_min) * 0.0015, local_range * 0.04)
        trough_indices = [
            candidate_index
            for candidate_index in range(left, right)
            if closes[candidate_index] <= local_min + trough_tolerance
        ]
        if not trough_indices:
            return index
        return max(trough_indices, key=lambda candidate_index: self._shape_bottom_score(closes, candidate_index))

    def _is_local_trough(self, closes, index, radius=None):
        if radius is None:
            radius = max(2, self.WINDOW_RADIUS // 2)
        left = max(0, index - radius)
        right = min(len(closes), index + radius + 1)
        local_prices = closes[left:right]
        local_min = min(local_prices)
        local_range = max(max(local_prices) - local_min, 1e-6)
        tolerance = max(abs(local_min) * 0.0015, local_range * 0.05)
        return closes[index] <= local_min + tolerance

    def _normalized_slope(self, values):
        if len(values) < 2:
            return 0.0
        start_value = max(abs(values[0]), 1e-6)
        return (values[-1] - values[0]) / start_value / max(len(values) - 1, 1)

    def _trend_consistency(self, values, direction):
        if len(values) < 2:
            return 0.0
        tolerance = max(abs(values[0]), 1e-6) * 0.0015
        diffs = [current - previous for previous, current in zip(values, values[1:])]
        if direction < 0:
            consistent_steps = sum(1 for diff in diffs if diff <= tolerance)
        else:
            consistent_steps = sum(1 for diff in diffs if diff >= -tolerance)
        return consistent_steps / len(diffs)

    def _path_efficiency(self, values):
        if len(values) < 2:
            return 0.0
        net_distance = abs(values[-1] - values[0])
        traveled_distance = sum(abs(current - previous) for previous, current in zip(values, values[1:]))
        return net_distance / max(traveled_distance, 1e-6)

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
    def __init__(self, input_size, image_height, image_width):
        super().__init__()
        self.feature_layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, 48),
            torch.nn.ReLU(),
            torch.nn.Linear(48, 24),
            torch.nn.ReLU(),
        )
        self.image_layers = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(8, 12, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        with torch.no_grad():
            sample_output = self.image_layers(torch.zeros(1, 1, image_height, image_width))
            image_feature_size = sample_output.numel()

        self.image_projection = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(image_feature_size, 24),
            torch.nn.ReLU(),
        )
        self.output_layers = torch.nn.Sequential(
            torch.nn.Linear(48, 24),
            torch.nn.ReLU(),
            torch.nn.Linear(24, 1),
        )

    def forward(self, features, images):
        feature_embedding = self.feature_layers(features)
        image_embedding = self.image_projection(self.image_layers(images))
        combined_embedding = torch.cat([feature_embedding, image_embedding], dim=1)
        return self.output_layers(combined_embedding).squeeze(-1)
