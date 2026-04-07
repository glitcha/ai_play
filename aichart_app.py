import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

import gi
gi.require_version('Adw', '1')
gi.require_version('Gtk', '4.0')
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gi.repository import Adw, Gdk, GLib, Gtk

from aichart import AIChart
from data_sources import load_training_series

class AIChartWindow(Adw.ApplicationWindow):
    def __init__(self, application):
        super().__init__(application=application, title="AIChart Stock Dip Predictor")
        self.base_path = Path(__file__).parent
        self.model_path = self.base_path / "trained_model.pt"
        self.chart_points = []
        self.chart_error = ""
        self.chart_image_path = Path(tempfile.gettempdir()) / "aichart_preview.png"
        self.ui_state_path = Path(GLib.get_user_config_dir()) / "aichart" / "ui_state.json"
        self.ai_chart = AIChart()
        self.model_analysis = None
        self.training_series = load_training_series(self.base_path)
        self._restoring_paned_position = True

        self._load_saved_model()

        window_state = self._load_ui_state()
        width = window_state.get("window_width", 800)
        height = window_state.get("window_height", 400)
        self.set_default_size(width, height)
        self._install_css()

        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        header_bar = Adw.HeaderBar()
        header_bar.set_title_widget(Gtk.Label(label="AIChart Stock Dip Predictor"))
        toolbar_view.add_top_bar(header_bar)

        self.paned = Gtk.Paned.new(Gtk.Orientation.HORIZONTAL)
        toolbar_view.set_content(self.paned)

        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        left_box.set_margin_top(10)
        left_box.set_margin_bottom(10)
        left_box.set_margin_start(10)
        left_box.set_margin_end(10)
        left_box.add_css_class("card")
        self.paned.set_start_child(left_box)

        left_label = Gtk.Label(label="Paste JSON Data:")
        left_label.set_xalign(0)
        left_box.append(left_label)

        self.textview = Gtk.TextView()
        self.textview.set_wrap_mode(Gtk.WrapMode.WORD)
        self.text_buffer = self.textview.get_buffer()
        text_scroll = Gtk.ScrolledWindow()
        text_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        text_scroll.set_child(self.textview)
        text_scroll.set_hexpand(True)
        text_scroll.set_vexpand(True)
        left_box.append(text_scroll)

        self._load_initial_data()
        self.text_buffer.connect("changed", self.on_text_changed)

        self.run_button = Gtk.Button(label="Run Model")
        self.run_button.connect("clicked", self.on_run_model)
        left_box.append(self.run_button)

        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        right_box.set_margin_top(10)
        right_box.set_margin_bottom(10)
        right_box.set_margin_start(10)
        right_box.set_margin_end(10)
        right_box.add_css_class("card")
        self.paned.set_end_child(right_box)

        right_label = Gtk.Label(label="Chart and Results:")
        right_label.set_xalign(0)
        right_box.append(right_label)

        self.chart_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.chart_container.set_hexpand(True)
        self.chart_container.set_vexpand(True)
        self.chart_container.set_valign(Gtk.Align.FILL)
        self.chart_container.add_css_class("chart-container")
        right_box.append(self.chart_container)

        self.chart_picture = Gtk.Picture()
        self.chart_picture.set_hexpand(True)
        self.chart_picture.set_vexpand(False)
        self.chart_picture.set_valign(Gtk.Align.START)
        self.chart_picture.set_can_shrink(True)
        self.chart_container.append(self.chart_picture)

        self.result_label = Gtk.Label(label="Chart updates when JSON changes.")
        self.result_label.set_xalign(0)
        self.result_label.set_wrap(True)
        right_box.append(self.result_label)

        GLib.idle_add(self._restore_paned_position)
        self.paned.connect("notify::position", self.on_paned_position_changed)
        self.connect("close-request", self.on_close_request)

        self.update_chart_from_buffer()

    def on_run_model(self, widget):
        self._load_initial_data()

    def on_text_changed(self, buffer):
        self.update_chart_from_buffer()

    def on_paned_position_changed(self, paned, _property_spec):
        if self._restoring_paned_position:
            return

        position = paned.get_position()
        if position > 0:
            state = self._load_ui_state()
            state["paned_position"] = position
            self._save_ui_state(state)

    def on_close_request(self, window):
        state = self._load_ui_state()
        width = self.get_width()
        height = self.get_height()
        if width > 0 and height > 0:
            state["window_width"] = width
            state["window_height"] = height
        position = self.paned.get_position()
        if position > 0:
            state["paned_position"] = position
        self._save_ui_state(state)
        return False

    def update_chart_from_buffer(self):
        json_text = self._get_buffer_text()

        try:
            self.chart_points = self._parse_chart_points(json_text)
            self.chart_error = ""
            if self.chart_points:
                self._train_model_from_chart_points()
            else:
                self.model_analysis = None
                self.result_label.set_text("No data to plot.")
        except ValueError as exc:
            self.chart_points = []
            self.chart_error = str(exc)
            self.model_analysis = None
            self.result_label.set_text(self.chart_error)

        self._render_chart_image()

    def _render_chart_image(self):
        figure, axis = plt.subplots(figsize=(6.4, 4.2), dpi=100)
        figure.patch.set_facecolor("#1e1e1e")
        axis.set_facecolor("#1e1e1e")

        if self.chart_error:
            axis.text(0.5, 0.5, self.chart_error, color="#d1d5db", ha="center", va="center", wrap=True)
            axis.set_axis_off()
        elif len(self.chart_points) < 2:
            axis.text(
                0.5,
                0.5,
                "Paste JSON with at least two close values.",
                color="#d1d5db",
                ha="center",
                va="center",
                wrap=True,
            )
            axis.set_axis_off()
        else:
            labels = [point[0] for point in self.chart_points]
            values = [point[1] for point in self.chart_points]
            x_values = list(range(len(labels)))
            axis.plot(x_values, values, color="#8b5cf6", linewidth=1.1)
            axis.tick_params(axis="x", colors="#cfd8e3", labelrotation=45)
            axis.tick_params(axis="y", colors="#cfd8e3")
            axis.spines["bottom"].set_color("#5c6570")
            axis.spines["left"].set_color("#5c6570")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(color="#343a40", linestyle="--", linewidth=0.6, alpha=0.7)
            axis.set_title("Daily Close", color="#f3f4f6")

            if self.model_analysis and self.model_analysis["dip_indices"]:
                dip_indices = self.model_analysis["dip_indices"]
                dip_values = [values[index] for index in dip_indices]
                axis.scatter(
                    dip_indices,
                    dip_values,
                    s=95,
                    facecolors="none",
                    edgecolors="#22c55e",
                    linewidths=1.8,
                    zorder=4,
                )
                axis.scatter(
                    dip_indices,
                    dip_values,
                    s=28,
                    color="#22c55e",
                    zorder=5,
                )

            projected_dip = self.model_analysis.get("projected_dip") if self.model_analysis else None
            if projected_dip:
                projected_x = x_values[-1] + projected_dip["bars_ahead"]
                projected_y = projected_dip["target_price"]
                projected_label = self._projected_label(labels[-1], projected_dip["bars_ahead"], projected_y)

                axis.plot(
                    [x_values[-1], projected_x],
                    [values[-1], projected_y],
                    color="#f59e0b",
                    linewidth=1.6,
                    linestyle="--",
                    zorder=3,
                )
                axis.scatter(
                    [projected_x],
                    [projected_y],
                    s=42,
                    color="#f59e0b",
                    zorder=6,
                )
                axis.annotate(
                    projected_label,
                    xy=(projected_x, projected_y),
                    xytext=(10, -14),
                    textcoords="offset points",
                    color="#fbbf24",
                    fontsize=9,
                    bbox={"boxstyle": "round,pad=0.25", "fc": "#2b2110", "ec": "#f59e0b", "alpha": 0.95},
                )
                axis.set_xlim(0, projected_x + 2)

            max_labels = 8
            if len(labels) > max_labels:
                step = max(1, len(labels) // max_labels)
                visible_positions = list(range(0, len(labels), step))
            else:
                visible_positions = list(range(len(labels)))

            visible_labels = [labels[index] for index in visible_positions]
            if projected_dip:
                projected_x = x_values[-1] + projected_dip["bars_ahead"]
                visible_positions.append(projected_x)
                visible_labels.append(self._projected_tick_label(labels[-1], projected_dip["bars_ahead"]))

            axis.set_xticks(visible_positions)
            axis.set_xticklabels(visible_labels, ha="right")

        figure.tight_layout()
        figure.savefig(self.chart_image_path, facecolor=figure.get_facecolor())
        plt.close(figure)
        self.chart_picture.set_filename(str(self.chart_image_path))

    def _parse_chart_points(self, json_text):
        if not json_text.strip():
            return []

        payload = json.loads(json_text)
        if not isinstance(payload, list):
            raise ValueError("JSON must be an array of daily records.")

        points = []
        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Row {index} must be an object.")
            if "close" not in item:
                raise ValueError(f"Row {index} is missing 'close'.")

            label = str(item.get("reported_at", index))
            try:
                close_value = float(item["close"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Row {index} has an invalid 'close' value.") from exc

            points.append((label, close_value))

        points.sort(key=lambda point: point[0])
        return points

    def _get_buffer_text(self):
        start_iter, end_iter = self.text_buffer.get_bounds()
        return self.text_buffer.get_text(start_iter, end_iter, True)

    def _train_model_from_chart_points(self):
        closes = [point[1] for point in self.chart_points]
        if self.ai_chart.model is None and self.training_series:
            training_closes = [[close for _, close in points] for _, points in self.training_series]
            self.ai_chart.fit(training_closes)
            self.ai_chart.save_model(self.model_path)

        analysis = self.ai_chart.predict(closes)
        self.model_analysis = analysis

        if analysis["samples"] == 0:
            self.result_label.set_text("Need at least 5 closing prices to train dip detection.")
        else:
            training_file_count = len(self.training_series)
            projected_dip = analysis.get("projected_dip")
            if projected_dip and self.chart_points:
                projected_date = self._projected_date(self.chart_points[-1][0], projected_dip["bars_ahead"])
                projection_text = f" Next target: {projected_date} at ${projected_dip['target_price']:.2f}."
            else:
                projection_text = ""

            training_summary = self.ai_chart.training_summary or {}
            model_detail = ""
            if training_summary:
                model_detail = f" Trained torch model on {training_summary.get('series_count', training_file_count)} files."

            self.result_label.set_text(
                f"Loaded learned model. Highlighting {len(analysis['dip_indices'])} dip bottoms.{model_detail}{projection_text}"
            )

    def _load_saved_model(self):
        try:
            self.ai_chart.load_model(self.model_path)
        except (OSError, ValueError, RuntimeError):
            self.ai_chart.model = None
            self.ai_chart.model_metadata = None
            self.ai_chart.training_summary = None
            self.ai_chart.feature_mean = None
            self.ai_chart.feature_std = None

    def _load_initial_data(self):
        json_text = self._read_data_file()
        self.text_buffer.set_text(json_text)

    def _projected_label(self, last_label, bars_ahead, target_price):
        projected_date = self._projected_date(last_label, bars_ahead)
        return f"Projected dip\n{projected_date}\n${target_price:.2f}"

    def _projected_tick_label(self, last_label, bars_ahead):
        return self._projected_date(last_label, bars_ahead)

    def _projected_date(self, last_label, bars_ahead):
        try:
            current_date = datetime.strptime(last_label, "%Y-%m-%d").date()
        except ValueError:
            return f"+{bars_ahead} bars"

        projected_date = current_date
        remaining_bars = bars_ahead
        while remaining_bars > 0:
            projected_date += timedelta(days=1)
            if projected_date.weekday() < 5:
                remaining_bars -= 1

        return projected_date.isoformat()

    def _read_data_file(self):
        data_path = Path(__file__).with_name("data.json")

        try:
            return data_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    def _restore_paned_position(self):
        state = self._load_ui_state()
        position = state.get("paned_position")

        if isinstance(position, int) and position > 0:
            self.paned.set_position(position)

        self._restoring_paned_position = False
        return GLib.SOURCE_REMOVE

    def _load_ui_state(self):
        try:
            return json.loads(self.ui_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_ui_state(self, state):
        self.ui_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.ui_state_path.write_text(json.dumps(state), encoding="utf-8")

    def _install_css(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            b".chart-container { background-color: rgba(20, 20, 24, 0.92); border-radius: 12px; padding: 10px; }"
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )


class AIChartApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id="com.example.aichart")
        style_manager = Adw.StyleManager.get_default()
        style_manager.set_color_scheme(Adw.ColorScheme.DEFAULT)

    def do_activate(self):
        window = self.props.active_window
        if window is None:
            window = AIChartWindow(self)
        window.present()
