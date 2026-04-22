import json
import tempfile
import urllib.parse
import webbrowser
from pathlib import Path
from datetime import datetime, timedelta

import gi
gi.require_version('Adw', '1')
gi.require_version('Gdk', '4.0')
gi.require_version('Gtk', '4.0')
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from gi.repository import Adw, Gdk, GLib, GObject, Gtk

from aichart import AIChart
from data_sources import fetch_tiingo_daily_prices, load_dip_labels, load_time_series_file, load_training_samples

APP_NAME = "Cronse"
APP_TAGLINE = "Stock Dip Predictor"
APP_TITLE = f"{APP_NAME} {APP_TAGLINE}"
APP_ID = "com.colin.cronse"
APP_ICON_NAME = "cronse"

class AIChartWindow(Adw.ApplicationWindow):
    FALLBACK_CHART_WIDTH = 900
    FALLBACK_CHART_HEIGHT = 560
    RENDER_DPI = 220
    RENDER_SCALE_MULTIPLIER = 2

    def __init__(self, application):
        super().__init__(application=application, title=APP_TITLE)
        self.base_path = Path(__file__).parent
        self.model_path = self.base_path / "trained_model.pt"
        self.training_data_dir = self.base_path / "training_data"
        self.chart_points = []
        self.chart_error = ""
        self.chart_image_path = Path(tempfile.gettempdir()) / "cronse_preview.png"
        self.current_ticker = ""
        self.chart_render_info = None
        self.chart_hover_index = None
        self.ui_state_path = Path(GLib.get_user_config_dir()) / "cronse" / "ui_state.json"
        self.show_training_points = False
        self.ai_chart = AIChart()
        self.model_analysis = None
        self.dip_labels_by_ticker = load_dip_labels(self.base_path)
        self.training_samples = load_training_samples(self.base_path)
        self.available_tickers = self._available_training_tickers()
        window_state = self._load_ui_state()
        self.favourite_tickers = self._load_favourite_tickers(window_state)
        self._updating_favourite_button = False
        self._restoring_paned_position = True

        # Restore favourites filter toggle state
        self._favourites_filter_active = window_state.get("favourites_filter_active", False)

        self._install_app_icon()

        self._load_saved_model()
        width = window_state.get("window_width", 800)
        height = window_state.get("window_height", 400)
        self.set_default_size(width, height)
        self._install_css()

        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        header_bar = Adw.HeaderBar()
        header_bar.set_title_widget(self._build_title_widget())
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

        # left_label = Gtk.Label(label="Available Tickers:")
        # left_label.set_xalign(0)
        # left_box.append(left_label)

        add_ticker_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        left_box.append(add_ticker_row)

        self.new_ticker_entry = Gtk.Entry()
        self.new_ticker_entry.set_hexpand(True)
        self.new_ticker_entry.set_placeholder_text("Add ticker")
        self.new_ticker_entry.connect("activate", self.on_add_ticker)
        add_ticker_row.append(self.new_ticker_entry)

        self.add_ticker_button = Gtk.Button(label="Add")
        self.add_ticker_button.connect("clicked", self.on_add_ticker)
        add_ticker_row.append(self.add_ticker_button)

        ticker_scroll = Gtk.ScrolledWindow()
        ticker_scroll.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        ticker_scroll.set_hexpand(True)
        ticker_scroll.set_vexpand(True)
        left_box.append(ticker_scroll)

        self.ticker_list = Gtk.ListBox()
        self.ticker_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.ticker_list.connect("row-selected", self.on_ticker_selected)
        ticker_scroll.set_child(self.ticker_list)

        # Add favourites filter toggle
        self.favourites_filter_toggle = Gtk.ToggleButton(label="Favourites only")
        self.favourites_filter_toggle.set_active(self._favourites_filter_active)
        self.favourites_filter_toggle.set_tooltip_text("Show only favourite tickers")
        self.favourites_filter_toggle.connect("toggled", self.on_favourites_filter_toggled)
        left_box.append(self.favourites_filter_toggle)

        self._refresh_ticker_list()
    def on_favourites_filter_toggled(self, button):
        # Save the toggle state
        state = self._load_ui_state()
        state["favourites_filter_active"] = button.get_active()
        self._save_ui_state(state)
        self._refresh_ticker_list()

    def _refresh_ticker_list(self, select_ticker=None):
        """Rebuilds the ticker list, applying the favourites filter if enabled."""
        row = self.ticker_list.get_first_child()
        while row is not None:
            next_row = row.get_next_sibling()
            self.ticker_list.remove(row)
            row = next_row

        show_favourites_only = getattr(self, "favourites_filter_toggle", None) and self.favourites_filter_toggle.get_active()
        if show_favourites_only:
            tickers = [t for t in self.available_tickers if t in self.favourite_tickers]
        else:
            tickers = list(self.available_tickers)

        for ticker in tickers:
            self._append_ticker_row(ticker)

        if select_ticker:
            self._select_ticker_in_list(select_ticker)

        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        right_box.set_margin_top(10)
        right_box.set_margin_bottom(10)
        right_box.set_margin_start(10)
        right_box.set_margin_end(10)
        right_box.add_css_class("card")
        self.paned.set_end_child(right_box)

        # right_label = Gtk.Label(label="Chart and Results:")
        # right_label.set_xalign(0)
        # right_box.append(right_label)

        self.chart_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.chart_container.set_hexpand(True)
        self.chart_container.set_vexpand(True)
        self.chart_container.set_valign(Gtk.Align.FILL)
        self.chart_container.add_css_class("chart-container")
        right_box.append(self.chart_container)

        chart_controls = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.chart_container.append(chart_controls)

        self.add_point_button = Gtk.ToggleButton(label="Add Point")
        self.add_point_button.connect("toggled", self.on_add_point_toggled)
        chart_controls.append(self.add_point_button)

        self.training_points_toggle = Gtk.ToggleButton(label="Training Points")
        self.training_points_toggle.set_active(False)
        self.training_points_toggle.connect("toggled", self.on_training_points_toggled)
        chart_controls.append(self.training_points_toggle)

        self.share_to_x_button = Gtk.Button(label="Add to X")
        self.share_to_x_button.connect("clicked", self.on_add_to_x_clicked)
        chart_controls.append(self.share_to_x_button)

        self.favourite_button = Gtk.ToggleButton()
        self.favourite_button.set_icon_name("non-starred-symbolic")
        self.favourite_button.set_tooltip_text("Add to favourites")
        self.favourite_button.connect("toggled", self.on_favourite_toggled)
        chart_controls.append(self.favourite_button)

        self.chart_hint_label = Gtk.Label(label="Hover the line for date/value. Right click a label to remove it.")
        self.chart_hint_label.set_xalign(0)
        self.chart_hint_label.set_hexpand(True)
        chart_controls.append(self.chart_hint_label)

        self.chart_overlay = Gtk.Overlay()
        self.chart_overlay.set_hexpand(True)
        self.chart_overlay.set_vexpand(True)
        self.chart_container.append(self.chart_overlay)

        self.chart_picture = Gtk.Picture()
        self.chart_picture.set_hexpand(True)
        self.chart_picture.set_vexpand(False)
        self.chart_picture.set_valign(Gtk.Align.START)
        self.chart_picture.set_can_shrink(True)
        self.chart_picture.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.chart_picture.set_cursor_from_name("crosshair")
        chart_click_gesture = Gtk.GestureClick.new()
        chart_click_gesture.set_button(0)
        chart_click_gesture.connect("pressed", self.on_chart_clicked)
        self.chart_picture.add_controller(chart_click_gesture)
        chart_motion_controller = Gtk.EventControllerMotion.new()
        chart_motion_controller.connect("motion", self.on_chart_motion)
        chart_motion_controller.connect("leave", self.on_chart_leave)
        self.chart_picture.add_controller(chart_motion_controller)
        self.chart_overlay.set_child(self.chart_picture)

        self.hover_popover = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        self.hover_popover.add_css_class("chart-hover-popover")
        self.hover_popover.set_halign(Gtk.Align.START)
        self.hover_popover.set_valign(Gtk.Align.START)
        self.hover_popover.set_visible(False)
        self.hover_popover.set_can_target(False)

        self.hover_date_label = Gtk.Label(xalign=0)
        self.hover_date_label.add_css_class("chart-hover-date")
        self.hover_value_label = Gtk.Label(xalign=0)
        self.hover_value_label.add_css_class("chart-hover-value")
        self.hover_popover.append(self.hover_date_label)
        self.hover_popover.append(self.hover_value_label)
        self.chart_overlay.add_overlay(self.hover_popover)

        self.result_label = Gtk.Label(label="Select a ticker to load its data.")
        self.result_label.set_xalign(0)
        self.result_label.set_wrap(True)
        right_box.append(self.result_label)

        GLib.idle_add(self._restore_paned_position)
        GLib.idle_add(self._rerender_chart_after_layout)
        self.paned.connect("notify::position", self.on_paned_position_changed)
        self.connect("close-request", self.on_close_request)

        self._update_favourite_button()
        self._select_initial_ticker()

    def on_ticker_selected(self, list_box, row):
        del list_box
        if row is None:
            return

        ticker = self._ticker_from_row(row)
        if ticker:
            self._load_ticker_data(ticker)

    def on_add_ticker(self, widget):
        del widget
        ticker = self.new_ticker_entry.get_text().strip().upper()
        if not ticker:
            self.result_label.set_text("Enter a ticker symbol to add.")
            return

        self._fetch_and_add_ticker(ticker)

    def on_add_point_toggled(self, button):
        if button.get_active():
            self.result_label.set_text(f"Add Point enabled for {self.current_ticker}. Click the chart to add a dip label.")
        elif self.current_ticker:
            self.result_label.set_text(
                f"Loaded {self.current_ticker}. Hover the line for date/value. Right click a labeled date to remove it."
            )

    def on_chart_clicked(self, gesture, _n_press, x, y):
        if not self.chart_points or not self.current_ticker or self.chart_render_info is None:
            return

        image_coords = self._picture_coords_to_chart_image(x, y)
        if image_coords is None:
            return

        image_x, image_y = image_coords
        point_index = self._nearest_chart_point_index(image_x, image_y)
        if point_index is None:
            return

        button = gesture.get_current_button()
        point_label = self.chart_points[point_index][0][:10]
        if button == Gdk.BUTTON_SECONDARY:
            updated = self._remove_dip_label(point_label)
            if updated:
                status_message = f"Removed dip label {point_label} for {self.current_ticker}."
        elif self.add_point_button.get_active():
            updated = self._add_dip_label(point_label)
            if updated:
                status_message = f"Added dip label {point_label} for {self.current_ticker}."
            else:
                self.result_label.set_text(f"{point_label} is already labeled for {self.current_ticker}.")
            self.add_point_button.set_active(False)
        else:
            updated = False

        if updated:
            current_ticker = self.current_ticker
            self._load_ticker_data(current_ticker)
            self.result_label.set_text(status_message)

    def on_chart_motion(self, _controller, x, y):
        if not self.chart_points or self.chart_render_info is None:
            self._hide_hover_popover()
            return

        image_coords = self._picture_coords_to_chart_image(x, y)
        if image_coords is None:
            self._hide_hover_popover()
            return

        image_x, image_y = image_coords
        hover_index = self._hover_chart_point_index(image_x, image_y)
        if hover_index is None:
            self._hide_hover_popover()
            return

        self.chart_hover_index = hover_index
        point_label, point_value = self.chart_points[hover_index]
        self.hover_date_label.set_text(point_label[:10])
        self.hover_value_label.set_text(f"${point_value:.2f}")
        self.hover_popover.set_margin_start(max(0, min(int(x + 14), max(self.chart_picture.get_allocated_width() - 160, 0))))
        self.hover_popover.set_margin_top(max(0, min(int(y - 8), max(self.chart_picture.get_allocated_height() - 60, 0))))
        self.hover_popover.set_visible(True)

    def on_chart_leave(self, _controller):
        self._hide_hover_popover()

    def on_add_to_x_clicked(self, _button):
        if not self.current_ticker or len(self.chart_points) < 2:
            self.result_label.set_text("Load a ticker chart before sending it to X.")
            return

        self._render_chart_image()

        try:
            self._copy_chart_image_to_clipboard()
        except GLib.Error as exc:
            self.result_label.set_text(f"Unable to copy chart image for X: {exc.message}")
            return
        except (OSError, ValueError) as exc:
            self.result_label.set_text(f"Unable to copy chart image for X: {exc}")
            return

        share_url = self._x_compose_url()
        try:
            opened = webbrowser.open(share_url, new=2)
        except webbrowser.Error as exc:
            self.result_label.set_text(f"Copied chart image, but could not open X: {exc}")
            return

        if not opened:
            self.result_label.set_text("Copied chart image, but could not open X in your default browser.")
            return

        self.result_label.set_text(
            "Copied the chart image to your clipboard and opened X. Paste in the composer to attach it."
        )

    def on_favourite_toggled(self, button):
        if self._updating_favourite_button:
            return

        if not self.current_ticker:
            self.result_label.set_text("Load a ticker before adding it to favourites.")
            self._update_favourite_button()
            return

        ticker = self.current_ticker
        if button.get_active():
            self.favourite_tickers.add(ticker)
            self.result_label.set_text(f"Added {ticker} to favourites.")
        else:
            self.favourite_tickers.discard(ticker)
            self.result_label.set_text(f"Removed {ticker} from favourites.")

        self._persist_favourites()
        self._refresh_ticker_icons()
        self._update_favourite_button()

    def on_training_points_toggled(self, button):
        self.show_training_points = button.get_active()
        if self.chart_points or self.chart_error:
            self._render_chart_image()

    def on_paned_position_changed(self, paned, _property_spec):
        if self._restoring_paned_position:
            return

        position = paned.get_position()
        if position > 0:
            state = self._load_ui_state()
            state["paned_position"] = position
            self._save_ui_state(state)

        if self.chart_points or self.chart_error:
            self._render_chart_image()

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

    def update_chart_from_rows(self, rows):
        try:
            self.chart_points = self._parse_chart_rows(rows)
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
        chart_width, chart_height = self._chart_pixel_size()
        figure = self._build_chart_figure(chart_width, chart_height)
        figure.savefig(
            self.chart_image_path,
            facecolor=figure.get_facecolor(),
            format="png",
            dpi=self.RENDER_DPI,
        )
        self.chart_picture.set_filename(str(self.chart_image_path))

    def _build_chart_figure(self, chart_width, chart_height):
        figure = Figure(
            figsize=(chart_width / self.RENDER_DPI, chart_height / self.RENDER_DPI),
            dpi=self.RENDER_DPI,
        )
        axis = figure.subplots()
        self.chart_render_info = None
        figure.patch.set_facecolor("#07111c")
        axis.set_facecolor("#091525")

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
            min_value = min(values)
            max_value = max(values)
            value_range = max(max_value - min_value, 1e-6)
            baseline = min_value - (value_range * 0.14)

            axis.set_axisbelow(True)
            axis.fill_between(x_values, values, baseline, color="#22d3ee", alpha=0.1, zorder=1)
            axis.plot(x_values, values, color="#22d3ee", linewidth=9.0, alpha=0.06, solid_capstyle="round", zorder=2)
            axis.plot(x_values, values, color="#a855f7", linewidth=4.2, alpha=0.12, solid_capstyle="round", zorder=3)
            axis.plot(x_values, values, color="#7dd3fc", linewidth=1.8, solid_capstyle="round", zorder=4)
            axis.scatter([x_values[-1]], [values[-1]], s=44, color="#67e8f9", edgecolors="#ecfeff", linewidths=0.8, zorder=7)

            axis.tick_params(axis="x", colors="#c7d2fe", labelrotation=45, labelsize=8)
            axis.tick_params(axis="y", colors="#dbeafe", labelsize=8)
            axis.spines["bottom"].set_color("#28445f")
            axis.spines["left"].set_color("#28445f")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.grid(color="#38bdf8", linestyle=":", linewidth=0.7, alpha=0.18)
            axis.set_title(self.current_ticker or "Daily Close", color="#f8fafc", fontsize=13, fontweight="bold", pad=10)
            axis.set_ylim(baseline, max_value + (value_range * 0.1))

            labeled_indices = self._labeled_dip_indices(labels)
            if self.show_training_points and labeled_indices:
                labeled_values = [values[index] for index in labeled_indices]
                axis.scatter(
                    labeled_indices,
                    labeled_values,
                    s=36,
                    color="#a855f7",
                    edgecolors="#f3e8ff",
                    linewidths=0.8,
                    zorder=4,
                )
                axis.scatter(
                    labeled_indices,
                    labeled_values,
                    s=150,
                    color="#a855f7",
                    alpha=0.08,
                    linewidths=0,
                    zorder=4,
                )
                for labeled_index, labeled_value in zip(labeled_indices, labeled_values):
                    axis.annotate(
                        labels[labeled_index],
                        xy=(labeled_index, labeled_value),
                        xytext=(0, -16),
                        textcoords="offset points",
                        ha="center",
                        color="#f3e8ff",
                        fontsize=8,
                        bbox={"boxstyle": "round,pad=0.2", "fc": "#1f1230", "ec": "#a855f7", "alpha": 0.95},
                        zorder=5,
                    )

            if self.model_analysis and self.model_analysis["dip_indices"]:
                dip_indices = self.model_analysis["dip_indices"]
                dip_values = [values[index] for index in dip_indices]
                axis.scatter(
                    dip_indices,
                    dip_values,
                    s=220,
                    color="#22c55e",
                    alpha=0.08,
                    linewidths=0,
                    zorder=5,
                )
                axis.scatter(
                    dip_indices,
                    dip_values,
                    s=95,
                    facecolors="none",
                    edgecolors="#22c55e",
                    linewidths=1.8,
                    zorder=5,
                )
                axis.scatter(
                    dip_indices,
                    dip_values,
                    s=28,
                    color="#22c55e",
                    zorder=6,
                )
                for dip_index, dip_value in zip(dip_indices, dip_values):
                    axis.annotate(
                        labels[dip_index],
                        xy=(dip_index, dip_value),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha="center",
                        color="#86efac",
                        fontsize=8,
                        bbox={"boxstyle": "round,pad=0.2", "fc": "#0f1f14", "ec": "#22c55e", "alpha": 0.95},
                        zorder=7,
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

            # axis.text(
            #     0.015,
            #     0.98,
            #     "Slight Neon View",
            #     transform=axis.transAxes,
            #     va="top",
            #     ha="left",
            #     color="#67e8f9",
            #     fontsize=8,
            #     alpha=0.85,
            #     bbox={"boxstyle": "round,pad=0.22", "fc": "#08131f", "ec": "#164e63", "alpha": 0.92},
            # )

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
        canvas = FigureCanvasAgg(figure)
        canvas.draw()

        if not self.chart_error and len(self.chart_points) >= 2:
            axis_bounds = axis.bbox.bounds
            x_pixels = [axis.transData.transform((index, value))[0] for index, value in enumerate(values)]
            y_pixels = [chart_height - axis.transData.transform((index, value))[1] for index, value in enumerate(values)]
            self.chart_render_info = {
                "render_width": chart_width,
                "render_height": chart_height,
                "plot_left": axis_bounds[0],
                "plot_right": axis_bounds[0] + axis_bounds[2],
                "plot_top": chart_height - (axis_bounds[1] + axis_bounds[3]),
                "plot_bottom": chart_height - axis_bounds[1],
                "x_pixels": x_pixels,
                "y_pixels": y_pixels,
            }

        return figure

    def _chart_pixel_size(self):
        scale_factor = max(self.get_scale_factor(), 1)
        width = self.chart_overlay.get_allocated_width()
        height = self.chart_overlay.get_allocated_height()

        if width <= 1:
            width = max(self.get_width() // 2, self.FALLBACK_CHART_WIDTH)
        if height <= 1:
            height = max(self.get_height() - 140, self.FALLBACK_CHART_HEIGHT)

        render_scale = scale_factor * self.RENDER_SCALE_MULTIPLIER
        return max(width * render_scale, self.FALLBACK_CHART_WIDTH), max(
            height * render_scale,
            self.FALLBACK_CHART_HEIGHT,
        )

    def _rerender_chart_after_layout(self):
        if self.chart_points or self.chart_error:
            self._render_chart_image()
        return GLib.SOURCE_REMOVE

    def _parse_chart_rows(self, rows):
        if not rows:
            return []

        points = []
        for index, item in enumerate(rows, start=1):
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

    def _train_model_from_chart_points(self):
        closes = [point[1] for point in self.chart_points]
        if self.ai_chart.model is None and self.training_samples:
            self.ai_chart.fit(self.training_samples)
            self.ai_chart.save_model(self.model_path)

        analysis = self.ai_chart.predict(closes)
        self.model_analysis = analysis

        if analysis["samples"] == 0:
            self.result_label.set_text("Need at least 5 closing prices to train dip detection.")
        else:
            training_file_count = len(self.training_samples)
            projected_dip = analysis.get("projected_dip")
            if projected_dip and self.chart_points:
                projected_date = self._projected_date(self.chart_points[-1][0], projected_dip["bars_ahead"])
                projection_text = f" Next target: {projected_date} at ${projected_dip['target_price']:.2f}."
            else:
                projection_text = ""

            training_summary = self.ai_chart.training_summary or {}
            model_detail = ""
            if training_summary:
                model_detail = (
                    f" Trained torch model on {training_summary.get('series_count', training_file_count)} files"
                    f" with explicit dip labels on {training_summary.get('explicitly_labeled_series_count', 0)} files."
                )

            self.result_label.set_text(
                f"{self.current_ticker or 'Current symbol'}: highlighting {len(analysis['dip_indices'])} dip bottoms.{model_detail}{projection_text}"
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

    def _labeled_dip_indices(self, labels):
        ticker = self.current_ticker
        if not ticker:
            return []

        labeled_dates = set(self.dip_labels_by_ticker.get(ticker, []))
        if not labeled_dates:
            return []

        return [index for index, label in enumerate(labels) if str(label)[:10] in labeled_dates]

    def _available_training_tickers(self):
        tickers = []
        for data_path in sorted(self.training_data_dir.glob("*.json")):
            ticker = data_path.stem.upper()
            if ticker:
                tickers.append(ticker)
        return tickers

    def _load_favourite_tickers(self, state):
        saved_favourites = state.get("favourite_tickers", state.get("favorite_tickers", []))
        if not isinstance(saved_favourites, list):
            return set()
        return {
            str(ticker).strip().upper()
            for ticker in saved_favourites
            if str(ticker).strip()
        }

    def _persist_favourites(self):
        state = self._load_ui_state()
        state["favourite_tickers"] = sorted(self.favourite_tickers)
        self._save_ui_state(state)

    def _ticker_icon_name(self, ticker):
        return "starred-symbolic" if ticker in self.favourite_tickers else "text-x-generic-symbolic"

    def _ticker_from_row(self, row):
        return str(getattr(row, "ticker", "")).strip().upper()

    def _refresh_ticker_icons(self):
        row = self.ticker_list.get_first_child()
        while row is not None:
            self._sync_ticker_row_icon(row)
            row = row.get_next_sibling()

    def _sync_ticker_row_icon(self, row):
        icon = getattr(row, "ticker_icon", None)
        ticker = self._ticker_from_row(row)
        if icon is None or not ticker:
            return

        icon.set_from_icon_name(self._ticker_icon_name(ticker))
        if ticker in self.favourite_tickers:
            icon.add_css_class("favourite-ticker-icon")
        else:
            icon.remove_css_class("favourite-ticker-icon")

    def _update_favourite_button(self):
        if not hasattr(self, "favourite_button"):
            return

        is_favourite = bool(self.current_ticker and self.current_ticker in self.favourite_tickers)
        self._updating_favourite_button = True
        self.favourite_button.set_sensitive(bool(self.current_ticker))
        self.favourite_button.set_active(is_favourite)
        self.favourite_button.set_icon_name("starred-symbolic" if is_favourite else "non-starred-symbolic")
        self.favourite_button.set_tooltip_text(
            "Remove from favourites" if is_favourite else "Add to favourites"
        )
        self._updating_favourite_button = False

    def _append_ticker_row(self, ticker):
        row = Gtk.ListBoxRow()
        row.set_selectable(True)
        row.set_activatable(True)
        row.ticker = ticker

        row_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row_box.set_margin_top(4)
        row_box.set_margin_bottom(4)
        row_box.set_margin_start(4)
        row_box.set_margin_end(4)

        row.ticker_icon = Gtk.Image.new_from_icon_name(self._ticker_icon_name(ticker))
        row.ticker_icon.set_pixel_size(16)
        self._sync_ticker_row_icon(row)
        row_box.append(row.ticker_icon)

        label = Gtk.Label(label=ticker, xalign=0)
        label.set_hexpand(True)
        row_box.append(label)

        row.set_child(row_box)
        self.ticker_list.append(row)
        return row

    def _select_initial_ticker(self):
        if not self.available_tickers:
            self.chart_error = "No ticker files found in training_data."
            self.model_analysis = None
            self.result_label.set_text(self.chart_error)
            self._update_favourite_button()
            self._render_chart_image()
            return

        first_row = self.ticker_list.get_row_at_index(0)
        if first_row is not None:
            self.ticker_list.select_row(first_row)

    def _load_ticker_data(self, ticker):
        data_path = self.training_data_dir / f"{ticker.lower()}.json"

        try:
            points = load_time_series_file(data_path)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            self.current_ticker = ticker
            self.chart_points = []
            self.chart_error = f"Unable to load {ticker}: {exc}"
            self.model_analysis = None
            self._update_favourite_button()
            self.result_label.set_text(self.chart_error)
            self._render_chart_image()
            return

        rows = [
            {
                "reported_at": label,
                "close": close,
            }
            for label, close in points
        ]
        self.current_ticker = ticker
        self.update_chart_from_rows(rows)
        self._update_favourite_button()
        self.result_label.set_text(f"Loaded {ticker}. Hover the line for date/value. Right click a labeled date to remove it.")

    def _fetch_and_add_ticker(self, ticker):
        output_path = self.training_data_dir / f"{ticker.lower()}.json"

        try:
            rows = fetch_tiingo_daily_prices(
                ticker,
                settings_path=self.base_path / "settings.json",
            )
        except ValueError as exc:
            self.result_label.set_text(f"Unable to fetch {ticker}: {exc}")
            return

        if not rows:
            self.result_label.set_text(f"No data returned for {ticker}.")
            return

        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        self.training_samples = load_training_samples(self.base_path)

        if ticker not in self.available_tickers:
            self.available_tickers.append(ticker)
            self.available_tickers.sort()
            self._refresh_ticker_list(select_ticker=ticker)
        else:
            self._select_ticker_in_list(ticker)

        self.new_ticker_entry.set_text("")
        self._load_ticker_data(ticker)
        self.result_label.set_text(f"Fetched and loaded {ticker} into training_data/{output_path.name}.")

    # _rebuild_ticker_list is replaced by _refresh_ticker_list

    def _select_ticker_in_list(self, ticker):
        row = self.ticker_list.get_first_child()
        while row is not None:
            row_ticker = self._ticker_from_row(row)
            if row_ticker == ticker:
                self.ticker_list.select_row(row)
                return
            row = row.get_next_sibling()

    def _picture_coords_to_chart_image(self, x, y):
        if self.chart_render_info is None:
            return None

        picture_width = self.chart_picture.get_allocated_width()
        picture_height = self.chart_picture.get_allocated_height()
        render_width = self.chart_render_info["render_width"]
        render_height = self.chart_render_info["render_height"]
        if picture_width <= 0 or picture_height <= 0 or render_width <= 0 or render_height <= 0:
            return None

        scale = min(picture_width / render_width, picture_height / render_height)
        displayed_width = render_width * scale
        displayed_height = render_height * scale
        offset_x = (picture_width - displayed_width) / 2
        offset_y = (picture_height - displayed_height) / 2

        if x < offset_x or x > offset_x + displayed_width or y < offset_y or y > offset_y + displayed_height:
            return None

        image_x = (x - offset_x) / scale
        image_y = (y - offset_y) / scale
        return image_x, image_y

    def _nearest_chart_point_index(self, image_x, image_y):
        if self.chart_render_info is None:
            return None

        plot_left = self.chart_render_info["plot_left"]
        plot_right = self.chart_render_info["plot_right"]
        plot_top = self.chart_render_info["plot_top"]
        plot_bottom = self.chart_render_info["plot_bottom"]
        if image_x < plot_left or image_x > plot_right or image_y < max(0, plot_top - 40) or image_y > plot_bottom + 60:
            return None

        x_pixels = self.chart_render_info["x_pixels"]
        if not x_pixels:
            return None

        closest_index = min(range(len(x_pixels)), key=lambda index: abs(x_pixels[index] - image_x))
        return closest_index

    def _hover_chart_point_index(self, image_x, image_y):
        if self.chart_render_info is None:
            return None

        x_pixels = self.chart_render_info["x_pixels"]
        y_pixels = self.chart_render_info.get("y_pixels", [])
        if len(x_pixels) < 2 or len(y_pixels) < 2:
            return None

        if image_x < x_pixels[0] or image_x > x_pixels[-1]:
            return None

        right_index = next((index for index, value in enumerate(x_pixels) if value >= image_x), len(x_pixels) - 1)
        left_index = max(0, right_index - 1)
        if right_index == left_index:
            line_y = y_pixels[left_index]
        else:
            x0 = x_pixels[left_index]
            x1 = x_pixels[right_index]
            if x1 == x0:
                line_y = y_pixels[left_index]
            else:
                ratio = (image_x - x0) / (x1 - x0)
                line_y = y_pixels[left_index] + ratio * (y_pixels[right_index] - y_pixels[left_index])

        if abs(image_y - line_y) > 18:
            return None

        if abs(image_x - x_pixels[left_index]) <= abs(image_x - x_pixels[right_index]):
            return left_index
        return right_index

    def _hide_hover_popover(self):
        self.chart_hover_index = None
        self.hover_popover.set_visible(False)

    def _add_dip_label(self, label_date):
        ticker = self.current_ticker
        existing_dates = set(self.dip_labels_by_ticker.get(ticker, []))
        if label_date in existing_dates:
            return False

        existing_dates.add(label_date)
        self._write_dip_labels(ticker, sorted(existing_dates))
        return True

    def _remove_dip_label(self, label_date):
        ticker = self.current_ticker
        existing_dates = set(self.dip_labels_by_ticker.get(ticker, []))
        if label_date not in existing_dates:
            return False

        existing_dates.remove(label_date)
        self._write_dip_labels(ticker, sorted(existing_dates))
        return True

    def _write_dip_labels(self, ticker, label_dates):
        labels_path = self.base_path / "dip_labels.json"
        try:
            payload = json.loads(labels_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {"tickers": {}}

        tickers_payload = payload.get("tickers")
        if not isinstance(tickers_payload, dict):
            tickers_payload = {}
            payload["tickers"] = tickers_payload

        tickers_payload[ticker] = {"sample_dips": label_dates}
        labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.dip_labels_by_ticker[ticker] = label_dates
        self.training_samples = load_training_samples(self.base_path)

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
        self.ui_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def _copy_chart_image_to_clipboard(self):
        if not self.chart_image_path.exists():
            raise OSError(f"Missing chart image at {self.chart_image_path}")

        display = Gdk.Display.get_default()
        if display is None:
            raise ValueError("No graphical display is available.")

        texture = Gdk.Texture.new_from_filename(str(self.chart_image_path))
        texture_value = GObject.Value()
        texture_value.init(Gdk.Texture)
        texture_value.set_object(texture)

        provider = Gdk.ContentProvider.new_for_value(texture_value)
        clipboard = display.get_clipboard()
        if not clipboard.set_content(provider):
            raise ValueError("The clipboard rejected the chart image.")

    def _x_compose_url(self):
        query = urllib.parse.urlencode({"text": self._x_share_text()})
        return f"https://twitter.com/intent/tweet?{query}"

    def _x_share_text(self):
        if not self.current_ticker:
            return "Chart from Cronse"

        projected_dip = self.model_analysis.get("projected_dip") if self.model_analysis else None
        if projected_dip:
            projected_date = self._projected_date(self.chart_points[-1][0], projected_dip["bars_ahead"])
            return (
                f"{self.current_ticker} chart AI projected buying point: "
                f"{projected_date} near ${projected_dip['target_price']:.2f}."
            )

        current_close = self.chart_points[-1][1] if self.chart_points else None
        if current_close is None:
            return f"{self.current_ticker} chart from Cronse"
        return f"{self.current_ticker} chart current close: ${current_close:.2f}."

    def _install_css(self):
        css_provider = Gtk.CssProvider()
        css_provider.load_from_data(
            b".card { background: linear-gradient(180deg, rgba(9, 18, 31, 0.94), rgba(10, 14, 24, 0.92)); border: 1px solid rgba(74, 222, 128, 0.08); box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(125, 211, 252, 0.06); border-radius: 16px; }"
            b" .chart-container { background: linear-gradient(180deg, rgba(8, 17, 28, 0.98), rgba(10, 23, 38, 0.95)); border: 1px solid rgba(34, 211, 238, 0.18); border-radius: 18px; padding: 12px; box-shadow: inset 0 1px 0 rgba(103, 232, 249, 0.08), 0 18px 50px rgba(8, 15, 24, 0.45); }"
            b" .chart-hover-popover { background-color: rgba(9, 16, 28, 0.98); border: 1px solid rgba(103, 232, 249, 0.7); border-radius: 10px; padding: 7px 9px; box-shadow: 0 10px 30px rgba(34, 211, 238, 0.15); }"
            b" .chart-hover-date { color: #f5f3ff; font-weight: 700; }"
            b" .chart-hover-value { color: #67e8f9; }"
            b" .favourite-ticker-icon { color: #22c55e; }"
        )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            css_provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _install_app_icon(self):
        display = Gdk.Display.get_default()
        if display is None:
            return

        icon_root = self.base_path / "icons"
        icon_theme = Gtk.IconTheme.get_for_display(display)
        icon_theme.add_search_path(str(icon_root))
        Gtk.Window.set_default_icon_name(APP_ICON_NAME)
        self.set_icon_name(APP_ICON_NAME)

    def _build_title_widget(self):
        title_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)

        icon = Gtk.Image.new_from_icon_name(APP_ICON_NAME)
        icon.set_pixel_size(24)
        title_box.append(icon)

        label = Gtk.Label(label=APP_TITLE)
        label.add_css_class("title-3")
        title_box.append(label)

        return title_box


class AIChartApp(Adw.Application):
    def __init__(self):
        super().__init__(application_id=APP_ID)
        style_manager = Adw.StyleManager.get_default()
        style_manager.set_color_scheme(Adw.ColorScheme.DEFAULT)

    def do_activate(self):
        window = self.props.active_window
        if window is None:
            window = AIChartWindow(self)
        window.present()
