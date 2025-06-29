import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
import threading
import time
import logging
from collections import deque
from dwell_timer import DwellTimer

class GazeTracker:
    def __init__(self, root):
        self.root = root
        self.gaze_tracking = False
        self.gaze_thread = None
        self.smooth_x = 0
        self.smooth_y = 0
        self.smoothing_factor = 0.5
        self.cursor_visible = False
        self.paused = False  # Track pause state
        self.cursor_position = (0, 0)
        self.last_gaze_time = time.time()
        self.last_highlighted_widget = None
        self.last_highlight_time = 0
        self.position_history = deque(maxlen=5)
        self.snap_padding = 20
        self.dwell_timer = DwellTimer(dwell_time=1.0)  # Initialize DwellTimer

    def start_gaze_tracking(self):
        logging.info("Starting gaze tracking")
        if not self.root.cursor_manager.is_cursor_canvas_ready():
            logging.warning("Cursor canvas not ready, attempting to reinitialize")
            self.root.calibrator.show_calibration_window()
            if not self.root.cursor_manager.is_cursor_canvas_ready():
                self.root.status_label.configure(text="Error: Cursor canvas not initialized")
                messagebox.showerror("Error", "Failed to initialize cursor canvas. Please restart the application.")
                return

        try:
            max_retries = 5
            retry_delay = 0.5
            for attempt in range(max_retries):
                if self.root.calibrator.cap is None or not self.root.calibrator.cap.isOpened():
                    logging.info(f"Camera not available, attempt {attempt + 1}/{max_retries} to initialize...")
                    self.root.calibrator.cap = cv2.VideoCapture(0)
                    if self.root.calibrator.cap.isOpened():
                        logging.info("Camera initialized successfully")
                        break
                    else:
                        logging.warning(f"Failed to open camera on attempt {attempt + 1}")
                        time.sleep(retry_delay)
                else:
                    logging.info("Camera already available")
                    break
            else:
                raise Exception(f"Failed to open camera after {max_retries} attempts")

            if self.root.calibrator.face_mesh is None:
                logging.info("Face mesh not available, initializing...")
                self.root.calibrator.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
                if self.root.calibrator.face_mesh is None:
                    raise Exception("Failed to initialize face mesh")
                else:
                    logging.info("Face mesh initialized successfully")
            else:
                logging.info("Face mesh already available")

            self.gaze_tracking = True
            self.cursor_visible = True
            self.paused = False
            logging.info(f"Gaze tracking set to: {self.gaze_tracking}, Cursor visible: {self.cursor_visible}")

            self.gaze_thread = threading.Thread(target=self.track_gaze)
            self.gaze_thread.daemon = True
            self.gaze_thread.start()
            logging.info("Gaze tracking thread started")

            self.root.cursor_manager.show_cursor()
            logging.info("Cursor set to visible state")

            self.update_cursor()
            logging.info("Cursor update loop started")

            self.root.ui_manager.show_keyboard_area()
            logging.info("Keyboard area displayed after calibration")

            if hasattr(self.root, 'calibration_window'):
                self.root.calibration_window.withdraw()
                logging.info("Calibration window withdrawn after starting gaze tracking")

        except Exception as e:
            logging.error(f"Error starting gaze tracking: {str(e)}")
            self.root.status_label.configure(text=f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to start gaze tracking: {str(e)}")
            self.stop_gaze_tracking()
            self.root.stop_calibration()

    def stop_gaze_tracking(self):
        """Stop gaze tracking and clean up resources."""
        logging.info("Stopping gaze tracking")
        self.gaze_tracking = False
        self.cursor_visible = False
        self.paused = False
        if self.gaze_thread and self.gaze_thread.is_alive():
            self.gaze_thread.join(timeout=1.0)
            logging.info("Gaze tracking thread joined")
        self.gaze_thread = None
        self.dwell_timer.reset()  # Reset the dwell timer
        self.root.cursor_manager.hide_cursor()
        logging.info("Cursor hidden during gaze tracking stop")

    def toggle_pause(self):
        """Toggle the pause state."""
        self.paused = not self.paused
        if self.paused:
            self.cursor_visible = False
            self.root.cursor_manager.hide_cursor()
            logging.info("Paused: Cursor hidden, typing disabled")
        else:
            self.cursor_visible = True
            self.root.cursor_manager.show_cursor()
            logging.info("Unpaused: Cursor visible, typing enabled")

    def track_gaze(self):
        logging.info("Gaze tracking thread started")
        frame_count = 0
        last_update_time = time.time()
        error_count = 0
        max_errors = 5

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.smooth_x = screen_width // 2
        self.smooth_y = screen_height // 2

        while self.gaze_tracking:
            try:
                if not self.root.calibrator.cap or not self.root.calibrator.cap.isOpened():
                    raise Exception("Camera not available")

                ret, frame = self.root.calibrator.cap.read()
                if not ret:
                    raise Exception("Failed to read frame")

                frame_count += 1
                current_time = time.time()

                if current_time - last_update_time >= 1.0:
                    fps = frame_count / (current_time - last_update_time)
                    logging.debug(f"Gaze tracking FPS: {fps:.1f}")
                    frame_count = 0
                    last_update_time = current_time

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.root.calibrator.face_mesh.process(rgb)

                if results.multi_face_landmarks and self.root.calibrator.gaze_model is not None:
                    landmarks = results.multi_face_landmarks[0].landmark
                    LEFT_IRIS, RIGHT_IRIS = 468, 473
                    lx, ly = landmarks[LEFT_IRIS].x, landmarks[LEFT_IRIS].y
                    rx, ry = landmarks[RIGHT_IRIS].x, landmarks[RIGHT_IRIS].y
                    cx, cy = (lx + rx) / 2, (ly + ry) / 2

                    pred_x, pred_y = self.root.calibrator.gaze_model.predict([[cx, cy]])[0]
                    pred_x = max(0, min(pred_x, screen_width))
                    pred_y = max(0, min(pred_y, screen_height))

                    self.smooth_x = int((1 - self.smoothing_factor) * self.smooth_x + 
                                      self.smoothing_factor * pred_x)
                    self.smooth_y = int((1 - self.smoothing_factor) * self.smooth_y + 
                                      self.smoothing_factor * pred_y)

                    self.position_history.append((self.smooth_x, self.smooth_y))

                    avg_x = sum(pos[0] for pos in self.position_history) / len(self.position_history)
                    avg_y = sum(pos[1] for pos in self.position_history) / len(self.position_history)

                    self.cursor_position = (int(avg_x), int(avg_y))
                    error_count = 0

                time.sleep(0.016)

            except Exception as e:
                error_count += 1
                logging.error(f"Error in gaze tracking: {str(e)}")

                if error_count >= max_errors:
                    logging.error("Too many consecutive errors, stopping gaze tracking")
                    self.root.after(0, lambda: self.root.status_label.configure(
                        text="Error: Gaze tracking stopped due to errors"
                    ))
                    self.stop_gaze_tracking()
                    self.root.stop_calibration()
                    break

                time.sleep(0.1)

    def update_cursor(self):
        if self.cursor_visible and not self.paused:  # Skip cursor updates when paused
            x, y = self.cursor_position

            widget = self.root.winfo_containing(x, y)
            current_time = time.time()

            if widget and hasattr(widget, 'invoke'):
                btn_x = widget.winfo_rootx()
                btn_y = widget.winfo_rooty()
                btn_width = widget.winfo_width()
                btn_height = widget.winfo_height()

                snap_left = btn_x - self.snap_padding
                snap_right = btn_x + btn_width + self.snap_padding
                snap_top = btn_y - self.snap_padding
                snap_bottom = btn_y + btn_height + self.snap_padding

                if (snap_left <= x <= snap_right and snap_top <= y <= snap_bottom):
                    center_x = btn_x + btn_width // 2
                    center_y = btn_y + btn_height // 2
                    self.cursor_position = (center_x, center_y)
                    x, y = self.cursor_position
                    logging.debug(f"Snapped cursor to button center at ({center_x}, {center_y})")

                    if self.dwell_timer.should_trigger_action(widget):
                        logging.debug(f"Dwell-based click triggered at ({x}, {y})")
                        self.perform_click()

            self.root.cursor_manager.move_cursor(x, y)

            widget = self.root.winfo_containing(x, y)

            highlight_cooldown = 0.5
            if widget and hasattr(widget, 'invoke'):
                if (widget != self.last_highlighted_widget or 
                    (current_time - self.last_highlight_time) > highlight_cooldown):
                    if self.last_highlighted_widget:
                        original_color = "#3A3A3A" if self.last_highlighted_widget == self.root.ui_manager.get_speak_btn() else "#1E1E1E"
                        self.last_highlighted_widget.configure(bg=original_color)

                    widget.configure(bg="#555555")
                    self.last_highlighted_widget = widget
                    self.last_highlight_time = current_time
                    logging.debug(f"Highlighted widget at ({x}, {y})")

            elif self.last_highlighted_widget:
                original_color = "#3A3A3A" if self.last_highlighted_widget == self.root.ui_manager.get_speak_btn() else "#1E1E1E"
                self.last_highlighted_widget.configure(bg=original_color)
                self.last_highlighted_widget = None
                logging.debug("Reset highlight as cursor moved off widget")

        self.root.after(50, self.update_cursor)

    def perform_click(self):
        if self.paused:  # Skip clicks when paused
            return

        x, y = self.cursor_position
        widget = self.root.winfo_containing(x, y)

        if widget:
            if hasattr(widget, 'invoke'):
                logging.info(f"Invoking widget at ({x}, {y}): {widget.cget('text')}")
                widget.invoke()
                logging.info(f"Button clicked at ({x}, {y})")
            elif isinstance(widget, tk.Entry):
                widget.focus_set()
                self.root.event_generate('<Button-1>', x=x, y=y)
                logging.info(f"Focused entry widget at ({x}, {y})")
        else:
            logging.warning(f"No widget found at ({x}, {y})")