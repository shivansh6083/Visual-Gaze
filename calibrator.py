import tkinter as tk
from tkinter import messagebox
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import logging
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

class Calibrator:
    def __init__(self, root):
        self.root = root
        self.calibration_running = False
        self.calibration_thread = None
        self.cap = None
        self.face_mesh = None
        self.calibration_data = []
        self.gaze_model = None

    def show_calibration_window(self):
        logging.info("Showing calibration window")
        if not self.root.cursor_manager.is_cursor_canvas_ready():
            logging.error("Cursor canvas not ready")
            messagebox.showerror("Error", "Application not properly initialized. Please restart.")
            return

        try:
            self.root.calibration_window = tk.Toplevel(self.root)
            self.root.calibration_window.title("Eye Tracking Calibration")
            self.root.calibration_window.state('zoomed')
            self.root.calibration_window.configure(bg="black")
            self.root.calibration_window.attributes('-topmost', True)

            self.start_frame = tk.Frame(self.root.calibration_window, bg="black")
            self.start_frame.place(relx=0.5, rely=0.5, anchor="center")

            self.start_calibration_btn = tk.Button(
                self.start_frame,
                text="Click to Start Calibration",
                font=("Segoe UI", 16, "bold"),
                fg="white",
                bg="#2196F3",
                activebackground="#1976D2",
                activeforeground="white",
                relief="flat",
                padx=20,
                pady=10,
                command=self.start_calibration
            )
            self.start_calibration_btn.pack()

            self.root.calibration_canvas = tk.Canvas(
                self.root.calibration_window, 
                bg="black",
                highlightthickness=0
            )

            self.close_calibration_btn = tk.Button(
                self.root.calibration_window,
                text="✕",
                font=("Segoe UI", 14),
                fg="white",
                bg="#FF4444",
                activebackground="#FF6666",
                activeforeground="white",
                relief="flat",
                command=self.root.stop_calibration
            )
            self.close_calibration_btn.place(relx=0.98, rely=0.02, anchor="ne")

            self.root.status_label = tk.Label(
                self.root.calibration_window,
                text="",
                font=("Segoe UI", 12),
                fg="white",
                bg="black"
            )
            self.root.status_label.place(relx=0.5, rely=0.1, anchor="center")

            self.root.debug_label = tk.Label(
                self.root.calibration_window,
                text="",
                font=("Segoe UI", 10),
                fg="yellow",
                bg="black"
            )
            self.root.debug_label.place(relx=0.5, rely=0.95, anchor="center")

            self.root.calibration_window.protocol("WM_DELETE_WINDOW", self.root.stop_calibration)
            logging.info("Calibration window created successfully")

            self.root.calibration_window.deiconify()
            self.root.calibration_window.lift()
            self.start_frame.lift()

        except Exception as e:
            logging.error(f"Error showing calibration window: {str(e)}")
            messagebox.showerror("Error", "Failed to show calibration window")

    def start_calibration(self):
        logging.info("Starting calibration process")
        try:
            self.start_frame.place_forget()
            self.root.calibration_canvas.pack(fill="both", expand=True)
            self.root.status_label.configure(text="Calibration in progress... Look at each red dot")

            test_cap = cv2.VideoCapture(0)
            if not test_cap.isOpened():
                raise Exception("Failed to open camera")
            test_cap.release()

            screen_width = self.root.calibration_window.winfo_screenwidth()
            screen_height = self.root.calibration_window.winfo_screenheight()

            if not self.calibration_running:
                self.calibration_running = True
                self.calibration_thread = threading.Thread(
                    target=self.run_calibration,
                    args=(screen_width, screen_height)
                )
                self.calibration_thread.daemon = True
                self.calibration_thread.start()
                logging.info("Calibration thread started")

        except Exception as e:
            logging.error(f"Error starting calibration: {str(e)}")
            self.root.status_label.configure(text="Error: Failed to start calibration")
            messagebox.showerror("Error", f"Failed to start calibration: {str(e)}")

    def run_calibration(self, screen_width, screen_height):
        logging.info("Running calibration")
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            logging.info("Camera opened successfully for calibration")

            self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
            if self.face_mesh is None:
                raise Exception("Failed to initialize face mesh")
            logging.info("Face mesh initialized successfully for calibration")

            self.calibration_data = []

            LEFT_IRIS, RIGHT_IRIS = 468, 473

            calibration_targets = [
                (int(screen_width * 0.1), int(screen_height * 0.1)),
                (int(screen_width * 0.9), int(screen_height * 0.1)),
                (int(screen_width * 0.1), int(screen_height * 0.9)),
                (int(screen_width * 0.9), int(screen_height * 0.9)),
                (int(screen_width * 0.5), int(screen_height * 0.1)),
                (int(screen_width * 0.5), int(screen_height * 0.9)),
                (int(screen_width * 0.1), int(screen_height * 0.5)),
                (int(screen_width * 0.9), int(screen_height * 0.5)),
                (int(screen_width * 0.3), int(screen_height * 0.3)),
                (int(screen_width * 0.7), int(screen_height * 0.3)),
                (int(screen_width * 0.3), int(screen_height * 0.7)),
                (int(screen_width * 0.7), int(screen_height * 0.7)),
            ]

            input_iris = []
            output_screen = []
            current_target = 0
            target_start_time = time.time()
            frame_count = 0
            last_update_time = time.time()

            while self.calibration_running and current_target < len(calibration_targets):
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to read frame from camera")
                    break

                frame_count += 1
                current_time = time.time()

                if current_time - last_update_time >= 1.0:
                    fps = frame_count / (current_time - last_update_time)
                    self.root.after(0, lambda: self.root.debug_label.configure(
                        text=f"FPS: {fps:.1f} | Target: {current_target + 1}/{len(calibration_targets)}"
                    ))
                    frame_count = 0
                    last_update_time = current_time

                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (screen_width, screen_height))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb)

                tx, ty = calibration_targets[current_target]
                cv2.circle(frame, (tx, ty), 20, (0, 0, 255), -1)

                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = face_landmarks.landmark
                        lx, ly = landmarks[LEFT_IRIS].x, landmarks[LEFT_IRIS].y
                        rx, ry = landmarks[RIGHT_IRIS].x, landmarks[RIGHT_IRIS].y
                        cx, cy = (lx + rx) / 2, (ly + ry) / 2

                        if current_time - target_start_time > 2.0:
                            input_iris.append([cx, cy])
                            output_screen.append([tx, ty])
                            logging.info(f"Collected calibration point {current_target + 1}")
                            current_target += 1
                            target_start_time = current_time
                            self.root.after(0, lambda: self.root.status_label.configure(
                                text=f"Look at the red dot: {current_target + 1}/{len(calibration_targets)}"
                            ))

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                photo = ImageTk.PhotoImage(image=image)
                self.root.after(0, lambda p=photo: self.update_calibration_canvas(p))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            self.calibration_running = False

            if len(input_iris) >= 4:
                logging.info("Training gaze model")
                self.gaze_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
                self.gaze_model.fit(np.array(input_iris), np.array(output_screen))

                predictions = self.gaze_model.predict(np.array(input_iris))
                mse = np.mean((predictions - np.array(output_screen)) ** 2)
                logging.info(f"Model MSE: {mse:.2f}")

                self.root.after(0, lambda: self.root.status_label.configure(
                    text=f"Calibration complete! (MSE: {mse:.2f}) Starting gaze tracking..."
                ))
                time.sleep(1)
                self.root.gaze_tracker.start_gaze_tracking()
            else:
                logging.warning("Insufficient calibration points collected")
                self.root.after(0, lambda: self.root.status_label.configure(
                    text="Calibration failed: Not enough points collected"
                ))

            if not self.root.gaze_tracker.gaze_tracking:
                self.root.stop_calibration()

        except Exception as e:
            logging.error(f"Error during calibration: {str(e)}")
            self.root.after(0, lambda: self.root.status_label.configure(
                text=f"Calibration error: {str(e)}"
            ))
            self.root.stop_calibration()

    def update_calibration_canvas(self, photo):
        """Update the calibration canvas with the given photo."""
        if hasattr(self.root, 'calibration_canvas'):
            self.root.calibration_canvas.delete("all")
            self.root.calibration_canvas.create_image(0, 0, image=photo, anchor="nw")
            self.root.calibration_canvas.photo = photo