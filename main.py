import tkinter as tk
from tkinter import messagebox
import threading
import logging
import traceback
import sys
from datetime import datetime
from calibrator import Calibrator
from gaze_trackerold5 import GazeTracker
from cursor_manager import CursorManager
from ui_manager import UIManager
from debug_window import DebugWindow

# Set up detailed logging with UTF-8 encoding
log_filename = f'eye_tracking_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler(stream=sys.stdout)
    ]
)

# Ensure StreamHandler uses UTF-8 encoding
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler):
        handler.setStream(sys.stdout)
        handler.stream.reconfigure(encoding='utf-8', errors='replace')

class EyeComApp(tk.Tk):
    def __init__(self):
        super().__init__()
        logging.info("=== Starting EyeComApp Initialization ===")

        try:
            self.title("Eye Com - Premium Black Theme")
            self.configure(bg="#121212")

            self.cleanup_in_progress = False
            self.debug_window = None
            self.transition_to_gaze = False

            self.init_status = tk.Label(
                self,
                text="Initializing...",
                font=("Segoe UI", 12),
                fg="white",
                bg="#121212"
            )
            self.init_status.place(relx=0.5, rely=0.5, anchor="center")
            self.update_idletasks()

            self.cursor_manager = CursorManager(self)

            self.init_status.configure(text="Initializing cursor canvas...")
            self.update_idletasks()

            self.calibrator = Calibrator(self)
            self.gaze_tracker = GazeTracker(self)
            self.ui_manager = UIManager(self, self.start_calibration, self.show_debug_window)

            self.init_status.destroy()
            self.state('zoomed')

            self.protocol("WM_DELETE_WINDOW", self.on_closing)

            logging.info("=== EyeComApp Initialization Complete ===")

        except Exception as e:
            logging.error(f"Critical error during initialization: {str(e)}\n{traceback.format_exc()}")
            if hasattr(self, 'init_status'):
                self.init_status.configure(text=f"Initialization failed: {str(e)}")
            messagebox.showerror("Critical Error", f"Failed to initialize application: {str(e)}")
            raise

    def start_calibration(self):
        """Start the calibration process."""
        self.calibrator.show_calibration_window()

    def show_debug_window(self):
        """Show the debug window."""
        if self.debug_window is None or not self.debug_window.winfo_exists():
            self.debug_window = DebugWindow(self)
        else:
            self.debug_window.lift()

    def toggle_pause(self):
        """Toggle the pause/unpause state for GazeTracker."""
        self.gaze_tracker.toggle_pause()

    def on_closing(self):
        """Handle window closing."""
        logging.info("Main window closing")
        self.stop_calibration()
        self.quit()
        self.destroy()

    def force_cleanup(self):
        """Perform a forced cleanup of resources."""
        logging.info("Force cleanup initiated")
        try:
            self.calibrator.calibration_running = False
            self.gaze_tracker.stop_gaze_tracking()

            if self.calibrator.cap is not None:
                try:
                    self.calibrator.cap.release()
                    logging.info("Camera released successfully during force cleanup")
                except Exception as e:
                    logging.error(f"Error releasing camera during force cleanup: {str(e)}")
                self.calibrator.cap = None

            if self.calibrator.face_mesh is not None:
                try:
                    self.calibrator.face_mesh.close()
                    logging.info("Face mesh closed successfully during force cleanup")
                except Exception as e:
                    logging.error(f"Error closing face mesh during force cleanup: {str(e)}")
                self.calibrator.face_mesh = None

            self.cursor_manager.hide_cursor()

            if hasattr(self, 'calibration_window'):
                try:
                    self.calibration_window.destroy()
                    logging.info("Calibration window destroyed during force cleanup")
                except Exception as e:
                    logging.error(f"Error destroying calibration window during force cleanup: {str(e)}")
                delattr(self, 'calibration_window')

            # Close the PhrasesWindow if it exists
            self.ui_manager.close_phrases_window()

            logging.info("Force cleanup completed")
        except Exception as e:
            logging.error(f"Error during force cleanup: {str(e)}")

    def stop_calibration(self):
        """Stop calibration and clean up resources."""
        if self.cleanup_in_progress:
            logging.warning("Cleanup already in progress")
            return

        self.cleanup_in_progress = True
        logging.info("Stopping calibration and cleaning up")

        try:
            self.calibrator.calibration_running = False
            self.gaze_tracker.stop_gaze_tracking()

            if self.calibrator.calibration_thread and self.calibrator.calibration_thread.is_alive():
                self.calibrator.calibration_thread.join(timeout=1.0)
                logging.info("Calibration thread joined")

            if self.calibrator.cap is not None:
                try:
                    self.calibrator.cap.release()
                    logging.info("Camera released successfully")
                except Exception as e:
                    logging.error(f"Error releasing camera: {str(e)}")
                finally:
                    self.calibrator.cap = None

            if self.calibrator.face_mesh is not None:
                try:
                    self.calibrator.face_mesh.close()
                    logging.info("Face mesh closed successfully")
                except Exception as e:
                    logging.error(f"Error closing face mesh: {str(e)}")
                finally:
                    self.calibrator.face_mesh = None

            self.cursor_manager.hide_cursor()

            if hasattr(self, 'calibration_window'):
                try:
                    self.calibration_window.destroy()
                    logging.info("Calibration window destroyed")
                except Exception as e:
                    logging.error(f"Error destroying calibration window: {str(e)}")
                finally:
                    delattr(self, 'calibration_window')

            # Close the PhrasesWindow if it exists
            self.ui_manager.close_phrases_window()

            logging.info("Cleanup completed successfully")

        except Exception as e:
            logging.error(f"Error during cleanup: {str(e)}")
            self.force_cleanup()
        finally:
            self.cleanup_in_progress = False

    def update_calibration_canvas(self, photo):
        """Update the calibration canvas with the given photo."""
        if hasattr(self, 'calibration_canvas'):
            self.calibration_canvas.delete("all")
            self.calibration_canvas.create_image(0, 0, image=photo, anchor="nw")
            self.calibration_canvas.photo = photo

if __name__ == "__main__":
    app = EyeComApp()
    app.mainloop()