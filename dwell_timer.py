import time
import logging

class DwellTimer:
    def __init__(self, dwell_time=1.0):
        self.dwell_time = dwell_time  # Dwell time in seconds (previously click_cooldown)
        self.last_action_time = 0  # Tracks the last time an action was triggered
        self.last_widget = None  # Tracks the last widget that was acted upon
        self.current_widget = None  # Tracks the current widget under the cursor
        self.dwell_start_time = 0  # Tracks when the cursor started dwelling on the current widget

    def should_trigger_action(self, widget):
        """
        Determine if an action should be triggered based on dwell time.
        Returns True if the dwell time has been met for the current widget.
        """
        current_time = time.time()

        # Update the current widget
        if widget != self.current_widget:
            self.current_widget = widget
            self.dwell_start_time = current_time
            logging.debug(f"Started dwelling on new widget: {widget}")
            return False

        # Check if the dwell time has been met
        if self.current_widget is None:
            return False

        time_on_widget = current_time - self.dwell_start_time
        if time_on_widget < self.dwell_time:
            return False

        # Check if the widget is different from the last acted-upon widget
        # and if enough time has passed since the last action
        if (self.current_widget != self.last_widget and
                current_time - self.last_action_time >= self.dwell_time):
            logging.debug(f"Dwell time of {self.dwell_time}s met for widget: {widget}")
            self.last_action_time = current_time
            self.last_widget = self.current_widget
            return True

        return False

    def reset(self):
        """Reset the dwell timer state."""
        self.last_action_time = 0
        self.last_widget = None
        self.current_widget = None
        self.dwell_start_time = 0
        logging.debug("DwellTimer reset")

    def set_dwell_time(self, new_dwell_time):
        """Update the dwell time."""
        if new_dwell_time <= 0:
            logging.warning("Dwell time must be positive")
            raise ValueError("Dwell time must be a positive number")
        self.dwell_time = new_dwell_time
        logging.info(f"Dwell time set to {self.dwell_time} seconds")