import threading
import time
from pynput import keyboard

class KeyboardInput:
    """
    Minimal keyboard input reader.

    Provides:
        vx, vy, wz in [-1, 1] based on "w", "a", "s", "d" and arrow keys.
        Speed adjustment using "Page Up" and "Page Down" keys.
    """

    def __init__(self, deadzone=0.05):
        self.deadzone = deadzone

        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.speed = 0.1  # Initial speed

        self._running = False
        self._thread = None
        self._key_states = set()

    def start(self):
        """Start background reader thread."""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        # Start listening to keyboard events
        self.listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def stop(self):
        """Stop reader thread."""
        self._running = False
        if self._thread:
            self._thread.join()
        if self.listener:
            self.listener.stop()

    def get(self, alpha_lin=1., alpha_ang=1.):
        return alpha_lin * self.vx * self.speed, alpha_lin * self.vy * self.speed, alpha_ang * self.wz * self.speed

    def _apply_deadzone(self, v):
        return 0.0 if abs(v) < self.deadzone else v

    def _on_press(self, key):
        try:
            key_name = key.char if hasattr(key, 'char') else key.name
            self._key_states.add(key_name)

            # Adjust speed with Page Up and Page Down
            if key_name == "page_up":
                self.speed = min(self.speed + 0.1, 1.0)
            elif key_name == "page_down":
                self.speed = max(self.speed - 0.1, 0.1)
        except AttributeError:
            pass

    def _on_release(self, key):
        try:
            key_name = key.char if hasattr(key, 'char') else key.name
            self._key_states.discard(key_name)
        except AttributeError:
            pass

    def _loop(self):
        try:
            while self._running:
                # Reset values
                self.vx, self.vy, self.wz = 0.0, 0.0, 0.0

                # Check for key presses
                if "w" in self._key_states or "up" in self._key_states:
                    self.vx = 1.0
                elif "s" in self._key_states or "down" in self._key_states:
                    self.vx = -1.0

                if "a" in self._key_states or "left" in self._key_states:
                    self.vy = -1.0
                elif "d" in self._key_states or "right" in self._key_states:
                    self.vy = 1.0

                if "left" in self._key_states:
                    self.wz = -1.0
                elif "right" in self._key_states:
                    self.wz = 1.0

                # Apply deadzone
                self.vx = self._apply_deadzone(self.vx)
                self.vy = self._apply_deadzone(self.vy)
                self.wz = self._apply_deadzone(self.wz)

                time.sleep(0.01)

        except Exception as e:
            print(f"[KeyboardInput] Error: {e}")
            self._running = False