"""GUI-only R reset request; never reset PhysX from an input/render callback."""


class PlayResetKeyboard:
    def __init__(self):
        # Import only after AppLauncher, and never construct in headless/train.
        import carb.input
        import omni.appwindow
        self._types = carb.input.KeyboardEventType
        self._interface = carb.input.acquire_input_interface()
        window = omni.appwindow.get_default_app_window()
        if window is None:
            raise RuntimeError("GUI reset hotkey requires an Isaac Sim app window")
        self._keyboard = window.get_keyboard()
        self._pending = False
        self._held = False
        self._subscription = self._interface.subscribe_to_keyboard_events(self._keyboard, self._on_event)

    def _on_event(self, event, *args):
        if event.input.name != "R":
            return True
        if event.type == self._types.KEY_PRESS and not self._held:
            self._pending = True
            self._held = True
        elif event.type == self._types.KEY_RELEASE:
            self._held = False
        return True

    def consume(self):
        """Consume once at the next behavior-loop boundary, outside callbacks."""
        pending, self._pending = self._pending, False
        return pending

    def close(self):
        if self._subscription is not None:
            self._interface.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None
        self._pending = False
        self._held = False
