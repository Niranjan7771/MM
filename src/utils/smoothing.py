"""
Exponential Moving Average (EMA) filter for temporal signal smoothing.

MediaPipe landmark data is inherently noisy frame-to-frame. This module
provides a lightweight EMA filter that smooths scalar values (angles,
ratios, distances) to produce stable, non-jittery readouts in the HUD.

Formula:  smoothed_t = alpha * raw_t + (1 - alpha) * smoothed_(t-1)
"""


class EMAFilter:
    """
    Exponential Moving Average filter for a single scalar signal.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1]. Higher values track faster but
        are noisier. Lower values are smoother but lag more.
        Typical: 0.3 for angles, 0.5 for ratios.
    """

    def __init__(self, alpha=0.3):
        self.alpha = max(0.01, min(1.0, alpha))
        self._value = None

    def update(self, raw_value):
        """Feed a new raw sample and return the smoothed value."""
        if self._value is None:
            self._value = raw_value
        else:
            self._value = self.alpha * raw_value + (1.0 - self.alpha) * self._value
        return self._value

    def get(self):
        """Return the current smoothed value (None if never updated)."""
        return self._value

    def reset(self):
        """Clear the filter state."""
        self._value = None


class EMAFilterBank:
    """
    Manages a collection of named EMA filters for convenient multi-signal
    smoothing (e.g., one filter per joint angle).

    Usage:
        bank = EMAFilterBank(alpha=0.3)
        smoothed = bank.update('right_elbow', raw_angle)
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._filters = {}

    def update(self, key, raw_value):
        """Update (or auto-create) the filter named `key` and return smoothed value."""
        if key not in self._filters:
            self._filters[key] = EMAFilter(self.alpha)
        return self._filters[key].update(raw_value)

    def get(self, key, default=None):
        """Return the current smoothed value for `key`, or default."""
        if key in self._filters:
            val = self._filters[key].get()
            return val if val is not None else default
        return default

    def reset_all(self):
        """Reset every filter in the bank."""
        for f in self._filters.values():
            f.reset()
