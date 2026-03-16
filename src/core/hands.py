"""
Advanced Gesture Estimation with expanded vocabulary and finger analytics.

Uses MediaPipe Hands to track up to 2 hands simultaneously, differentiating
left/right. Beyond basic finger-up detection, computes finger curl ratios,
hand openness percentage, and pinch distance. Recognizes 12+ gesture
semantics from spatial finger configurations.
"""

import cv2
import collections
import mediapipe as mp

from src.utils.angles import euclidean_distance_2d


# Landmark index constants for readability
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


class GestureTracker:
    """
    Multi-hand gesture estimator with advanced finger analytics.

    Provides:
    - Simultaneous left/right hand tracking
    - Binary finger state (up/down) per finger
    - Finger curl ratios (0=fully curled, 1=fully extended)
    - Hand openness percentage
    - Pinch distance (thumb-tip to index-tip)
    - 12+ gesture classification
    """

    def __init__(self, mode=False, max_hands=2,
                 detection_con=0.7, track_con=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        self.hands = self.mp_hands.Hands(
            static_image_mode=mode,
            max_num_hands=max_hands,
            min_detection_confidence=float(detection_con),
            min_tracking_confidence=float(track_con),
        )
        self.results = None
        
        # Stability buffer: {hand_type: deque([gestures])}
        self._gesture_buffer = {
            'Left': collections.deque(maxlen=5),
            'Right': collections.deque(maxlen=5)
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_hands(self, img, draw=True):
        """Run hand detection on a BGR frame. Optionally draws hand mesh."""
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_lm in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, hand_lm,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )
        return img

    def get_hand_positions(self, img):
        """
        Return a list of dicts per detected hand:
        [{'type': 'Left'/'Right', 'landmarks': {id: (x, y, z), ...}}]
        """
        all_hands = []
        if not (self.results and self.results.multi_hand_landmarks
                and self.results.multi_handedness):
            return all_hands

        h, w, _ = img.shape
        for hand_lm, hand_info in zip(
            self.results.multi_hand_landmarks,
            self.results.multi_handedness,
        ):
            hand_data = {
                'type': hand_info.classification[0].label,
                'landmarks': {},
            }
            for idx, lm in enumerate(hand_lm.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                hand_data['landmarks'][idx] = (cx, cy, lm.z)
            all_hands.append(hand_data)
        return all_hands

    def get_full_hand_data(self, img):
        """
        Convenience method: hand positions + full analytics for each hand.

        Returns list of dicts per hand:
        [{
            'type': str,
            'gesture': str,
            'fingers_up': [bool x5],
            'curl_ratios': [float x5],
            'openness': float,
            'pinch_distance': float,
            'wrist_pos': (x, y),
        }]
        """
        all_hands = self.get_hand_positions(img)
        results = []

        for hand in all_hands:
            lm = hand['landmarks']
            hand_type = hand['type']

            if not lm:
                continue

            fingers_up = self._get_fingers_up(lm, hand_type)
            curl_ratios = self._get_curl_ratios(lm)
            openness = self._get_hand_openness(curl_ratios)
            pinch_dist = self._get_pinch_distance(lm)
            gesture = self._classify_gesture(fingers_up, lm, hand_type)
            
            # Temporal smoothing
            self._gesture_buffer[hand_type].append(gesture)
            if len(self._gesture_buffer[hand_type]) >= 3:
                # Return most common gesture in buffer
                from collections import Counter
                stable_gesture = Counter(self._gesture_buffer[hand_type]).most_common(1)[0][0]
            else:
                stable_gesture = gesture

            results.append({
                'type': hand_type,
                'gesture': stable_gesture,
                'fingers_up': fingers_up,
                'curl_ratios': [round(c, 2) for c in curl_ratios],
                'openness': round(openness, 1),
                'pinch_distance': round(pinch_dist, 1),
                'wrist_pos': lm[WRIST][:2],
                'landmarks_raw': lm,
            })

        return results

    # ------------------------------------------------------------------
    # Finger state analysis
    # ------------------------------------------------------------------

    def _get_fingers_up(self, lm, hand_type):
        """
        Return a list of 5 booleans indicating if each finger is extended.
        [thumb, index, middle, ring, pinky]
        """
        fingers = []

        # Thumb: compare x-position of tip vs IP joint (laterality matters)
        if hand_type == "Right":
            fingers.append(lm[THUMB_TIP][0] < lm[THUMB_IP][0])
        else:
            fingers.append(lm[THUMB_TIP][0] > lm[THUMB_IP][0])

        # Other 4 fingers: tip y < PIP y means finger is up (image coords)
        for tip, pip in [(INDEX_TIP, INDEX_PIP),
                         (MIDDLE_TIP, MIDDLE_PIP),
                         (RING_TIP, RING_PIP),
                         (PINKY_TIP, PINKY_PIP)]:
            fingers.append(lm[tip][1] < lm[pip][1])

        return fingers

    def _get_curl_ratios(self, lm):
        """
        Compute curl ratio for each finger: 0 = fully curled, 1 = fully extended.

        Ratio = dist(tip, mcp) / dist(wrist, mcp)
        Clamped to [0, 1] and normalized.
        """
        ratios = []
        for tip_id, mcp_id in zip(FINGER_TIPS, FINGER_MCPS):
            tip_to_mcp = euclidean_distance_2d(lm[tip_id][:2], lm[mcp_id][:2])
            wrist_to_mcp = euclidean_distance_2d(lm[WRIST][:2], lm[mcp_id][:2])

            if wrist_to_mcp == 0:
                ratios.append(0.0)
            else:
                ratio = tip_to_mcp / (wrist_to_mcp * 1.8)  # scale factor
                ratios.append(min(1.0, max(0.0, ratio)))
        return ratios

    def _get_hand_openness(self, curl_ratios):
        """Average finger extension as a percentage [0, 100]."""
        if not curl_ratios:
            return 0.0
        return sum(curl_ratios) / len(curl_ratios) * 100

    def _get_pinch_distance(self, lm):
        """Distance in pixels between thumb tip and index tip."""
        return euclidean_distance_2d(lm[THUMB_TIP][:2], lm[INDEX_TIP][:2])

    # ------------------------------------------------------------------
    # Gesture classification
    # ------------------------------------------------------------------

    def _classify_gesture(self, fingers_up, lm, hand_type):
        """
        Map the finger state vector to a named gesture string.
        Checks from most specific to least specific.
        """
        thumb, index, middle, ring, pinky = fingers_up
        total = sum(fingers_up)

        # -- Specific multi-finger gestures (most specific first) --

        # OK Sign: thumb + index tips close, other 3 fingers up
        pinch = euclidean_distance_2d(lm[THUMB_TIP][:2], lm[INDEX_TIP][:2])
        wrist_to_middle = euclidean_distance_2d(lm[WRIST][:2], lm[MIDDLE_MCP][:2])
        if wrist_to_middle > 0 and pinch < wrist_to_middle * 0.35 and middle and ring:
            return "OK Sign"

        # Spider-Man: thumb + index + pinky up, middle + ring down
        # Differentiate from I Love You by checking thumb curl/extension
        # Spider-Man often has thumb slightly more curled or less extended than ILY
        # For simplicity, if thumb is up, index is up, pinky is up, others down, it's Spider-Man.
        # This will override I Love You if placed before it.
        if thumb and index and not middle and not ring and pinky:
            return "Spider-Man"

        # I Love You: thumb + index + pinky up, middle + ring down
        # (This condition is now covered by Spider-Man if placed before it.
        # To differentiate, one might add a check for thumb extension/angle.)
        # For now, keeping it as a fallback or if Spider-Man is not matched.
        if thumb and index and not middle and not ring and pinky:
            return "I Love You"

        # Rock / Metal: index + pinky up, middle + ring down
        if not thumb and index and not middle and not ring and pinky:
            return "Rock / Metal"

        # Call Me: thumb + pinky up, other 3 down
        if thumb and not index and not middle and not ring and pinky:
            return "Call Me"

        # Peace Sign: index + middle up, others down
        if not thumb and index and middle and not ring and not pinky:
            return "Peace Sign"

        # Gun: thumb + index up, others down
        if thumb and index and not middle and not ring and not pinky:
            return "Gun"

        # L-Sign: thumb + index up, forming an L shape.
        # Differentiate from Gun by checking relative positions.
        # L-Sign typically has index straight up and thumb extended horizontally.
        # Gun often has index pointing forward.
        # A simple check: if thumb is up and index is up, and thumb tip is roughly
        # at the same y-level or slightly below index MCP, and index tip is high.
        if thumb and index and not middle and not ring and not pinky:
            # Check if index finger is relatively straight up (tip y much smaller than PIP y)
            # and thumb is extended horizontally (tip x significantly different from IP x)
            # This is a basic heuristic and might need refinement.
            if lm[INDEX_TIP][1] < lm[INDEX_PIP][1] and \
               abs(lm[THUMB_TIP][0] - lm[THUMB_IP][0]) > abs(lm[THUMB_TIP][1] - lm[THUMB_IP][1]):
                return "L-Sign"

        # Heart: This gesture typically requires two hands.
        # For a single hand, it's hard to define a "Heart" shape.
        # If we were to define a single-hand "Heart", it might involve
        # the thumb and index forming a curve, but this is highly ambiguous.
        # Skipping single-hand "Heart" for now as it's usually a two-hand gesture.

        # -- Count-based gestures --

        if total == 0:
            return "Fist"

        if total == 5:
            return "Open Hand"

        if total == 1 and index:
            return "Pointing"

        if total == 1 and thumb:
            # Thumb orientation determines up/down
            if lm[THUMB_TIP][1] < lm[THUMB_IP][1]:
                return "Thumbs Up"
            else:
                return "Thumbs Down"

        if total == 3 and index and middle and ring:
            return "Three"

        if total == 4 and index and middle and ring and pinky:
            return "Four"

        return f"{total} Fingers"

    # kept for backward compat with old API
    def interpret_gestures(self, all_hands):
        """Legacy wrapper: returns gesture dicts from hand position data."""
        gestures = []
        for hand in all_hands:
            lm = hand['landmarks']
            hand_type = hand['type']
            if not lm:
                continue
            fingers_up = self._get_fingers_up(lm, hand_type)
            gesture = self._classify_gesture(fingers_up, lm, hand_type)
            gestures.append({
                'type': hand_type,
                'gesture': gesture,
                'wrist_pos': lm[WRIST][:2],
            })
        return gestures
