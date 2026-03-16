"""
ASL Alphabet Sign Language Classifier.

Uses MediaPipe hand landmark geometry to classify static ASL alphabet
signs without any ML training. Computes a feature vector from normalized
distances and angular relationships between landmarks, then applies
rule-based geometric constraints to identify 20 common ASL letters:
A, B, C, D, E, F, G, H, I, K, L, O, P, Q, R, S, U, V, W, Y.

Each letter is defined by a set of finger-state conditions (extended/curled)
plus specific spatial constraints (thumb position, finger spread, etc.).
"""

from src.utils.angles import euclidean_distance_2d


# Landmark indices
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20


class SignLanguageClassifier:
    """
    Rule-based ASL alphabet classifier from hand landmarks.

    Supports letters: A, B, C, D, E, F, G, H, I, K, L, O, P, Q, R, S, U, V, W, Y

    Usage:
        clf = SignLanguageClassifier()
        letter, confidence = clf.classify(hand_landmarks, hand_type)
    """

    def __init__(self):
        self._last_letter = None
        self._last_confidence = 0.0
        self._sentence = []
        self._stable_count = 0
        self._stable_threshold = 10  # frames before adding to sentence

    @property
    def sentence(self):
        """Current spelled-out sentence."""
        return ''.join(self._sentence)

    def clear_sentence(self):
        self._sentence = []

    def backspace(self):
        if self._sentence:
            self._sentence.pop()

    def add_space(self):
        self._sentence.append(' ')

    def classify(self, landmarks, hand_type='Right'):
        """
        Classify the hand pose as an ASL letter.

        Parameters
        ----------
        landmarks : dict {id: (x, y, z)}
            Hand landmarks from GestureTracker.
        hand_type : str
            'Left' or 'Right' for thumb logic.

        Returns
        -------
        (letter: str or None, confidence: float)
        """
        if not landmarks or len(landmarks) < 21:
            return None, 0.0

        fingers = self._get_finger_states(landmarks, hand_type)
        curls = self._get_curl_values(landmarks)
        letter, conf = self._match_letter(fingers, curls, landmarks, hand_type)

        # Stability tracking for sentence building
        if letter == self._last_letter and letter is not None:
            self._stable_count += 1
            if self._stable_count == self._stable_threshold:
                # Add to sentence only once per stable detection
                self._sentence.append(letter)
        else:
            self._stable_count = 0

        self._last_letter = letter
        self._last_confidence = conf
        return letter, conf

    def _get_finger_states(self, lm, hand_type):
        """
        Get binary finger extended states.
        Returns [thumb, index, middle, ring, pinky] as booleans.
        """
        fingers = []

        # Thumb
        if hand_type == 'Right':
            fingers.append(lm[THUMB_TIP][0] < lm[THUMB_IP][0])
        else:
            fingers.append(lm[THUMB_TIP][0] > lm[THUMB_IP][0])

        # Other 4 fingers: tip y < PIP y means extended
        for tip, pip in [
            (INDEX_TIP, INDEX_PIP),
            (MIDDLE_TIP, MIDDLE_PIP),
            (RING_TIP, RING_PIP),
            (PINKY_TIP, PINKY_PIP),
        ]:
            fingers.append(lm[tip][1] < lm[pip][1])

        return fingers

    def _get_curl_values(self, lm):
        """
        Compute curl ratio for each finger (0=curled, 1=extended).
        """
        tips = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
        mcps = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
        curls = []
        for tip, mcp in zip(tips, mcps):
            tip_mcp = euclidean_distance_2d(lm[tip][:2], lm[mcp][:2])
            wrist_mcp = euclidean_distance_2d(lm[WRIST][:2], lm[mcp][:2])
            if wrist_mcp == 0:
                curls.append(0.0)
            else:
                curls.append(min(1.0, tip_mcp / (wrist_mcp * 1.8)))
        return curls

    def _match_letter(self, fingers, curls, lm, hand_type):
        """
        Rule-based letter matching.
        Returns (letter, confidence) or (None, 0.0).
        """
        thumb, index, middle, ring, pinky = fingers
        total_up = sum(fingers)

        # --- Letter A: Fist with thumb to the side (not tucked) ---
        if total_up <= 1 and not index and not middle and not ring and not pinky:
            if thumb:
                return 'A', 0.85
            # Thumb ambiguous but fist-like
            if curls[1] < 0.3 and curls[2] < 0.3 and curls[3] < 0.3 and curls[4] < 0.3:
                return 'A', 0.65

        # --- Letter E: Fist with thumb tucked in front (all fingers curled) ---
        if total_up == 0 and not thumb:
            return 'E', 0.80

        # --- Letter S: Fist with thumb over fingers ---
        if total_up == 0 and thumb:
            # Check for tight fist curl
            if all(c < 0.4 for c in curls[1:]):
                return 'S', 0.80

        # --- Letter B: All 4 fingers up, thumb curled across palm ---
        if not thumb and index and middle and ring and pinky:
            # Check fingers are together (not spread)
            idx_mid_dist = euclidean_distance_2d(lm[INDEX_TIP][:2], lm[MIDDLE_TIP][:2])
            mid_ring_dist = euclidean_distance_2d(lm[MIDDLE_TIP][:2], lm[RING_TIP][:2])
            ref = euclidean_distance_2d(lm[WRIST][:2], lm[MIDDLE_MCP][:2])
            if ref > 0 and idx_mid_dist / ref < 0.4 and mid_ring_dist / ref < 0.4:
                return 'B', 0.85

        # --- Letter C: Curved hand, fingers together, thumb opposed ---
        if total_up >= 3:
            # Check if fingertips form a C-curve (thumb tip far from index tip)
            thumb_idx = euclidean_distance_2d(lm[THUMB_TIP][:2], lm[INDEX_TIP][:2])
            ref = euclidean_distance_2d(lm[WRIST][:2], lm[MIDDLE_MCP][:2])
            if ref > 0:
                # Fingers curved (moderate curl) and thumb opposed
                avg_curl = sum(curls[1:]) / 4
                if 0.3 < avg_curl < 0.7 and thumb_idx / ref > 0.3:
                    return 'C', 0.70

        # --- Letter D: Index up, middle+ring+pinky curled, thumb touching middle ---
        if index and not middle and not ring and not pinky:
            thumb_mid = euclidean_distance_2d(lm[THUMB_TIP][:2], lm[MIDDLE_TIP][:2])
            ref = euclidean_distance_2d(lm[WRIST][:2], lm[INDEX_MCP][:2])
            if ref > 0 and thumb_mid / ref < 0.5:
                return 'D', 0.80

        # --- Letter F: "OK" sign (Index + Thumb touch, others up) ---
        if not index and middle and ring and pinky:
            thumb_idx = euclidean_distance_2d(lm[THUMB_TIP][:2], lm[INDEX_TIP][:2])
            ref = euclidean_distance_2d(lm[WRIST][:2], lm[INDEX_MCP][:2])
            if ref > 0 and thumb_idx / ref < 0.25:
                return 'F', 0.90

        # --- Letter G/H: Side-pointing index or index+middle ---
        if index and not ring and not pinky:
            dx_idx = abs(lm[INDEX_TIP][0] - lm[INDEX_MCP][0])
            dy_idx = abs(lm[INDEX_TIP][1] - lm[INDEX_MCP][1])
            if dx_idx > dy_idx: # Pointing horizontally
                if middle: return 'H', 0.80
                else: return 'G', 0.80

        # --- Letter I: Only pinky extended ---
        if not thumb and not index and not middle and not ring and pinky:
            return 'I', 0.85

        # --- Letter K: V-shape with thumb tucked between index and middle ---
        if index and middle and not ring and not pinky:
            thumb_idx = euclidean_distance_2d(lm[THUMB_TIP][:2], lm[INDEX_MCP][:2])
            ref = euclidean_distance_2d(lm[WRIST][:2], lm[INDEX_MCP][:2])
            if ref > 0 and 0.1 < thumb_idx / ref < 0.4:
                return 'K', 0.75

        # --- Letter L: Thumb + Index extended (L-shape) ---
        if thumb and index and not middle and not ring and not pinky:
            return 'L', 0.80

        # --- Letter O: All fingers curled into O-shape, tips touching ---
        if total_up == 0:
            thumb_idx = euclidean_distance_2d(lm[THUMB_TIP][:2], lm[INDEX_TIP][:2])
            ref = euclidean_distance_2d(lm[WRIST][:2], lm[MIDDLE_MCP][:2])
            if ref > 0 and thumb_idx / ref < 0.3:
                return 'O', 0.75

        # --- Letter P/Q: Downward pointing logic ---
        if lm[INDEX_TIP][1] > lm[INDEX_MCP][1]: # Pointing down
            if index and middle: return 'P', 0.70
            if index and not middle: return 'Q', 0.70

        # --- Letter R: Index and middle crossed ---
        if index and middle and not ring and not pinky:
            idx_mid = euclidean_distance_2d(lm[INDEX_TIP][:2], lm[MIDDLE_TIP][:2])
            ref = euclidean_distance_2d(lm[WRIST][:2], lm[MIDDLE_MCP][:2])
            if ref > 0 and idx_mid / ref < 0.15:
                return 'R', 0.75

        # --- Letter U/V: Index and middle extended ---
        if index and middle and not ring and not pinky:
            idx_mid = euclidean_distance_2d(lm[INDEX_TIP][:2], lm[MIDDLE_TIP][:2])
            ref = euclidean_distance_2d(lm[WRIST][:2], lm[MIDDLE_MCP][:2])
            if ref > 0:
                if idx_mid / ref < 0.35: return 'U', 0.80
                else: return 'V', 0.85

        # --- Letter W: Index + Middle + Ring extended ---
        if not thumb and index and middle and ring and not pinky:
            return 'W', 0.80

        # --- Letter Y: Thumb + Pinky extended ---
        if thumb and not index and not middle and not ring and pinky:
            return 'Y', 0.85

        return None, 0.0

