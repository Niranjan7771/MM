"""
Pose Symmetry Analyzer for bilateral body comparison.

Compares left-side vs right-side joint angles to compute a real-time
symmetry score. Useful for physiotherapy, ergonomic assessment, and
movement quality analysis.

Provides per-joint symmetry ratios and an overall body symmetry percentage.
"""


# Joint pairs: (left_key, right_key, display_name)
SYMMETRY_PAIRS = [
    ('left_elbow', 'right_elbow', 'Elbows'),
    ('left_shoulder', 'right_shoulder', 'Shoulders'),
    ('left_knee', 'right_knee', 'Knees'),
    ('left_hip', 'right_hip', 'Hips'),
]


class SymmetryAnalyzer:
    """
    Body symmetry scorer comparing bilateral joint angles.

    For each joint pair (left/right), computes:
    - Absolute difference in degrees
    - Symmetry ratio [0, 100%] where 100% = perfect symmetry

    Overall score = average of all pair symmetry ratios.
    """

    def __init__(self):
        self._last_result = {}

    def analyze(self, angles):
        """
        Compute symmetry metrics from joint angles.

        Parameters
        ----------
        angles : dict
            Joint angle dict from PoseTracker.compute_all_angles().

        Returns
        -------
        dict with keys:
            'pairs': list of {name, left_angle, right_angle, diff, symmetry_pct}
            'overall_score': float [0, 100]
        """
        pairs = []
        valid_scores = []

        for left_key, right_key, display_name in SYMMETRY_PAIRS:
            left_val = angles.get(left_key)
            right_val = angles.get(right_key)

            if left_val is None or right_val is None:
                pairs.append({
                    'name': display_name,
                    'left_angle': left_val,
                    'right_angle': right_val,
                    'diff': None,
                    'symmetry_pct': None,
                })
                continue

            diff = abs(left_val - right_val)
            # Symmetry: 100% when diff=0, 0% when diff>=180
            max_angle = max(left_val, right_val, 1.0)
            symmetry = max(0.0, (1.0 - diff / max_angle)) * 100

            pairs.append({
                'name': display_name,
                'left_angle': round(left_val, 1),
                'right_angle': round(right_val, 1),
                'diff': round(diff, 1),
                'symmetry_pct': round(symmetry, 1),
            })
            valid_scores.append(symmetry)

        overall = round(sum(valid_scores) / len(valid_scores), 1) if valid_scores else 0.0

        self._last_result = {
            'pairs': pairs,
            'overall_score': overall,
        }
        return self._last_result

    @property
    def last_result(self):
        return self._last_result
