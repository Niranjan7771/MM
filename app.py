"""
Multi-Modal Human Pose & Gesture Estimation System -- Web Dashboard
===================================================================

Flask application providing a web-based dashboard for all analysis
features. Runs the analysis pipeline in a background thread and serves
a real-time MJPEG video stream alongside JSON analytics.

Usage:
    python app.py

Then open http://localhost:5000 in your browser.
"""

import sys
from flask import Flask

from src.web.routes import bp, set_stream_manager
from src.web.stream import StreamManager


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'mm-estimation-2024'

    # Initialize the shared stream manager
    stream_manager = StreamManager()
    set_stream_manager(stream_manager)

    # Register blueprint
    app.register_blueprint(bp)

    # Start the background capture thread when first request comes in
    @app.before_request
    def start_stream():
        if getattr(stream_manager, '_camera_failed', False):
            return
            
        if not stream_manager._running:
            try:
                stream_manager.start()
                print("[STREAM] Background capture started.")
            except RuntimeError as e:
                print(f"[ERROR] {e}")
                stream_manager._camera_failed = True

    return app


if __name__ == '__main__':
    print("=" * 62)
    print("  Multi-Modal Human Pose & Gesture Estimation System")
    print("  Web Dashboard v3.0")
    print("=" * 62)
    print()
    print("  Open http://localhost:5000 in your browser")
    print()

    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
