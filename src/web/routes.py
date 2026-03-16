"""
Flask route definitions for the web dashboard.

Provides page routes, MJPEG video stream, JSON analytics API,
and control endpoints for toggling modules and recording.
"""

from flask import Blueprint, render_template, Response, jsonify, request

bp = Blueprint('main', __name__)

# The stream manager instance is set by app.py
_stream = None


def set_stream_manager(sm):
    """Called by app.py to inject the shared StreamManager."""
    global _stream
    _stream = sm


# ------------------------------------------------------------------
# Page routes
# ------------------------------------------------------------------

@bp.route('/')
def index():
    return render_template('index.html')


@bp.route('/camera')
def camera():
    return render_template('camera.html')


@bp.route('/sign-language')
def sign_language():
    return render_template('sign_language.html')


@bp.route('/game')
def game():
    return render_template('game.html')


# ------------------------------------------------------------------
# Video stream
# ------------------------------------------------------------------

def _mjpeg_generator():
    """Generate MJPEG frames for streaming."""
    import time
    while True:
        frame = _stream.get_frame_jpeg()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.033)  # Max ~30 FPS
        else:
            time.sleep(0.05)


@bp.route('/video_feed')
def video_feed():
    return Response(_mjpeg_generator(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ------------------------------------------------------------------
# JSON API
# ------------------------------------------------------------------

@bp.route('/api/analytics')
def api_analytics():
    return jsonify(_stream.get_analytics())


@bp.route('/api/sign_language')
def api_sign_language():
    return jsonify(_stream.get_sign_data())


@bp.route('/api/game_state')
def api_game_state():
    return jsonify(_stream.get_game_state())


# ------------------------------------------------------------------
# Control API
# ------------------------------------------------------------------

@bp.route('/api/toggle/<module>', methods=['POST'])
def api_toggle(module):
    result = _stream.toggle_module(module)
    return jsonify({'module': module, 'state': result})


@bp.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    path = _stream.take_snapshot()
    return jsonify({'path': path})


@bp.route('/api/reset_exercise', methods=['POST'])
def api_reset_exercise():
    _stream.reset_exercise()
    return jsonify({'status': 'ok'})


@bp.route('/api/sign/clear', methods=['POST'])
def api_sign_clear():
    _stream.clear_sign_sentence()
    return jsonify({'status': 'ok'})


@bp.route('/api/sign/backspace', methods=['POST'])
def api_sign_backspace():
    _stream.sign_backspace()
    return jsonify({'status': 'ok'})


@bp.route('/api/sign/space', methods=['POST'])
def api_sign_space():
    _stream.sign_space()
    return jsonify({'status': 'ok'})
