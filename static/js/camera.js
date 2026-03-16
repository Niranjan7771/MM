/**
 * Camera Analysis Page - Frontend Logic
 */

let lastPosture = "Straight";
let lastExerciseFeedback = {};

function showToast(message, type = 'blue') {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerText = message;
    container.appendChild(toast);

    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (toast.parentNode) toast.remove();
    }, 3000);
}

// Poll analytics
setInterval(fetchAnalytics, 250);

function fetchAnalytics() {
    fetch('/api/analytics')
        .then(r => r.json())
        .then(data => {
            updatePose(data.pose);
            updateHands(data.hands);
            updateFace(data.face);
            updateActivity(data.activity);
            updateExercise(data.exercise);
            updateSymmetry(data.symmetry);
            updatePrediction(
                data.pose ? data.pose.motion_prediction : null, 
                data.pose ? data.pose.prediction_accuracy : null
            );
            updateToggles(data);
        })
        .catch(() => {});
}

function updatePose(data) {
    if (!data) return;
    const conf = document.getElementById('pose-conf');
    const c = data.confidence || 0;
    conf.textContent = c.toFixed(0) + '%';
    conf.className = 'value ' + (c > 70 ? 'good' : c > 40 ? 'warn' : 'bad');
    
    // Posture Alert
    if (data.posture && data.posture !== "Straight" && data.posture !== lastPosture) {
        showToast(`Posture Alert: ${data.posture}!`, 'warning');
        lastPosture = data.posture;
    } else if (data.posture === "Straight") {
        lastPosture = "Straight";
    }

    const container = document.getElementById('angles-container');
    let html = '';
    if (data.angles) {
        for (const [name, val] of Object.entries(data.angles)) {
            const label = name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
            html += '<div class="panel-row"><span class="label">' + label +
                    '</span><span class="value">' + val.toFixed(0) + '&deg;</span></div>';
        }
    }
    container.innerHTML = html;
}

function updateHands(hands) {
    const container = document.getElementById('hand-data');
    if (!hands || hands.length === 0) {
        container.innerHTML = '<span style="color:var(--text-dim);font-size:0.82rem;">No hands detected</span>';
        return;
    }
    let html = '';
    hands.forEach(h => {
        html += '<div class="panel-row"><span class="label">' + h.type +
                '</span><span class="value" style="color:var(--accent-cyan);">' +
                h.gesture + '</span></div>';
        html += '<div class="panel-row"><span class="label" style="padding-left:0.5rem;">Openness</span><span class="value">' +
                h.openness.toFixed(0) + '%</span></div>';
        html += '<div class="panel-row"><span class="label" style="padding-left:0.5rem;">Pinch</span><span class="value">' +
                h.pinch_distance.toFixed(0) + 'px</span></div>';
    });
    container.innerHTML = html;
}

function updateFace(face) {
    const container = document.getElementById('face-data');
    if (!face || Object.keys(face).length === 0) {
        container.innerHTML = '<span style="color:var(--text-dim);font-size:0.82rem;">No face detected</span>';
        return;
    }
    let html = '';
    html += '<div class="panel-row"><span class="label">Expression</span><span class="value" style="color:var(--accent-green);">' +
            (face.expression || 'N/A') + '</span></div>';
    html += '<div class="panel-row"><span class="label">EAR</span><span class="value">' +
            (face.ear_avg || 0).toFixed(3) + '</span></div>';
    html += '<div class="panel-row"><span class="label">MAR</span><span class="value">' +
            (face.mar || 0).toFixed(3) + '</span></div>';
    html += '<div class="panel-row"><span class="label">Head Tilt</span><span class="value">' +
            (face.head_tilt || 0).toFixed(1) + '&deg;</span></div>';
    container.innerHTML = html;
}

function updateActivity(activity) {
    document.getElementById('activity-label').textContent = activity || '--';
}

function updateExercise(exercises) {
    const container = document.getElementById('exercise-data');
    if (!exercises || exercises.length === 0) {
        container.innerHTML = '<span style="color:var(--text-dim);font-size:0.82rem;">No data</span>';
        return;
    }
    let html = '';
    exercises.forEach(ex => {
        if (ex.active || ex.reps > 0) {
            const phaseColor = ex.phase === 'Down' ? 'var(--accent-orange)' : 'var(--accent-cyan)';
            const colorClass = ex.phase === 'Down' ? 'warn' : 'good'; // For toast type
            
            // Form Feedback Toast
            if (ex.feedback && ex.feedback !== lastExerciseFeedback[ex.label]) {
                let toastType = 'blue'; // Default
                if (ex.feedback.includes("deeper") || ex.feedback.includes("Alert")) toastType = 'warning';
                else if (ex.feedback.includes("Good")) toastType = 'success';
                showToast(`${ex.label}: ${ex.feedback}`, toastType);
                lastExerciseFeedback[ex.label] = ex.feedback;
            }

            html += '<div class="panel-row"><span class="label">' + ex.label +
                    '</span><span class="value" style="color:' + phaseColor + ';">' +
                    ex.reps + ' [' + ex.phase + ']</span></div>';
        }
    });
    if (!html) html = '<span style="color:var(--text-dim);font-size:0.82rem;">Waiting for exercise...</span>';
    container.innerHTML = html;
}

function updateSymmetry(sym) {
    if (!sym) return;
    const overall = document.getElementById('sym-overall');
    const score = sym.overall_score || 0;
    overall.textContent = score.toFixed(0) + '%';
    overall.className = 'value ' + (score >= 85 ? 'good' : score >= 60 ? 'warn' : 'bad');

    const container = document.getElementById('sym-pairs');
    let html = '';
    if (sym.pairs) {
        sym.pairs.forEach(p => {
            if (p.symmetry_pct !== null) {
                html += '<div class="panel-row"><span class="label">' + p.name +
                        '</span><span class="value">' + p.symmetry_pct.toFixed(0) + '%</span></div>';
            }
        });
    }
    container.innerHTML = html;
}

function updateToggles(data) {
    const modules = data.modules || {};
    setActive('btn-pose', modules.pose);
    setActive('btn-hands', modules.hands);
    setActive('btn-face', modules.face);
    setActive('btn-trails', data.trails);
    setActive('btn-graphs', data.graphs);
}

function setActive(id, state) {
    const el = document.getElementById(id);
    if (el) {
        if (state) el.classList.add('active');
        else el.classList.remove('active');
    }
}

// Control functions
function toggle(module) {
    fetch('/api/toggle/' + module, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (module === 'seg') {
                const btn = document.getElementById('btn-seg');
                btn.textContent = 'Seg: ' + data.state;
                btn.classList.toggle('active', data.state !== 'Off');
            }
        });
}

function snapshot() {
    fetch('/api/snapshot', { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.path) alert('Snapshot saved: ' + data.path);
        });
}

function resetExercise() {
    fetch('/api/reset_exercise', { method: 'POST' });
}

function updatePrediction(pred, accuracy) {
    const accSpan = document.getElementById('pred-accuracy');
    if (accuracy !== undefined && accuracy !== null) {
        accSpan.textContent = accuracy.toFixed(1) + '%';
        accSpan.className = 'value ' + (accuracy >= 80 ? 'good' : accuracy >= 50 ? 'warn' : 'bad');
    }

    const container = document.getElementById('prediction-data');
    if (!pred || !pred.length) {
        container.innerHTML = '<span style="color:var(--text-dim);font-size:0.82rem;">Buffering 20-frame sequence...</span>';
        return;
    }
    
    // pred is an array of 5 frames, each frame has 9 angles
    // Indexes: 0: L.Elbow, 1: R.Elbow, 2: L.Shoulder, 3: R.Shoulder, 4: L.Hip, 5: R.Hip, 6: L.Knee, 7: R.Knee, 8: Neck
    const lastFrame = pred[pred.length - 1]; // T+5 prediction
    
    let html = '';
    html += '<div class="panel-row"><span class="label">L.Elbow (T+5)</span><span class="value" style="color:var(--accent-purple);">' + lastFrame[0].toFixed(0) + '&deg;</span></div>';
    html += '<div class="panel-row"><span class="label">R.Elbow (T+5)</span><span class="value" style="color:var(--accent-purple);">' + lastFrame[1].toFixed(0) + '&deg;</span></div>';
    html += '<div class="panel-row"><span class="label">L.Shoulder (T+5)</span><span class="value" style="color:var(--accent-purple);">' + lastFrame[2].toFixed(0) + '&deg;</span></div>';
    html += '<div class="panel-row"><span class="label">R.Shoulder (T+5)</span><span class="value" style="color:var(--accent-purple);">' + lastFrame[3].toFixed(0) + '&deg;</span></div>';
    html += '<div class="panel-row"><span class="label">Neck Incl. (T+5)</span><span class="value" style="color:var(--accent-purple);">' + lastFrame[8].toFixed(0) + '&deg;</span></div>';
    
    container.innerHTML = html;
}
