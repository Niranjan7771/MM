/**
 * Sign Language Page - Frontend Logic
 *
 * Polls /api/sign_language every 300ms for detected letter + sentence.
 */

setInterval(fetchSign, 300);

function fetchSign() {
    fetch('/api/sign_language')
        .then(r => r.json())
        .then(data => {
            const letterEl = document.getElementById('detected-letter');
            const confEl = document.getElementById('sign-confidence');
            const sentenceEl = document.getElementById('sentence-display');

            if (data.letter) {
                letterEl.textContent = data.letter;
                letterEl.style.color = 'var(--accent-cyan)';
            } else {
                letterEl.textContent = '?';
                letterEl.style.color = 'var(--text-dim)';
            }

            const conf = (data.confidence || 0) * 100;
            confEl.textContent = conf.toFixed(0) + '%';
            confEl.className = 'value ' + (conf > 70 ? 'good' : conf > 40 ? 'warn' : '');

            const sentence = data.sentence || '';
            sentenceEl.textContent = sentence ? sentence + '_' : '_';
        })
        .catch(() => {});
}

function signSpace() {
    fetch('/api/sign/space', { method: 'POST' });
}

function signBackspace() {
    fetch('/api/sign/backspace', { method: 'POST' });
}

function signClear() {
    fetch('/api/sign/clear', { method: 'POST' });
}
