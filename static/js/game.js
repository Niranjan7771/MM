/**
 * Gesture-Controlled Dodging Game
 *
 * HTML5 Canvas game controlled by body pose via /api/game_state.
 * Player dodges incoming obstacles using lean (move), arms up (jump),
 * and crouch (duck).
 */

const canvas = document.getElementById('gameCanvas');
const ctx = canvas.getContext('2d');

// Game state
let gameRunning = false;
let score = 0;
let frameCount = 0;
let playerX = canvas.width / 2;
let playerY = canvas.height - 80;
let playerW = 40;
let playerH = 60;
let playerAction = 'idle';
let isJumping = false;
let jumpY = 0;
let isDucking = false;
let obstacles = [];
let particles = [];
let groundY = canvas.height - 30;
let gameSpeed = 3;
let lastObstacleFrame = 0;

// Colors
const COLORS = {
    bg: '#ffffff',
    ground: '#f4f6f8',
    groundLine: '#e2e8f0',
    player: '#3182ce',
    playerJump: '#38a169',
    playerDuck: '#dd6b20',
    obstacle: '#e53e3e',
    obstacleHigh: '#805ad5',
    text: '#1a202c',
    score: '#dd6b20',
    particle: '#3182ce',
};

// Poll game state from backend
setInterval(fetchGameState, 100);

function fetchGameState() {
    fetch('/api/game_state')
        .then(r => r.json())
        .then(data => {
            playerAction = data.action || 'idle';

            // Update info panel
            document.getElementById('game-action').textContent =
                playerAction.charAt(0).toUpperCase() + playerAction.slice(1);
            document.getElementById('game-lean').textContent = (data.lean || 0).toFixed(2);
            document.getElementById('game-arms').textContent = data.arms_up ? 'Yes' : 'No';
            document.getElementById('game-crouch').textContent = data.crouching ? 'Yes' : 'No';

            // Apply controls
            if (gameRunning) {
                const centerX = canvas.width / 2;
                const lean = data.lean || 0;

                // Move player based on lean (-1 to 1 mapped to canvas width)
                playerX = centerX + lean * (canvas.width * 0.4);
                playerX = Math.max(playerW / 2, Math.min(canvas.width - playerW / 2, playerX));

                // Jump
                if (data.arms_up && !isJumping) {
                    isJumping = true;
                    jumpY = 0;
                }

                // Duck
                isDucking = data.crouching || false;
            }
        })
        .catch(() => {});
}

function startGame() {
    gameRunning = true;
    score = 0;
    frameCount = 0;
    obstacles = [];
    particles = [];
    gameSpeed = 3;
    playerX = canvas.width / 2;
    isJumping = false;
    jumpY = 0;
    isDucking = false;
    lastObstacleFrame = 0;
    document.getElementById('game-status').textContent = 'Playing!';
    document.getElementById('btn-start').textContent = 'Playing...';
    document.getElementById('btn-start').disabled = true;
    gameLoop();
}

function resetGame() {
    gameRunning = false;
    score = 0;
    obstacles = [];
    particles = [];
    document.getElementById('game-score').textContent = '0';
    document.getElementById('game-status').textContent = 'Press START to play';
    document.getElementById('btn-start').textContent = 'Start Game';
    document.getElementById('btn-start').disabled = false;
    drawIdle();
}

function gameOver() {
    gameRunning = false;
    document.getElementById('game-status').textContent = 'Game Over! Score: ' + score;
    document.getElementById('btn-start').textContent = 'Start Game';
    document.getElementById('btn-start').disabled = false;

    // Explosion particles
    for (let i = 0; i < 20; i++) {
        particles.push({
            x: playerX,
            y: playerY - jumpOffset(),
            vx: (Math.random() - 0.5) * 8,
            vy: (Math.random() - 0.5) * 8,
            life: 30,
        });
    }
}

function jumpOffset() {
    if (!isJumping) return 0;
    // Simple parabolic jump
    return Math.sin(jumpY * Math.PI) * 120;
}

function gameLoop() {
    if (!gameRunning) {
        drawFrame();
        return;
    }

    frameCount++;
    score = Math.floor(frameCount / 6);
    gameSpeed = 3 + Math.floor(frameCount / 300);

    // Update jump
    if (isJumping) {
        jumpY += 0.04;
        if (jumpY >= 1.0) {
            isJumping = false;
            jumpY = 0;
        }
    }

    // Spawn obstacles
    const spawnInterval = Math.max(40, 80 - Math.floor(frameCount / 200));
    if (frameCount - lastObstacleFrame > spawnInterval) {
        spawnObstacle();
        lastObstacleFrame = frameCount;
    }

    // Update obstacles
    for (let i = obstacles.length - 1; i >= 0; i--) {
        obstacles[i].y += gameSpeed;

        // Remove off-screen
        if (obstacles[i].y > canvas.height + 50) {
            obstacles.splice(i, 1);
            continue;
        }

        // Collision detection
        if (checkCollision(obstacles[i])) {
            gameOver();
            break;
        }
    }

    // Update particles
    for (let i = particles.length - 1; i >= 0; i--) {
        particles[i].x += particles[i].vx;
        particles[i].y += particles[i].vy;
        particles[i].life--;
        if (particles[i].life <= 0) {
            particles.splice(i, 1);
        }
    }

    document.getElementById('game-score').textContent = score;
    drawFrame();
    requestAnimationFrame(gameLoop);
}

function spawnObstacle() {
    const type = Math.random() < 0.3 ? 'high' : 'ground';
    const x = Math.random() * (canvas.width - 60) + 30;

    if (type === 'ground') {
        obstacles.push({
            x: x,
            y: -40,
            w: 35 + Math.random() * 20,
            h: 35 + Math.random() * 15,
            type: 'ground',
        });
    } else {
        obstacles.push({
            x: x,
            y: -40,
            w: 50 + Math.random() * 30,
            h: 20,
            type: 'high',
        });
    }
}

function checkCollision(obs) {
    const pJump = jumpOffset();
    const pY = playerY - pJump;
    const pH = isDucking ? playerH * 0.4 : playerH;

    // Simple AABB collision
    return (playerX - playerW / 2 < obs.x + obs.w / 2 &&
            playerX + playerW / 2 > obs.x - obs.w / 2 &&
            pY - pH < obs.y + obs.h / 2 &&
            pY > obs.y - obs.h / 2);
}

function drawFrame() {
    // Background
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Ground
    ctx.fillStyle = COLORS.ground;
    ctx.fillRect(0, groundY, canvas.width, canvas.height - groundY);
    ctx.strokeStyle = COLORS.groundLine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, groundY);
    ctx.lineTo(canvas.width, groundY);
    ctx.stroke();

    // Lane lines
    ctx.strokeStyle = COLORS.groundLine;
    ctx.setLineDash([10, 20]);
    for (let i = 1; i < 4; i++) {
        const lx = (canvas.width / 4) * i;
        ctx.beginPath();
        ctx.moveTo(lx, 0);
        ctx.lineTo(lx, groundY);
        ctx.stroke();
    }
    ctx.setLineDash([]);

    // Obstacles
    obstacles.forEach(obs => {
        ctx.fillStyle = obs.type === 'high' ? COLORS.obstacleHigh : COLORS.obstacle;
        ctx.fillRect(obs.x - obs.w / 2, obs.y - obs.h / 2, obs.w, obs.h);
        // Glow effect
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 10;
        ctx.fillRect(obs.x - obs.w / 2, obs.y - obs.h / 2, obs.w, obs.h);
        ctx.shadowBlur = 0;
    });

    // Player
    const pJump = jumpOffset();
    const pY = playerY - pJump;
    const pH = isDucking ? playerH * 0.4 : playerH;

    let pColor = COLORS.player;
    if (isJumping) pColor = COLORS.playerJump;
    if (isDucking) pColor = COLORS.playerDuck;

    ctx.fillStyle = pColor;
    ctx.shadowColor = pColor;
    ctx.shadowBlur = 15;
    ctx.fillRect(playerX - playerW / 2, pY - pH, playerW, pH);
    ctx.shadowBlur = 0;

    // Player face
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(playerX - 8, pY - pH + 10, 5, 5);
    ctx.fillRect(playerX + 3, pY - pH + 10, 5, 5);

    // Particles
    particles.forEach(p => {
        const alpha = p.life / 30;
        ctx.fillStyle = 'rgba(49, 130, 206, ' + alpha + ')';
        ctx.fillRect(p.x - 3, p.y - 3, 6, 6);
    });

    // Score on canvas
    ctx.fillStyle = COLORS.score;
    ctx.font = 'bold 20px Consolas, monospace';
    ctx.textAlign = 'right';
    ctx.fillText('Score: ' + score, canvas.width - 20, 30);
    ctx.textAlign = 'left';

    // Speed indicator
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Consolas, monospace';
    ctx.fillText('Speed: ' + gameSpeed, 20, 30);
}

function drawIdle() {
    ctx.fillStyle = COLORS.bg;
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = COLORS.ground;
    ctx.fillRect(0, groundY, canvas.width, canvas.height - groundY);

    ctx.fillStyle = COLORS.text;
    ctx.font = 'bold 28px Segoe UI, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Gesture-Controlled Game', canvas.width / 2, canvas.height / 2 - 20);

    ctx.fillStyle = COLORS.score;
    ctx.font = '16px Segoe UI, sans-serif';
    ctx.fillText('Use your body to dodge obstacles!', canvas.width / 2, canvas.height / 2 + 20);
    ctx.textAlign = 'left';
}

// Initial idle screen
drawIdle();
