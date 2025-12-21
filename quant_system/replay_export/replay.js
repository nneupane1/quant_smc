// replay.js
// Full animation engine for standalone HTML replay export.
// Renders:
//  • OHLC bars
//  • PnL curve
//  • Trades
//  • Reasoning per frame

(function() {

    let bars = PAYLOAD.bars;
    let pnl = PAYLOAD.pnl;
    let trades = PAYLOAD.trades;
    let reasoning = PAYLOAD.reasoning;

    let frame = 0;
    let playing = false;
    let speed = 24; // FPS

    const ohlcCanvas = document.getElementById("ohlc-canvas");
    const pnlCanvas = document.getElementById("pnl-canvas");
    const ctxO = ohlcCanvas.getContext("2d");
    const ctxP = pnlCanvas.getContext("2d");

    resizeCanvas();

    document.getElementById("btn-play").onclick = () => playing = true;
    document.getElementById("btn-pause").onclick = () => playing = false;
    document.getElementById("btn-back").onclick = () => step(-1);
    document.getElementById("btn-forward").onclick = () => step(1);

    document.getElementById("speed").oninput = (e) => {
        speed = parseInt(e.target.value);
    };

    const slider = document.getElementById("frame-slider");
    slider.max = bars.length - 1;
    slider.oninput = (e) => {
        frame = parseInt(e.target.value);
        drawFrame();
    };

    window.onresize = resizeCanvas;

    runLoop();

    // MAIN LOOP -------------------------------------------------------
    function runLoop() {
        if (playing) {
            frame += 1;
            if (frame >= bars.length) {
                frame = bars.length - 1;
                playing = false;
            }
            slider.value = frame;
            drawFrame();
        }
        setTimeout(runLoop, 1000 / speed);
    }

    // ACTIONS ----------------------------------------------------------
    function step(offset) {
        frame += offset;
        if (frame < 0) frame = 0;
        if (frame >= bars.length) frame = bars.length - 1;
        slider.value = frame;
        drawFrame();
    }

    // DRAWING ----------------------------------------------------------
    function drawFrame() {
        drawOHLC();
        drawPnL();
        drawReasoning();
    }

    function drawOHLC() {
        ctxO.clearRect(0, 0, ohlcCanvas.width, ohlcCanvas.height);

        const subset = bars.slice(Math.max(0, frame - 200), frame + 1);
        const w = ohlcCanvas.width;
        const h = ohlcCanvas.height;

        const xs = subset.map((_, i) => i * (w / subset.length));
        const highs = subset.map(b => b.h);
        const lows = subset.map(b => b.l);

        const max = Math.max(...highs);
        const min = Math.min(...lows);

        for (let i = 0; i < subset.length; i++) {
            const b = subset[i];
            const x = xs[i];

            const openY = map(b.o, min, max, h, 0);
            const closeY = map(b.c, min, max, h, 0);
            const highY = map(b.h, min, max, h, 0);
            const lowY = map(b.l, min, max, h, 0);

            const color = b.c >= b.o ? "#40C060" : "#D04040";

            // Wick
            ctxO.strokeStyle = color;
            ctxO.beginPath();
            ctxO.moveTo(x, highY);
            ctxO.lineTo(x, lowY);
            ctxO.stroke();

            // Body
            ctxO.fillStyle = color;
            ctxO.fillRect(x - 2, Math.min(openY, closeY),
                          4, Math.abs(openY - closeY));
        }
    }

    function drawPnL() {
        ctxP.clearRect(0, 0, pnlCanvas.width, pnlCanvas.height);

        const subset = pnl.slice(0, frame + 1);
        const w = pnlCanvas.width;
        const h = pnlCanvas.height;

        const xs = subset.map((_, i) => i * (w / subset.length));
        const ys = subset.map(v => v);

        const max = Math.max(...ys);
        const min = Math.min(...ys);

        ctxP.strokeStyle = "#3FA9F5";
        ctxP.beginPath();

        xs.forEach((x, i) => {
            const y = map(ys[i], min, max, h, 0);
            if (i === 0) ctxP.moveTo(x, y);
            else ctxP.lineTo(x, y);
        });

        ctxP.stroke();
    }

    function drawReasoning() {
        const dt = bars[frame].dt;
        const info = reasoning[dt];

        const panel = document.getElementById("reasoning-panel");
        panel.innerHTML = "";

        if (!info) {
            panel.innerHTML = "<div class='reason-line'>No reasoning.</div>";
            return;
        }

        Object.keys(info).forEach(k => {
            const val = info[k];
            const div = document.createElement("div");
            div.className = "reason-line";
            div.innerHTML = `<span class="reason-key">${k}</span>: 
                             <span class="reason-value">${JSON.stringify(val)}</span>`;
            panel.appendChild(div);
        });
    }

    // UTILS -----------------------------------------------------------
    function resizeCanvas() {
        ohlcCanvas.width = document.getElementById("chart-area").clientWidth;
        ohlcCanvas.height = document.getElementById("chart-area").clientHeight * 0.65;

        pnlCanvas.width = document.getElementById("chart-area").clientWidth;
        pnlCanvas.height = document.getElementById("chart-area").clientHeight * 0.35;

        drawFrame();
    }

    function map(v, min, max, outMin, outMax) {
        return outMin + (outMax - outMin) * (v - min) / (max - min + 1e-9);
    }

})();
