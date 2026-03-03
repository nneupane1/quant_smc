(function() {
    const bars = Array.isArray(PAYLOAD.bars) ? PAYLOAD.bars : [];
    const pnl = Array.isArray(PAYLOAD.pnl) ? PAYLOAD.pnl : [];
    const trades = Array.isArray(PAYLOAD.trades) ? PAYLOAD.trades : [];
    const reasoning = PAYLOAD.reasoning || {};

    let frame = Math.max(bars.length - 1, 0);
    let playing = false;
    let speed = 16;

    const ohlcCanvas = document.getElementById("ohlc-canvas");
    const pnlCanvas = document.getElementById("pnl-canvas");
    const ctxO = ohlcCanvas.getContext("2d");
    const ctxP = pnlCanvas.getContext("2d");
    const slider = document.getElementById("frame-slider");
    const badge = document.getElementById("frame-badge");

    document.getElementById("btn-play").onclick = () => { playing = true; };
    document.getElementById("btn-pause").onclick = () => { playing = false; };
    document.getElementById("btn-back").onclick = () => step(-1);
    document.getElementById("btn-forward").onclick = () => step(1);
    document.getElementById("speed").oninput = (e) => { speed = parseInt(e.target.value, 10) || 16; };
    slider.max = Math.max(bars.length - 1, 0);
    slider.value = frame;
    slider.oninput = (e) => {
        frame = clamp(parseInt(e.target.value, 10) || 0, 0, Math.max(bars.length - 1, 0));
        drawFrame();
    };

    window.onresize = resizeCanvas;
    resizeCanvas();
    drawFrame();
    runLoop();

    function runLoop() {
        if (playing && bars.length > 0) {
            frame += 1;
            if (frame >= bars.length) {
                frame = bars.length - 1;
                playing = false;
            }
            slider.value = frame;
            drawFrame();
        }
        setTimeout(runLoop, 1000 / Math.max(speed, 1));
    }

    function step(offset) {
        if (!bars.length) {
            return;
        }
        frame = clamp(frame + offset, 0, bars.length - 1);
        slider.value = frame;
        drawFrame();
    }

    function drawFrame() {
        drawOHLC();
        drawPnL();
        drawTrades();
        drawReasoning();
        drawStats();
    }

    function drawOHLC() {
        ctxO.clearRect(0, 0, ohlcCanvas.width, ohlcCanvas.height);
        if (!bars.length) {
            drawEmpty(ctxO, ohlcCanvas, "No bar data");
            return;
        }

        const subset = bars.slice(Math.max(0, frame - 180), frame + 1);
        const w = ohlcCanvas.width;
        const h = ohlcCanvas.height;
        const highs = subset.map((b) => num(b.h, b.high));
        const lows = subset.map((b) => num(b.l, b.low));
        const max = Math.max(...highs);
        const min = Math.min(...lows);
        const candleWidth = Math.max(3, Math.floor((w / Math.max(subset.length, 1)) * 0.55));

        subset.forEach((b, i) => {
            const x = ((i + 0.5) * w) / Math.max(subset.length, 1);
            const o = num(b.o, b.open);
            const hVal = num(b.h, b.high);
            const lVal = num(b.l, b.low);
            const c = num(b.c, b.close);
            const openY = mapValue(o, min, max, h - 18, 18);
            const closeY = mapValue(c, min, max, h - 18, 18);
            const highY = mapValue(hVal, min, max, h - 18, 18);
            const lowY = mapValue(lVal, min, max, h - 18, 18);
            const color = c >= o ? "#26f0a8" : "#ff5d78";

            ctxO.strokeStyle = color;
            ctxO.lineWidth = 1.5;
            ctxO.beginPath();
            ctxO.moveTo(x, highY);
            ctxO.lineTo(x, lowY);
            ctxO.stroke();

            ctxO.fillStyle = color;
            ctxO.fillRect(
                x - candleWidth / 2,
                Math.min(openY, closeY),
                candleWidth,
                Math.max(Math.abs(openY - closeY), 1.6)
            );
        });

        const last = subset[subset.length - 1];
        const lastClose = num(last.c, last.close);
        const lastY = mapValue(lastClose, min, max, h - 18, 18);
        ctxO.setLineDash([6, 6]);
        ctxO.strokeStyle = "rgba(77, 203, 255, 0.6)";
        ctxO.beginPath();
        ctxO.moveTo(0, lastY);
        ctxO.lineTo(w, lastY);
        ctxO.stroke();
        ctxO.setLineDash([]);
    }

    function drawPnL() {
        ctxP.clearRect(0, 0, pnlCanvas.width, pnlCanvas.height);
        if (!pnl.length) {
            drawEmpty(ctxP, pnlCanvas, "No equity data");
            return;
        }

        const series = pnl.slice(0, Math.min(frame + 1, pnl.length)).map((row, i) => {
            if (typeof row === "number") {
                return { idx: i, equity: row, drawdown: 0 };
            }
            return {
                idx: i,
                equity: num(row.equity, row.value, row.pnl),
                drawdown: num(row.drawdown),
            };
        });
        const w = pnlCanvas.width;
        const h = pnlCanvas.height;
        const maxEq = Math.max(...series.map((row) => row.equity));
        const minEq = Math.min(...series.map((row) => row.equity));

        ctxP.strokeStyle = "#4fd2ff";
        ctxP.lineWidth = 2;
        ctxP.beginPath();
        series.forEach((row, i) => {
            const x = ((i + 0.5) * w) / Math.max(series.length, 1);
            const y = mapValue(row.equity, minEq, maxEq, h - 14, 14);
            if (i === 0) ctxP.moveTo(x, y);
            else ctxP.lineTo(x, y);
        });
        ctxP.stroke();

        const ddFloor = Math.min(...series.map((row) => row.drawdown));
        if (ddFloor < 0) {
            ctxP.fillStyle = "rgba(255, 93, 120, 0.18)";
            series.forEach((row, i) => {
                if (row.drawdown >= 0) {
                    return;
                }
                const x = ((i + 0.5) * w) / Math.max(series.length, 1);
                const y0 = mapValue(0, ddFloor, 0, h - 10, 10);
                const y1 = mapValue(row.drawdown, ddFloor, 0, h - 10, 10);
                ctxP.fillRect(x - 2, Math.min(y0, y1), 4, Math.abs(y1 - y0));
            });
        }
    }

    function drawTrades() {
        if (!bars.length || !trades.length) {
            return;
        }
        const subset = bars.slice(Math.max(0, frame - 180), frame + 1);
        const highs = subset.map((b) => num(b.h, b.high));
        const lows = subset.map((b) => num(b.l, b.low));
        const max = Math.max(...highs);
        const min = Math.min(...lows);
        const w = ohlcCanvas.width;
        const h = ohlcCanvas.height;
        const startIdx = Math.max(0, frame - 180);

        trades.forEach((trade) => {
            ["entry", "exit"].forEach((kind) => {
                const idx = trade[`${kind}_idx`];
                if (idx === null || idx === undefined || idx < startIdx || idx > frame) {
                    return;
                }
                const rel = idx - startIdx;
                const x = ((rel + 0.5) * w) / Math.max(subset.length, 1);
                const price = num(trade[`${kind}_price`], trade.stop_price);
                const y = mapValue(price, min, max, h - 18, 18);
                const isEntry = kind === "entry";
                const gain = num(trade.pnl) >= 0;
                const color = isEntry ? "#ffe066" : (gain ? "#26f0a8" : "#ff5d78");

                ctxO.fillStyle = color;
                ctxO.beginPath();
                ctxO.arc(x, y, isEntry ? 4.4 : 3.6, 0, Math.PI * 2);
                ctxO.fill();
            });
        });
    }

    function drawReasoning() {
        const panel = document.getElementById("reasoning-panel");
        panel.innerHTML = "";
        if (!bars.length) {
            panel.innerHTML = "<div class='empty-state'>No replay payload.</div>";
            return;
        }

        const bar = bars[frame];
        const info = reasoning[bar.dt] || reasoning[String(bar.timestamp)] || bar.meta || {};
        if (!info || !Object.keys(info).length) {
            panel.innerHTML = "<div class='empty-state'>No reasoning captured for this frame.</div>";
            return;
        }

        Object.entries(info).forEach(([key, value]) => {
            const row = document.createElement("div");
            row.className = "reason-line";
            row.innerHTML = `<span class="reason-key">${escapeHtml(key)}</span><span class="reason-value">${escapeHtml(stringify(value))}</span>`;
            panel.appendChild(row);
        });
    }

    function drawStats() {
        const stats = document.getElementById("stats");
        if (!bars.length) {
            badge.textContent = "0 / 0";
            stats.innerHTML = "<div class='stat'><span>State</span><strong>No data</strong></div>";
            return;
        }

        const bar = bars[frame];
        const pnlRow = pnl[Math.min(frame, Math.max(pnl.length - 1, 0))] || {};
        const activeTrades = trades.filter((trade) => {
            const start = trade.entry_idx ?? Number.MAX_SAFE_INTEGER;
            const end = trade.exit_idx ?? Number.MAX_SAFE_INTEGER;
            return start <= frame && frame <= end;
        });
        badge.textContent = `${frame + 1} / ${bars.length}`;
        stats.innerHTML = [
            statHtml("Time", bar.dt || "n/a"),
            statHtml("Close", formatNum(num(bar.c, bar.close))),
            statHtml("Equity", formatNum(num(pnlRow.equity, pnlRow.value, pnlRow))),
            statHtml("Drawdown", formatNum(num(pnlRow.drawdown))),
            statHtml("Open Trades", String(activeTrades.length)),
        ].join("");
    }

    function drawEmpty(ctx, canvas, label) {
        ctx.fillStyle = "#8ca0b3";
        ctx.font = "14px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "center";
        ctx.fillText(label, canvas.width / 2, canvas.height / 2);
    }

    function statHtml(label, value) {
        return `<div class="stat"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></div>`;
    }

    function num(...vals) {
        for (const val of vals) {
            const n = Number(val);
            if (Number.isFinite(n)) {
                return n;
            }
        }
        return 0;
    }

    function mapValue(v, min, max, outMin, outMax) {
        if (!Number.isFinite(v)) {
            return outMin;
        }
        if (!Number.isFinite(min) || !Number.isFinite(max) || Math.abs(max - min) < 1e-9) {
            return (outMin + outMax) / 2;
        }
        return outMin + ((outMax - outMin) * (v - min)) / (max - min);
    }

    function stringify(value) {
        if (value === null || value === undefined) return "null";
        if (typeof value === "string") return value;
        try {
            return JSON.stringify(value);
        } catch (_) {
            return String(value);
        }
    }

    function formatNum(value) {
        if (!Number.isFinite(value)) {
            return "0.00";
        }
        return value.toLocaleString(undefined, { maximumFractionDigits: 2, minimumFractionDigits: 2 });
    }

    function clamp(v, low, high) {
        return Math.min(Math.max(v, low), high);
    }

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll("\"", "&quot;")
            .replaceAll("'", "&#39;");
    }
})();
