// replay.js
// TradingView Replay Engine for Backtests
// Responsible for chart creation, overlays, markers, stepping animation,
// and real-time metadata updates.

let chart;
let candleSeries;
let volumeSeries;

// Overlay series
let obSeries = [];      // Orderblocks (rectangles)
let fvgSeries = [];
let sweepSeries = [];
let bosSeries = [];
let chocSeries = [];
let markerSeries;

// Replay state
let replayData = [];
let ptr = 0;
let isPlaying = false;
let playInterval = null;

const MAX_CANDLES = 500;  // visible window size


// -------------------------------------------------------------
// Chart Initialization
// -------------------------------------------------------------
function initChart() {
    const chartEl = document.getElementById('chart');

    chart = LightweightCharts.createChart(chartEl, {
        layout: {
            background: { color: '#0D0D0F' },
            textColor: '#DDD',
        },
        grid: {
            vertLines: { color: '#1C1C1E' },
            horzLines: { color: '#1C1C1E' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        timeScale: {
            rightOffset: 6,
            barSpacing: 8,
            fixLeftEdge: false,
            fixRightEdge: false,
        },
        priceScale: {
            borderVisible: false,
        }
    });

    candleSeries = chart.addCandlestickSeries({
        upColor: "#26A69A",
        downColor: "#EF5350",
        wickUpColor: "#26A69A",
        wickDownColor: "#EF5350",
    });

    volumeSeries = chart.addHistogramSeries({
        priceFormat: { type: "volume" },
        scaleMargins: { top: 0.75, bottom: 0 },
        color: "rgba(0, 150, 255, 0.4)"
    });

    markerSeries = chart.addLineSeries({
        color: "transparent",
    });
}

initChart();


// -------------------------------------------------------------
// Main API Called by Streamlit
// -------------------------------------------------------------

// Load new payload
window.replay_load = function(payload_json) {
    const payload = JSON.parse(payload_json);
    replayData = replayData || [];

    // append new candle to timeline
    replayData.push(payload);
    ptr = replayData.length - 1;

    renderFrame(payload);
};

// Jump directly to a time
window.replay_jump = function(timestamp) {
    const idx = replayData.findIndex(x => x.candle.time >= timestamp);
    if (idx >= 0) {
        ptr = idx;
        renderFrame(replayData[idx]);
    }
};

// Next candle
window.replay_next = function() {
    if (ptr < replayData.length - 1) {
        ptr++;
        renderFrame(replayData[ptr]);
    }
};

// Previous candle
window.replay_prev = function() {
    if (ptr > 0) {
        ptr--;
        renderFrame(replayData[ptr]);
    }
};

// Autoplay
window.replay_play = function() {
    if (isPlaying) {
        stopPlay();
        return;
    }
    isPlaying = true;

    playInterval = setInterval(() => {
        if (ptr < replayData.length - 1) {
            ptr++;
            renderFrame(replayData[ptr]);
        } else {
            stopPlay();
        }
    }, 450); // speed (ms)
};

function stopPlay() {
    isPlaying = false;
    if (playInterval) clearInterval(playInterval);
}


// -------------------------------------------------------------
// Rendering Logic
// -------------------------------------------------------------
function renderFrame(frame) {
    if (!frame || !frame.candle) return;

    const c = frame.candle;
    const time = c.time;

    // 1) Append candle
    candleSeries.update({
        time: time,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close
    });

    volumeSeries.update({
        time: time,
        value: c.volume,
        color: c.close >= c.open ?
            "rgba(38,166,154,0.6)" : "rgba(239,83,80,0.6)"
    });

    // 2) Manage visible window
    trimOldCandles();

    // 3) Render overlays
    renderSMC(frame.smc);
    renderTrades(frame.trade);

    // 4) Update info panel
    renderInfoPanel(frame.meta);

    // 5) Autoscroll to latest
    chart.timeScale().scrollToRealTime();
}


// -------------------------------------------------------------
// Trim window to avoid huge chart memory
// -------------------------------------------------------------
function trimOldCandles() {
    const len = replayData.length;
    if (len <= MAX_CANDLES) return;

    const startIdx = len - MAX_CANDLES;
    const sliced = replayData.slice(startIdx);

    const candles = sliced.map(x => ({
        time: x.candle.time,
        open: x.candle.open,
        high: x.candle.high,
        low: x.candle.low,
        close: x.candle.close,
    }));

    candleSeries.setData(candles);
}


// -------------------------------------------------------------
// Render SMC overlays
// -------------------------------------------------------------
function clearOverlaySeries() {
    obSeries.forEach(s => chart.removeSeries(s));
    fvgSeries.forEach(s => chart.removeSeries(s));
    sweepSeries.forEach(s => chart.removeSeries(s));
    bosSeries.forEach(s => chart.removeSeries(s));
    chocSeries.forEach(s => chart.removeSeries(s));

    obSeries = [];
    fvgSeries = [];
    sweepSeries = [];
    bosSeries = [];
    chocSeries = [];
}

function renderSMC(smc) {
    clearOverlaySeries();

    // ORDERBLOCKS
    smc.orderblocks.forEach(ob => {
        let s = chart.addLineSeries({ color: ob.type === "demand" ? "#4CAF50AA" : "#F44336AA" });
        s.setData([
            { time: ob.start, value: ob.price_low },
            { time: ob.end, value: ob.price_low }
        ]);
        obSeries.push(s);
    });

    // FVG ZONES
    smc.fvg.forEach(fvg => {
        let s = chart.addLineSeries({ color: "rgba(255,215,0,0.7)" });
        s.setData([
            { time: fvg.start, value: fvg.low },
            { time: fvg.end, value: fvg.low }
        ]);
        fvgSeries.push(s);
    });

    // SWEEPS
    smc.sweeps.forEach(sw => {
        let s = chart.addLineSeries({ color: "rgba(79,195,247,1)" });
        s.setData([
            { time: sw.time, value: sw.price },
        ]);
        sweepSeries.push(s);
    });

    // BOS
    smc.bos_choch.filter(x => x.type === "BOS").forEach(sig => {
        let s = chart.addLineSeries({ color: "#00E676" });
        s.setData([{ time: sig.time, value: sig.price }]);
        bosSeries.push(s);
    });

    // CHOCH
    smc.bos_choch.filter(x => x.type === "CHOCH").forEach(sig => {
        let s = chart.addLineSeries({ color: "#FF4081" });
        s.setData([{ time: sig.time, value: sig.price }]);
        chocSeries.push(s);
    });
}


// -------------------------------------------------------------
// Render Trade Markers
// -------------------------------------------------------------
function renderTrades(trade) {
    let markers = [];

    trade.entries.forEach(e => {
        markers.push({
            time: replayData[ptr].candle.time,
            position: 'belowBar',
            color: "#00E676",
            shape: "arrowUp",
            text: `Entry ${e.side.toUpperCase()}`
        });
    });

    trade.exits.forEach(ex => {
        markers.push({
            time: replayData[ptr].candle.time,
            position: 'aboveBar',
            color: "#FFEB3B",
            shape: "arrowDown",
            text: `Exit (${ex.reason})`
        });
    });

    trade.stops.forEach(s => {
        markers.push({
            time: replayData[ptr].candle.time,
            position: 'aboveBar',
            color: "#FF5252",
            shape: "square",
            text: "STOP"
        });
    });

    trade.hedge.forEach(h => {
        markers.push({
            time: replayData[ptr].candle.time,
            position: 'aboveBar',
            color: "#29B6F6",
            shape: "circle",
            text: `Hedge ${h.ratio}`
        });
    });

    markerSeries.setMarkers(markers);
}


// -------------------------------------------------------------
// Info Panel (Confluence / EVR / Hazard / Risk Mode)
// -------------------------------------------------------------
function renderInfoPanel(meta) {
    document.getElementById("conf-value").innerText = meta.conf.toFixed(3);
    document.getElementById("evr-value").innerText = meta.evr.toFixed(3);
    document.getElementById("hazard-value").innerText = meta.hazard.toFixed(3);

    const r = meta.risk;
    document.getElementById("risk-value").innerText =
        `Lock ${r.lock_pct}%, Mode ${r.risk_mode}, Hedge ${r.hedge_ratio}`;
}


// -------------------------------------------------------------
// Bind Buttons
// -------------------------------------------------------------
document.getElementById("btn-next").onclick = () => window.replay_next();
document.getElementById("btn-prev").onclick = () => window.replay_prev();
document.getElementById("btn-play").onclick = () => window.replay_play();
