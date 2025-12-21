// trade_overlays.js
// Draws trade markers, stop lines, R-trails, moonshot halos
// Integrated with main.js event bus

window.TradeOverlayEngine = (function () {

    let tv = null;
    const entryMarkers = {};
    const stopLines = {};
    const exitMarkers = {};
    const rTrails = {};
    const haloMarkers = {};

    function init(tvWidget) {
        tv = tvWidget;
    }

    // ----------------------------------------------------
    // ENTRY MARKER
    // ----------------------------------------------------
    function drawEntry(payload) {
        const { trade_id, timestamp, price, color, side } = payload;

        entryMarkers[trade_id] = tv.chart().createShape(
            { time: timestamp, price: price },
            {
                shape: "arrow_up",
                lock: true,
                text: `${side.toUpperCase()} ENTRY`,
                color: color,
                textColor: "#FFFFFF",
                scale: 1.2,
            }
        );
    }

    // ----------------------------------------------------
    // STOP LINE
    // ----------------------------------------------------
    function drawStopline(payload) {
        const { trade_id, stop, color } = payload;

        if (stopLines[trade_id]) {
            stopLines[trade_id].remove();
        }

        stopLines[trade_id] = tv.chart().createMultipointShape(
            [
                { price: stop, time: "left" },
                { price: stop, time: "right" }
            ],
            {
                shape: "horizontal_line",
                color: color,
                linewidth: 2,
                lineStyle: 1,
                lock: true
            }
        );
    }

    // ----------------------------------------------------
    // EXIT MARKER
    // ----------------------------------------------------
    function drawExit(payload) {
        const { trade_id, timestamp, price, r_mult, color } = payload;

        exitMarkers[trade_id] = tv.chart().createShape(
            { time: timestamp, price: price },
            {
                shape: "arrow_down",
                lock: true,
                color: color,
                textColor: "#FFFFFF",
                text: `EXIT  (${r_mult.toFixed(2)}R)`,
                scale: 1.2
            }
        );
    }

    // ----------------------------------------------------
    // R-MULTIPLE TRAIL VISUAL
    // ----------------------------------------------------
    function drawRTrail(payload) {
        const { trade_id, series, color } = payload;

        if (rTrails[trade_id]) {
            rTrails[trade_id].remove();
        }

        const polylinePoints = series.map(s => ({
            price: s.price,
            time: s.timestamp
        }));

        rTrails[trade_id] = tv.chart().createMultipointShape(
            polylinePoints,
            {
                shape: "polyline",
                linewidth: 2,
                color: color,
                lock: true
            }
        );
    }

    // ----------------------------------------------------
    // MOONSHOT HALO
    // ----------------------------------------------------
    function drawHalo(payload) {
        const { trade_id, timestamp, price, glow_color, radius } = payload;

        haloMarkers[trade_id] = tv.chart().createShape(
            { time: timestamp, price: price },
            {
                shape: "circle",
                lock: true,
                color: glow_color,
                fillColor: glow_color,
                scale: radius / 10,
                text: "",
            }
        );
    }

    // ----------------------------------------------------
    // PUBLIC DISPATCH
    // ----------------------------------------------------
    function dispatch(event) {
        const type = event.type;
        const payload = event.payload;

        switch (type) {
            case "entry_marker": return drawEntry(payload);
            case "stop_line": return drawStopline(payload);
            case "exit_marker": return drawExit(payload);
            case "r_trail": return drawRTrail(payload);
            case "moonshot_halo": return drawHalo(payload);
        }
    }

    return {
        init,
        dispatch
    };

})();
