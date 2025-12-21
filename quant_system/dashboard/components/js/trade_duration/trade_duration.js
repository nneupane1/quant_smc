// trade_duration.js
// Creates duration bars under the candle chart.
// Bars grow as trades stay open. Closed trades freeze their bar.

window.TradeDuration = (function () {

    let tv = null;
    const openBars = {};
    const closedBars = {};

    function init(tvWidget) {
        tv = tvWidget;
    }

    // ---------------------------------------------------------
    // OPEN TRADE DURATION
    // ---------------------------------------------------------
    function drawOpen(payload) {
        const { trade_id, start, now, bars } = payload;

        const color = bars < 20 ? "#4CAF50" :
                      bars < 60 ? "#FFC400" :
                      "#D50000";

        const shapeData = [
            { time: start, price: -1 },
            { time: now, price: -1 }
        ];

        if (openBars[trade_id]) {
            openBars[trade_id].setPoints(shapeData);
        } else {
            openBars[trade_id] = tv.chart().createMultipointShape(
                shapeData,
                {
                    shape: "polyline",
                    color: color,
                    linewidth: 6,
                    lock: true
                }
            );
        }
    }

    // ---------------------------------------------------------
    // CLOSED TRADE DURATION
    // ---------------------------------------------------------
    function drawClosed(payload) {
        const { trade_id, start, end, bars, r_mult } = payload;

        const color = r_mult >= 0 ? "#00E676" : "#FF3D00";

        const shapeData = [
            { time: start, price: -1 },
            { time: end, price: -1 }
        ];

        closedBars[trade_id] = tv.chart().createMultipointShape(
            shapeData,
            {
                shape: "polyline",
                color: color,
                linewidth: 8,
                lock: true
            }
        );

        if (openBars[trade_id]) {
            openBars[trade_id].remove();
            delete openBars[trade_id];
        }
    }

    function dispatch(event) {
        switch (event.type) {
            case "duration_open": return drawOpen(event.payload);
            case "duration_closed": return drawClosed(event.payload);
        }
    }

    return { init, dispatch };

})();
