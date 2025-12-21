// footprint.js
// Draws bid/ask volume clusters on each candle.

window.FootprintOverlay = (function () {

    let tv = null;
    const footprintShapes = {};

    function init(tvWidget) {
        tv = tvWidget;
    }

    function render(payload) {
        const ts = payload.timestamp;
        const levels = payload.levels;

        if (footprintShapes[ts]) {
            footprintShapes[ts].remove();
        }

        const points = [];
        const colors = [];

        Object.keys(levels).forEach(price => {
            const lvl = levels[price];
            const bid = lvl.bid || 0;
            const ask = lvl.ask || 0;

            const heat = Math.min((bid + ask) / 200, 1);
            const color = `rgba(255,255,0,${heat})`;

            points.push({ time: ts, price: parseFloat(price) });
            colors.push(color);
        });

        footprintShapes[ts] = tv.chart().createMultipointShape(
            points,
            {
                shape: "circle",
                color: "#fff",
                backgroundColor: colors[0] || "rgba(255,255,0,0.3)",
                disableSelection: true,
                disableSave: true
            }
        );
    }

    function dispatch(event) {
        if (event.type === "footprint_update") {
            render(event.payload);
        }
    }

    return { init, dispatch };

})();
