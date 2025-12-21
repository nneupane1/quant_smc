// hedge_panel.js
// Draws hedge ratio line, net exposure curve, activation markers

window.HedgePanel = (function () {

    let tv = null;

    const ratioSeries = [];
    const exposureSeries = [];
    const activationMarkers = [];

    function init(tvWidget) {
        tv = tvWidget;
    }

    function drawRatioPoint(payload) {
        const { timestamp, ratio } = payload;
        ratioSeries.push({ time: timestamp, value: ratio });

        if (!tv.__hedgeRatioPlot) {
            tv.__hedgeRatioPlot = tv.chart().createMultipointShape(
                ratioSeries.map(p => ({
                    time: p.time,
                    price: p.value
                })),
                {
                    shape: "polyline",
                    color: "#00B0FF",
                    linewidth: 2,
                    lock: true
                }
            );
        } else {
            tv.__hedgeRatioPlot.setPoints(
                ratioSeries.map(p => ({
                    time: p.time,
                    price: p.value
                }))
            );
        }
    }

    function drawExposurePoint(payload) {
        const { timestamp, exposure } = payload;
        exposureSeries.push({ time: timestamp, value: exposure });

        if (!tv.__hedgeExposurePlot) {
            tv.__hedgeExposurePlot = tv.chart().createMultipointShape(
                exposureSeries.map(p => ({
                    time: p.time,
                    price: p.value
                })),
                {
                    shape: "polyline",
                    color: "#FFC400",
                    linewidth: 2,
                    lock: true
                }
            );
        } else {
            tv.__hedgeExposurePlot.setPoints(
                exposureSeries.map(p => ({
                    time: p.time,
                    price: p.value
                }))
            );
        }
    }

    function drawActivation(payload) {
        const { timestamp, action, ratio } = payload;

        activationMarkers.push(
            tv.chart().createShape(
                { time: timestamp, price: ratio },
                {
                    shape: "arrow_up",
                    lock: true,
                    text: action.toUpperCase(),
                    color: "#FF9100",
                    textColor: "#FFFFFF",
                    scale: 1.2
                }
            )
        );
    }

    function dispatch(event) {
        switch (event.type) {
            case "hedge_ratio_point": return drawRatioPoint(event.payload);
            case "hedge_exposure_point": return drawExposurePoint(event.payload);
            case "hedge_activation": return drawActivation(event.payload);
        }
    }

    return { init, dispatch };

})();
