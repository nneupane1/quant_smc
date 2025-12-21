// vwap_bands.js
// VWAP line + standard deviation bands.

export function createVWAPSeries(chart) {
    return {
        vwap: chart.addLineSeries({
            color: "rgba(255,255,0,0.9)",
            lineWidth: 2
        }),
        upper: chart.addLineSeries({
            color: "rgba(255,255,0,0.4)",
            lineWidth: 1
        }),
        lower: chart.addLineSeries({
            color: "rgba(255,255,0,0.4)",
            lineWidth: 1
        })
    };
}

export function updateVWAPSeries(series, vwapData) {
    // vwapData: [{t, vwap, upper, lower}]
    if (!vwapData || vwapData.length === 0) {
        series.vwap.setData([]);
        series.upper.setData([]);
        series.lower.setData([]);
        return;
    }

    series.vwap.setData(
        vwapData.map(p => ({ time: p.t, value: p.vwap }))
    );

    series.upper.setData(
        vwapData.map(p => ({ time: p.t, value: p.upper }))
    );

    series.lower.setData(
        vwapData.map(p => ({ time: p.t, value: p.lower }))
    );
}
