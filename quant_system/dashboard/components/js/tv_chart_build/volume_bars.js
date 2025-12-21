// volume_bars.js
// Volume histogram below main price chart.

export function createVolumeSeries(chart) {
    return chart.addHistogramSeries({
        color: "rgba(0, 150, 255, 0.5)",
        priceFormat: { type: "volume" },
        scaleMargins: {
            top: 0.8,
            bottom: 0
        }
    });
}

export function updateVolumeSeries(series, volData) {
    // volData: [{t, volume, direction}]
    // direction: "up" or "down"

    if (!volData || volData.length === 0) {
        series.setData([]);
        return;
    }

    const mapped = volData.map(v => ({
        time: v.t,
        value: v.volume,
        color: v.direction === "up"
            ? "rgba(38, 166, 154, 0.6)"
            : "rgba(239, 83, 80, 0.6)"
    }));

    series.setData(mapped);
}
