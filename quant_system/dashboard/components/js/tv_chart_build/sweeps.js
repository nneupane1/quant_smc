// sweeps.js
// Liquidity sweep markers with pulse animation.

export function createSweepSeries(chart) {
    return chart.addScatterSeries({
        color: "rgba(255,255,255,0.90)",
        radius: 4
    });
}

export function updateSweeps(series, sweeps) {
    // sweeps: [{t, price, type}]  type: "high" or "low"
    if (!sweeps || sweeps.length === 0) {
        series.setData([]);
        return;
    }

    const mapped = sweeps.map(p => ({
        time: p.t,
        value: p.price
    }));

    series.setData(mapped);

    // Pulse effect
    const pulse = () => {
        series.applyOptions({ radius: 6 });
        setTimeout(() => series.applyOptions({ radius: 4 }), 140);
    };

    pulse();
}
