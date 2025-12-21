// fvg_zones.js
// Fair Value Gap (FVG) zones with fade-in and translucent highlight.

export function createFVGSeries(chart) {
    return [];
}

export function updateFVGZones(fvgSeriesList, chart, fvgs) {
    // fvgs: [{t_start, t_end, upper, lower}]
    // upper/lower = price boundaries of the gap

    // Remove existing FVG zones
    fvgSeriesList.forEach(s => chart.removeSeries(s));
    fvgSeriesList.length = 0;

    if (!fvgs || fvgs.length === 0) return;

    fvgs.forEach(zone => {
        const series = chart.addAreaSeries({
            topColor: "rgba(128, 0, 255, 0.20)",
            bottomColor: "rgba(128, 0, 255, 0.05)",
            lineColor: "rgba(128, 0, 255, 0.60)",
            lineWidth: 1
        });

        // Represent FVG as a vertical price block between two times.
        const data = [
            { time: zone.t_start, value: zone.lower },
            { time: zone.t_end, value: zone.lower },
            { time: zone.t_end, value: zone.upper },
            { time: zone.t_start, value: zone.upper }
        ];

        series.setData(data);
        fvgSeriesList.push(series);
    });
}
