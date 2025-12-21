// orderblocks.js
// Animated Order Block overlays for lightweight-charts.

export function createOrderBlockSeries(chart) {
    return [];
}

export function updateOrderBlocks(obSeriesList, chart, blocks) {
    // blocks = [{t_start, t_end, price_high, price_low, type}]
    // type: "demand" | "supply"

    // Clear existing blocks
    obSeriesList.forEach(s => chart.removeSeries(s));
    obSeriesList.length = 0;

    if (!blocks || blocks.length === 0) return;

    blocks.forEach(block => {
        const series = chart.addAreaSeries({
            topColor: block.type === "demand"
                ? "rgba(0, 180, 255, 0.28)"
                : "rgba(255, 0, 80, 0.28)",
            bottomColor: block.type === "demand"
                ? "rgba(0, 180, 255, 0.05)"
                : "rgba(255, 0, 80, 0.05)",
            lineColor: block.type === "demand"
                ? "rgba(0, 180, 255, 0.9)"
                : "rgba(255, 0, 80, 0.9)",
            lineWidth: 1
        });

        const data = [
            { time: block.t_start, value: block.price_low },
            { time: block.t_end, value: block.price_low },
            { time: block.t_end, value: block.price_high },
            { time: block.t_start, value: block.price_high }
        ];

        series.setData(data);
        obSeriesList.push(series);
    });
}
