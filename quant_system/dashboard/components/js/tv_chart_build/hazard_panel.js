// hazard_panel.js
// Hazard probability panel (0–1 scale) stacked under main chart.

export function createHazardPanel(chartRoot) {
    const panel = document.createElement("div");
    panel.style.width = "100%";
    panel.style.height = "120px";
    panel.style.marginTop = "6px";
    panel.style.borderTop = "1px solid rgba(255,255,255,0.08)";

    chartRoot.appendChild(panel);

    const chart = LightweightCharts.createChart(panel, {
        layout: {
            background: { color: "transparent" },
            textColor: "rgba(230,230,230,0.85)",
            fontSize: 12,
            fontFamily: "Inter, system-ui, sans-serif"
        },
        grid: {
            vertLines: { color: "rgba(255,255,255,0.02)" },
            horzLines: { color: "rgba(255,255,255,0.03)" }
        },
        timeScale: {
            visible: false,
            borderColor: "rgba(255,255,255,0.05)"
        },
        rightPriceScale: {
            autoScale: true,
            borderColor: "rgba(255,255,255,0.05)"
        }
    });

    const hazardSeries = chart.addAreaSeries({
        lineColor: "rgba(255,0,90,0.9)",
        topColor: "rgba(255,0,90,0.25)",
        bottomColor: "rgba(255,0,90,0.02)",
        lineWidth: 2
    });

    // Glow/high-risk band
    const hazardGlow = chart.addHistogramSeries({
        priceFormat: { type: "price", precision: 2 },
        color: "rgba(255, 0, 0, 0.12)",
        scaleMargins: { top: 0.85, bottom: 0 }
    });

    return { chart, hazardSeries, hazardGlow };
}


export function updateHazardPanel(seriesObj, data) {
    if (!data) return;

    const { hazardSeries, hazardGlow } = seriesObj;

    // data.hazard: [{t, value}]
    // Higher hazard gets a stronger glow

    if (data.hazard && data.hazard.length > 0) {
        hazardSeries.setData(
            data.hazard.map(p => ({ time: p.t, value: p.value }))
        );

        const glowData = data.hazard.map(p => ({
            time: p.t,
            value: p.value,
            color:
                p.value > 0.6
                    ? "rgba(255,0,0,0.30)"
                    : p.value > 0.4
                    ? "rgba(255,0,0,0.18)"
                    : "rgba(255,0,0,0.08)"
        }));

        hazardGlow.setData(glowData);
    }
}
