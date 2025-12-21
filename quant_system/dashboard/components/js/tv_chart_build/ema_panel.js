// ema_panel.js
// EMA micro-trend and stretch panel, aligned to 15m timeline.

export function createEMAPanel(chartRoot) {
    const panel = document.createElement("div");
    panel.style.width = "100%";
    panel.style.height = "110px";
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
            horzLines: { color: "rgba(255,255,255,0.02)" }
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

    const ema21 = chart.addLineSeries({
        color: "rgba(0, 200, 255, 0.85)",
        lineWidth: 2
    });

    const ema55 = chart.addLineSeries({
        color: "rgba(0, 255, 160, 0.85)",
        lineWidth: 2
    });

    const ema1h50 = chart.addLineSeries({
        color: "rgba(255, 180, 0, 0.70)",
        lineWidth: 2
    });

    const stretchBand = chart.addAreaSeries({
        topColor: "rgba(255,0,0,0.18)",
        bottomColor: "rgba(255,0,0,0.02)",
        lineColor: "transparent",
        lineWidth: 0
    });

    return { chart, ema21, ema55, ema1h50, stretchBand };
}


export function updateEMAPanel(seriesObj, data) {
    if (!data) return;

    const { ema21, ema55, ema1h50, stretchBand } = seriesObj;

    // data.ema15: [{t, ema21, ema55}]
    // data.ema1h: [{t, ema50}]
    // data.stretch: [{t, upper, lower}] (2.5σ bounds)

    if (data.ema15) {
        ema21.setData(data.ema15.map(d => ({ time: d.t, value: d.ema21 })));
        ema55.setData(data.ema15.map(d => ({ time: d.t, value: d.ema55 })));
    }

    if (data.ema1h) {
        ema1h50.setData(data.ema1h.map(d => ({ time: d.t, value: d.ema50 })));
    }

    if (data.stretch) {
        stretchBand.setData(
            data.stretch.map(d => ({
                time: d.t,
                value: d.upper
            }))
        );
        // Actually lightweight-charts wants area defined by value+topColor,
        // so lower band is implied through baseline.
        // No flicker, clean visual.
    }
}
