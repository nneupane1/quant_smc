// confluence_panel.js
// High-aesthetic Confluence Score panel (0–1 scale) stacked under main chart.

export function createConfluencePanel(chartRoot) {
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

    const confSeries = chart.addAreaSeries({
        lineColor: "rgba(0, 200, 255, 0.95)",
        topColor: "rgba(0,200,255,0.25)",
        bottomColor: "rgba(0,200,255,0.02)",
        lineWidth: 2
    });

    const aLine = chart.addLineSeries({
        color: "rgba(255,255,255,0.15)",
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted
    });

    const aplusLine = chart.addLineSeries({
        color: "rgba(0,255,180,0.25)",
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted
    });

    return { chart, confSeries, aLine, aplusLine };
}


export function updateConfluencePanel(seriesObj, data) {
    if (!data) return;

    const { confSeries, aLine, aplusLine } = seriesObj;

    // data.conf: [{t, value}]
    // data.thresholds: {a, aplus}

    if (data.conf && data.conf.length > 0) {
        confSeries.setData(
            data.conf.map(p => ({ time: p.t, value: p.value }))
        );
    }

    if (data.thresholds) {
        const thresholdDataA = data.conf.map(p => ({
            time: p.t,
            value: data.thresholds.a
        }));

        const thresholdDataAPlus = data.conf.map(p => ({
            time: p.t,
            value: data.thresholds.aplus
        }));

        aLine.setData(thresholdDataA);
        aplusLine.setData(thresholdDataAPlus);
    }
}
