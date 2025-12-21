import { createChart } from "lightweight-charts";

(function() {
    let chart = null;
    let candleSeries = null;
    let smcSeries = {
        swings: null,
        bos: null,
        choch: null,
        fvg: null,
        sweeps: null,
        ob_zones: []
    };
    let lastTimestamp = null;

    function initChart() {
        const root = document.getElementById("root");

        chart = createChart(root, {
            width: root.clientWidth,
            height: root.clientHeight,
            layout: {
                background: { color: "transparent" },
                textColor: "#DADADA"
            },
            grid: {
                vertLines: { color: "rgba(255,255,255,0.05)" },
                horzLines: { color: "rgba(255,255,255,0.05)" }
            },
            timeScale: {
                rightOffset: 6,
                borderColor: "rgba(255,255,255,0.18)"
            },
            crosshair: {
                mode: 1,
                vertLine: {
                    color: "rgba(255, 255, 255, 0.25)",
                    width: 1,
                    style: 0
                },
                horzLine: {
                    color: "rgba(255, 255, 255, 0.25)",
                    width: 1,
                    style: 0
                }
            },
            priceScale: {
                borderColor: "rgba(255,255,255,0.15)"
            }
        });

        candleSeries = chart.addCandlestickSeries({
            upColor: "#1FC05D",
            downColor: "#E74C3C",
            borderVisible: false,
            wickUpColor: "#1FC05D",
            wickDownColor: "#E74C3C"
        });

        smcSeries.swings = chart.addLineSeries({
            color: "rgba(255,215,0,0.9)",
            lineWidth: 2
        });

        smcSeries.bos = chart.addLineSeries({
            color: "rgba(0,168,255,0.9)",
            lineWidth: 2,
            lineStyle: 1
        });

        smcSeries.choch = chart.addLineSeries({
            color: "rgba(255,0,128,0.9)",
            lineWidth: 2,
            lineStyle: 2
        });

        smcSeries.fvg = chart.addHistogramSeries({
            color: "rgba(128,0,255,0.25)"
        });

        smcSeries.sweeps = chart.addScatterSeries({
            color: "rgba(255,255,255,0.8)",
            radius: 4
        });

        window.addEventListener("resize", () => {
            chart.applyOptions({
                width: root.clientWidth,
                height: root.clientHeight
            });
        });
    }

    function fadeCandle(c) {
        candleSeries.update({
            time: c.t,
            open: c.o,
            high: c.h,
            low: c.l,
            close: c.c
        });
    }

    function applyFullUpdate(data) {
        if (!chart) initChart();

        const mapped = data.candles.map(c => ({
            time: c.t,
            open: c.o,
            high: c.h,
            low: c.l,
            close: c.c
        }));

        candleSeries.setData(mapped);

        if (mapped.length) {
            lastTimestamp = mapped[mapped.length - 1].time;
        }

        updateSMC(data);
    }

    function applyIncrementalUpdate(data) {
        if (!chart) initChart();

        if (!data.candles || data.candles.length === 0) return;

        const last = data.candles[data.candles.length - 1];

        if (!lastTimestamp || last.t > lastTimestamp) {
            fadeCandle(last);
            lastTimestamp = last.t;
        }

        updateSMC(data);
    }

    function updateSMC(data) {
        if (!data.smc) return;

        if (data.smc.swings) {
            const s = data.smc.swings.map(p => ({ time: p.t, value: p.price }));
            smcSeries.swings.setData(s);
        }

        if (data.smc.bos) {
            const b = data.smc.bos.map(p => ({ time: p.t, value: p.level }));
            smcSeries.bos.setData(b);
        }

        if (data.smc.choch) {
            const c = data.smc.choch.map(p => ({ time: p.t, value: p.level }));
            smcSeries.choch.setData(c);
        }

        if (data.smc.fvg) {
            const f = data.smc.fvg.map(p => ({ time: p.t, value: p.depth }));
            smcSeries.fvg.setData(f);
        }

        if (data.smc.sweeps) {
            const sw = data.smc.sweeps.map(p => ({ time: p.t, value: p.price }));
            smcSeries.sweeps.setData(sw);
        }
    }

    window.tv_chart_update = function(payload) {
        if (!chart) initChart();

        if (lastTimestamp === null) applyFullUpdate(payload);
        else applyIncrementalUpdate(payload);
    };

})();
