import { createChart, LineStyle } from "lightweight-charts";

import { createOrderBlockSeries, updateOrderBlocks } from "./overlays/orderblocks.js";
import { createFVGSeries, updateFVGZones } from "./overlays/fvg_zones.js";
import { createSweepSeries, updateSweeps } from "./overlays/sweeps.js";
import { createBOSCHOCHSeries, updateBOSCHOCH } from "./overlays/bos_choch.js";
import { createVWAPSeries, updateVWAPSeries } from "./overlays/vwap_bands.js";
import { createVolumeSeries, updateVolumeSeries } from "./overlays/volume_bars.js";

import { createConfluencePanel, updateConfluencePanel } from "./panels/confluence_panel.js";
import { createHazardPanel, updateHazardPanel } from "./panels/hazard_panel.js";
import { createRegimePanel, updateRegimePanel } from "./panels/regime_panel.js";
import { createEMAPanel, updateEMAPanel } from "./panels/ema_panel.js";

import { createConfluenceHalo } from "./indicators/confluence_halo.js";
import { createEVRHeatmap } from "./indicators/evr_heatmap.js";
import { createHazardRibbon } from "./indicators/hazard_ribbon.js";
import { createSwingTargets } from "./indicators/swing_targets.js";

import { loadTheme } from "./themes/theme_loader.js";

(function () {
    let chart = null;
    let candleSeries = null;
    let volumeSeries = null;
    let theme = null;

    let obSeriesList = [];
    let fvgSeriesList = [];
    let sweepSeries = null;
    let boschochSeries = null;
    let vwapSeries = null;

    let halo = null;
    let evrHeat = null;
    let hazRibbon = null;
    let swingTgt = null;

    let confPanel = null;
    let hazPanel = null;
    let regPanel = null;
    let emaPanel = null;

    let lastTimestamp = null;

    function initChart(payloadTheme) {
        const root = document.getElementById("root");
        theme = loadTheme(payloadTheme || "dark");

        chart = createChart(root, {
            width: root.clientWidth,
            height: 480,
            layout: {
                background: { color: theme.background },
                textColor: theme.text,
                fontSize: 12,
                fontFamily: "Inter, system-ui, sans-serif"
            },
            grid: {
                vertLines: { color: theme.grid },
                horzLines: { color: theme.grid }
            },
            timeScale: {
                rightOffset: 6,
                borderColor: theme.border
            },
            priceScale: {
                borderColor: theme.border
            },
            crosshair: {
                mode: 1,
                vertLine: { color: "rgba(255,255,255,0.35)" },
                horzLine: { color: "rgba(255,255,255,0.35)" }
            }
        });

        candleSeries = chart.addCandlestickSeries({
            upColor: theme.candles.up,
            downColor: theme.candles.down,
            wickUpColor: theme.candles.wickUp,
            wickDownColor: theme.candles.wickDown,
            borderVisible: false
        });

        volumeSeries = createVolumeSeries(chart);
        sweepSeries = createSweepSeries(chart);
        boschochSeries = createBOSCHOCHSeries(chart);
        vwapSeries = createVWAPSeries(chart);

        halo = createConfluenceHalo(chart, theme);
        evrHeat = createEVRHeatmap(chart);
        hazRibbon = createHazardRibbon(chart);
        swingTgt = createSwingTargets(chart);
        swingTgt.attach();

        const panelsRoot = document.getElementById("panels");

        confPanel = createConfluencePanel(panelsRoot);
        hazPanel = createHazardPanel(panelsRoot);
        regPanel = createRegimePanel(panelsRoot);
        emaPanel = createEMAPanel(panelsRoot);

        window.addEventListener("resize", () => {
            chart.applyOptions({
                width: root.clientWidth,
                height: 480
            });
        });
    }

    function fullUpdateCandles(candles) {
        const mapped = candles.map(c => ({
            time: c.t, open: c.o, high: c.h, low: c.l, close: c.c
        }));

        candleSeries.setData(mapped);
        if (mapped.length) lastTimestamp = mapped[mapped.length - 1].time;
    }

    function incrementalUpdate(candles) {
        if (!candles.length) return;

        const last = candles[candles.length - 1];
        if (!lastTimestamp || last.t > lastTimestamp) {
            candleSeries.update({
                time: last.t,
                open: last.o,
                high: last.h,
                low: last.l,
                close: last.c
            });
            lastTimestamp = last.t;
        }
    }

    function updateOverlays(payload) {
        if (payload.volume) updateVolumeSeries(volumeSeries, payload.volume);
        if (payload.smc) {
            updateOrderBlocks(obSeriesList, chart, payload.smc.orderblocks || []);
            updateFVGZones(fvgSeriesList, chart, payload.smc.fvg || []);
            updateSweeps(sweepSeries, payload.smc.sweeps || []);
            updateBOSCHOCH(boschochSeries, payload.smc.bos || [], payload.smc.choch || []);
        }
        if (payload.vwap) updateVWAPSeries(vwapSeries, payload.vwap);
    }

    function updatePanels(payload) {
        if (payload.ml) {
            updateConfluencePanel(confPanel, {
                conf: payload.ml.conf_series,
                thresholds: payload.ml.conf_thresholds
            });
        }
        if (payload.ml) {
            updateHazardPanel(hazPanel, { hazard: payload.ml.hazard_series });
        }
        if (payload.regime) {
            updateRegimePanel(regPanel, { regime: payload.regime });
        }
        if (payload.ema_panel) {
            updateEMAPanel(emaPanel, payload.ema_panel);
        }
    }

    function updateIndicators(payload) {
        if (payload.ml) {
            halo.setData(payload.ml.conf_latest, lastTimestamp);
            evrHeat.setData(payload.ml.evr_latest, lastTimestamp);
            hazRibbon.setData(payload.ml.hazard_latest, lastTimestamp);
            swingTgt.setData(payload.ml.swing_targets || []);
        }
    }

    function applyPayload(payload) {
        if (!chart) initChart(payload.theme);

        if (lastTimestamp === null) {
            fullUpdateCandles(payload.candles);
        } else {
            incrementalUpdate(payload.candles);
        }

        updateOverlays(payload);
        updatePanels(payload);
        updateIndicators(payload);
    }

    window.tv_chart_update = function (payload) {
        applyPayload(payload);
    };
})();
