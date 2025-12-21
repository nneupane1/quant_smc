// bloomberg.js
// Bloomberg Professional Terminal inspired theme

export const ThemeConfig = {
    background: "rgb(10, 10, 10)",
    text: "rgba(220,220,220,0.90)",
    grid: "rgba(255,255,255,0.03)",
    border: "rgba(255,255,255,0.10)",

    candles: {
        up: "#4BC57A",
        down: "#FF4D4D",
        wickUp: "#4BC57A",
        wickDown: "#FF4D4D"
    },

    overlays: {
        obDemand: "rgba(0,180,255,0.22)",
        obSupply: "rgba(255,80,0,0.22)",
        fvg: "rgba(180,140,255,0.18)",
        bos: "rgba(100,180,255,1.0)",
        choch: "rgba(255,100,160,1.0)",
        sweep: "rgba(255,255,255,0.85)"
    },

    panels: {
        confluence: {
            line: "rgba(100,180,255,1.0)",
            top: "rgba(100,180,255,0.28)",
            bottom: "rgba(100,180,255,0.02)"
        },
        hazard: {
            line: "rgba(255,80,60,0.95)",
            top: "rgba(255,80,60,0.28)",
            bottom: "rgba(255,80,60,0.02)"
        },
        regimeColors: {
            trend_up: "rgba(80,180,255,0.40)",
            trend_down: "rgba(255,90,110,0.40)",
            range: "rgba(200,200,200,0.30)",
            expansion: "rgba(160,120,255,0.35)",
            collapse: "rgba(255,140,0,0.35)"
        },
        emaColors: {
            ema21: "rgba(80,180,255,1.0)",
            ema55: "rgba(100,255,180,1.0)",
            ema1h50: "rgba(255,190,80,0.90)"
        }
    }
};
