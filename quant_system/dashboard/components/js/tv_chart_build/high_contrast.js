// high_contrast.js
// Strong contrast for trading floors or accessibility requirements.

export const ThemeConfig = {
    background: "black",
    text: "white",
    grid: "rgba(255,255,255,0.10)",
    border: "rgba(255,255,255,0.20)",

    candles: {
        up: "#00FF77",
        down: "#FF0044",
        wickUp: "#00FF77",
        wickDown: "#FF0044"
    },

    overlays: {
        obDemand: "rgba(0,255,255,0.30)",
        obSupply: "rgba(255,50,50,0.30)",
        fvg: "rgba(190,120,255,0.30)",
        bos: "rgba(0,255,255,1.0)",
        choch: "rgba(255,0,150,1.0)",
        sweep: "rgba(255,255,255,1.0)"
    },

    panels: {
        confluence: {
            line: "rgba(0,255,255,1.0)",
            top: "rgba(0,255,255,0.30)",
            bottom: "rgba(0,255,255,0.06)"
        },
        hazard: {
            line: "rgba(255,0,50,1.0)",
            top: "rgba(255,0,50,0.30)",
            bottom: "rgba(255,0,50,0.06)"
        },
        regimeColors: {
            trend_up: "rgba(0,255,255,0.45)",
            trend_down: "rgba(255,0,90,0.45)",
            range: "rgba(180,180,180,0.35)",
            expansion: "rgba(180,0,255,0.40)",
            collapse: "rgba(255,150,0,0.40)"
        },
        emaColors: {
            ema21: "rgba(0,255,255,1.0)",
            ema55: "rgba(0,255,170,1.0)",
            ema1h50: "rgba(255,255,0,1.0)"
        }
    }
};
