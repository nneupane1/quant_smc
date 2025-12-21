// evr_heatmap.js
// EVR heat strip under the latest candle. Green = positive EVR, Red = negative EVR.

export function createEVRHeatmap(chart) {
    const overlay = {
        _chart: chart,
        _evr: 0.0,
        _time: null,

        setData(evrValue, lastTime) {
            this._evr = evrValue;
            this._time = lastTime;
            this._chart.paint();
        },

        attach() {
            this._chart.addOverlay({
                id: "evr-heatmap",
                pane: "default",
                draw: (ctx, params) => this._draw(ctx, params)
            });
        },

        _draw(ctx, params) {
            if (!this._time) return;

            const x = params.timeScale.timeToCoordinate(this._time);
            if (x === null) return;

            const h = params.physicalHeight;
            const evr = this._evr;

            const color =
                evr >= 0
                    ? `rgba(0,255,120,${Math.min(0.45, Math.abs(evr))})`
                    : `rgba(255,50,50,${Math.min(0.45, Math.abs(evr))})`;

            ctx.save();
            ctx.fillStyle = color;
            ctx.globalCompositeOperation = "source-over";

            ctx.fillRect(x - 10, h - 12, 20, 8);
            ctx.restore();
        }
    };

    overlay.attach();
    return overlay;
}
