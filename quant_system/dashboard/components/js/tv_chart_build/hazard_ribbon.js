// hazard_ribbon.js
// Hazard ribbon at top of chart. Stronger color == higher hazard.

export function createHazardRibbon(chart) {
    const ribbon = {
        _chart: chart,
        _haz: 0.0,
        _time: null,

        setData(hazardValue, lastTime) {
            this._haz = hazardValue;
            this._time = lastTime;
            this._chart.paint();
        },

        attach() {
            this._chart.addOverlay({
                id: "hazard-ribbon",
                pane: "default",
                draw: (ctx, params) => this._draw(ctx, params)
            });
        },

        _draw(ctx, params) {
            if (!this._time) return;

            const hazard = Math.max(0, Math.min(this._haz, 1.0));
            if (hazard === 0) return;

            const width = params.physicalWidth;
            const height = 10;
            const intensity = 0.15 + hazard * 0.35;

            ctx.save();
            ctx.fillStyle = `rgba(255,0,50,${intensity})`;
            ctx.fillRect(0, 0, width, height);
            ctx.restore();
        }
    };

    ribbon.attach();
    return ribbon;
}
