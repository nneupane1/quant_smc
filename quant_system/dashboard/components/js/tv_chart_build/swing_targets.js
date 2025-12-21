// swing_targets.js
// ML-predicted swing targets drawn as dotted lines above/below price.

export function createSwingTargets(chart) {
    return {
        _chart: chart,
        _targets: [],

        setData(targetList) {
            this._targets = targetList || [];
            this._chart.paint();
        },

        attach() {
            this._chart.addOverlay({
                id: "swing-targets-overlay",
                pane: "default",
                draw: (ctx, params) => this._draw(ctx, params)
            });
        },

        _draw(ctx, params) {
            if (!this._targets.length) return;

            const { timeScale, priceScale } = params;

            ctx.save();
            ctx.strokeStyle = "rgba(0,200,255,0.85)";
            ctx.lineWidth = 1;
            ctx.setLineDash([6, 4]);

            this._targets.forEach(tgt => {
                const x1 = timeScale.timeToCoordinate(tgt.t_start);
                const x2 = timeScale.timeToCoordinate(tgt.t_end);
                const y = priceScale.priceToCoordinate(tgt.level);

                if (x1 === null || x2 === null || y === null) return;

                ctx.beginPath();
                ctx.moveTo(x1, y);
                ctx.lineTo(x2, y);
                ctx.stroke();
            });

            ctx.restore();
        }
    };
}
