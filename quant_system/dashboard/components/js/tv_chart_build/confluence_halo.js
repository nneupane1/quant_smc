// confluence_halo.js
// Cinematic halo effect behind the most recent candle based on Confluence score.
// Drawn using a custom pane renderer. Zero flicker, theme-compatible.

export function createConfluenceHalo(chart, theme) {
    const halo = {
        _chart: chart,
        _paneId: null,
        _conf: 0.0,
        _time: null,
        _theme: theme,

        setData(confScore, lastTime) {
            this._conf = confScore;
            this._time = lastTime;
            this._chart.paint();
        },

        attach() {
            this._paneId = this._chart.addOverlay({
                id: "confluence-halo-overlay",
                pane: "default",
                draw: (ctx, renderParams) => this._draw(ctx, renderParams),
            });
        },

        _draw(ctx, renderParams) {
            if (!this._time || this._conf <= 0) return;

            const { timeScale, pixelRatio } = renderParams;

            const x = timeScale.timeToCoordinate(this._time);
            if (x === null) return;

            const strength = Math.max(0, Math.min(this._conf, 1.0));

            const radiusBase = 28;
            const radius = radiusBase + strength * 35;

            const gradient = ctx.createRadialGradient(x, 0, 0, x, 0, radius);
            gradient.addColorStop(0.0, this._glowColor(strength));
            gradient.addColorStop(1.0, "rgba(0,0,0,0)");

            ctx.save();
            ctx.globalCompositeOperation = "lighter";
            ctx.fillStyle = gradient;
            ctx.globalAlpha = 0.40 * strength;

            const top = 0;
            const bottom = renderParams.physicalHeight;

            ctx.fillRect(x - radius, top, radius * 2, bottom - top);
            ctx.restore();
        },

        _glowColor(strength) {
            const baseColor = this._theme.panels.confluence.line || "rgba(0,200,255,1)";
            const r = 0;
            const g = Math.floor(160 + 50 * strength);
            const b = 255;
            const a = 0.65 + 0.25 * strength;
            return `rgba(${r},${g},${b},${a})`;
        }
    };

    halo.attach();
    return halo;
}
