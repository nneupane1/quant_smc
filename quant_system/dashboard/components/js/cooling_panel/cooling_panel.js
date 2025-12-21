// cooling_panel.js
// Circular countdown ring with color transitions
// Handles events: cooling_update, cooling_off

window.CoolingPanel = (function () {

    let container = null;
    let ring = null;
    let label = null;
    let timer = null;

    function init(targetElementId) {
        container = document.getElementById(targetElementId);

        if (!container) return;

        container.innerHTML = `
            <div id="cooling-circle">
                <svg id="cool-svg" width="120" height="120">
                    <circle id="cool-bg" cx="60" cy="60" r="50" stroke="#222" stroke-width="8" fill="none" />
                    <circle id="cool-fg" cx="60" cy="60" r="50" stroke="#4CAF50" stroke-width="8"
                            fill="none" stroke-linecap="round"
                            stroke-dasharray="314" stroke-dashoffset="0" />
                </svg>
                <div id="cool-label">READY</div>
            </div>
        `;

        ring = document.getElementById("cool-fg");
        label = document.getElementById("cool-label");
    }

    function update(payload) {
        const { cool_start, cool_end, remaining_sec } = payload;

        const end = new Date(cool_end).getTime();
        const now = Date.now();
        const total = (end - new Date(cool_start).getTime()) / 1000;
        const remaining = Math.max(remaining_sec, 0);

        const pct = remaining / total;
        const dashOffset = 314 - (314 * pct);
        ring.style.strokeDashoffset = dashOffset;

        let color = "#4CAF50"; // green
        if (pct < 0.66) color = "#FFC400"; // yellow
        if (pct < 0.33) color = "#D50000"; // red

        ring.style.stroke = color;
        label.textContent = `${Math.ceil(remaining)}s`;
    }

    function coolingOff() {
        ring.style.strokeDashoffset = 0;
        ring.style.stroke = "#4CAF50";
        label.textContent = "READY";
    }

    function dispatch(event) {
        switch (event.type) {
            case "cooling_update": return update(event.payload);
            case "cooling_off": return coolingOff();
        }
    }

    return { init, dispatch };

})();
