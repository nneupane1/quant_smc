// heatmap.js
// Dynamic animated heatmap grid using postMessage events.

(function() {
    window.addEventListener("message", (event) => {
        const payload = event.data;
        if (!payload || payload.type !== "portfolio_heatmap_update") return;

        const data = payload.data;
        if (!data) return;

        updateHeatmap(data);
    });

    function updateHeatmap(rows) {
        let container = document.getElementById("heatmap-root");
        if (!container) {
            container = document.createElement("div");
            container.id = "heatmap-root";
            container.className = "heatmap-grid";
            document.body.appendChild(container);

            renderHeader(container);
        }

        renderRows(container, rows);
    }

    function renderHeader(root) {
        const headers = ["Asset","Long","Short","Net","Risk Wt","Vol Z","Regime"];
        headers.forEach(h => {
            const div = document.createElement("div");
            div.className = "heatmap-header";
            div.innerText = h;
            root.appendChild(div);
        });
    }

    function renderRows(root, rows) {
        const existing = root.querySelectorAll(".cell");
        existing.forEach(e => e.remove());

        rows.forEach(r => {
            pushCell(root, r.asset, "");
            pushCell(root, format(r.long), "cell-long");
            pushCell(root, format(r.short), "cell-short");

            const netClass = r.net >= 0 ? "cell-net-pos" : "cell-net-neg";
            pushCell(root, format(r.net), netClass);

            pushCell(root, format(r.risk_weight), "");
            pushCell(root, format(r.vol_z), "");

            const regimeClass = "cell-regime-" + r.regime;
            pushCell(root, r.regime, regimeClass);
        });
    }

    function pushCell(root, text, cls) {
        const div = document.createElement("div");
        div.className = "cell " + cls;
        div.innerText = text;
        root.appendChild(div);
    }

    function format(v) {
        return Number(v).toFixed(2);
    }
})();
