// smart_alerts.js
// Animated sidebar that displays confluence, EVR, hazard, SMC, regime.

window.SmartAlerts = (function () {

    let container = null;

    function init(elementId) {
        container = document.getElementById(elementId);
    }

    function createBlock(title, body) {
        const wrap = document.createElement("div");
        wrap.className = "sa-block";

        const th = document.createElement("div");
        th.className = "sa-title";
        th.innerText = title;

        const bd = document.createElement("div");
        bd.className = "sa-body";
        bd.innerHTML = body;

        wrap.appendChild(th);
        wrap.appendChild(bd);
        return wrap;
    }

    function render(payload) {
        if (!container) return;

        container.innerHTML = "";

        const header = document.createElement("div");
        header.className = "sa-header";
        header.innerHTML = `
            <div class="sa-side ${payload.side}">${payload.side.toUpperCase()}</div>
            <div class="sa-symbol">${payload.symbol}</div>
            <div class="sa-ts">${payload.timestamp}</div>
        `;
        container.appendChild(header);

        // Confluence
        const c = payload.confluence;
        container.appendChild(createBlock(
            "Confluence Breakdown",
            `
            <table class="sa-table">
                <tr><td>BOS Prob</td><td>${(c.bos_prob*100).toFixed(1)}%</td></tr>
                <tr><td>Liq-Flow Prob</td><td>${(c.liq_prob*100).toFixed(1)}%</td></tr>
                <tr><td>Momo Prob</td><td>${(c.momo_prob*100).toFixed(1)}%</td></tr>
                <tr><td>RSV</td><td>${c.rsv_score.toFixed(3)}</td></tr>
                <tr><td>SMC Score</td><td>${c.smc_score.toFixed(3)}</td></tr>
                <tr><td>Session Weight</td><td>${c.session_weight.toFixed(2)}</td></tr>
                <tr><td><b>Total Confluence</b></td><td><b>${c.total.toFixed(3)}</b></td></tr>
            </table>
            `
        ));

        // EVR
        const e = payload.evr;
        container.appendChild(createBlock(
            "Expected-R Breakdown",
            `
            <table class="sa-table">
                <tr><td>1st Target (R)</td><td>${e.t1_r.toFixed(2)}</td></tr>
                <tr><td>T1 Prob</td><td>${(e.t1_prob*100).toFixed(1)}%</td></tr>

                <tr><td>2nd Target (R)</td><td>${e.t2_r.toFixed(2)}</td></tr>
                <tr><td>T2 Prob</td><td>${(e.t2_prob*100).toFixed(1)}%</td></tr>

                <tr><td>Failure Prob</td><td>${(e.fail_prob*100).toFixed(1)}%</td></tr>
                <tr><td>Costs (R)</td><td>${e.costs.toFixed(2)}</td></tr>

                <tr><td><b>EVR</b></td><td><b>${e.evr.toFixed(3)}</b></td></tr>
                <tr><td><b>Median-R</b></td><td><b>${e.median_r.toFixed(3)}</b></td></tr>
            </table>
            `
        ));

        // Hazard
        const h = payload.hazard;
        container.appendChild(createBlock(
            "Hazard Snapshot",
            `
            <table class="sa-table">
                <tr><td>Current h(t)</td><td>${h.ht.toFixed(3)}</td></tr>
                <tr><td>Threshold</td><td>${h.threshold.toFixed(3)}</td></tr>
                <tr><td>Action</td><td><b>${h.action}</b></td></tr>
            </table>
            `
        ));

        // Gates
        const g = payload.gates;
        container.appendChild(createBlock(
            "Gate Status",
            `
            <table class="sa-table">
                <tr><td>10h Gate</td><td>${g.g10h ? "PASS" : "FAIL"}</td></tr>
                <tr><td>6h Gate</td><td>${g.g6h ? "PASS" : "FAIL"}</td></tr>
                <tr><td>1h Flow</td><td>${g.g1h ? "PASS" : "FAIL"}</td></tr>
                <tr><td>15m Exec</td><td>${g.g15m ? "PASS" : "FAIL"}</td></tr>
            </table>
            `
        ));

        // SMC
        const s = payload.smc;
        container.appendChild(createBlock(
            "SMC Structure",
            `
            <table class="sa-table">
                <tr><td>Sweep</td><td>${s.sweep_flag}</td></tr>
                <tr><td>OB</td><td>${s.ob_label}</td></tr>
                <tr><td>FVG</td><td>${s.fvg_label}</td></tr>
                <tr><td>BOS</td><td>${s.bos_flag}</td></tr>
                <tr><td>CHOCH</td><td>${s.choch_flag}</td></tr>
            </table>
            `
        ));

        // Session + Regime
        const r = payload.regime;
        container.appendChild(createBlock(
            "Session & Regime",
            `
            <table class="sa-table">
                <tr><td>Session</td><td>${payload.session}</td></tr>
                <tr><td>Trend Prob</td><td>${(r.trend*100).toFixed(1)}%</td></tr>
                <tr><td>Range Prob</td><td>${(r.range*100).toFixed(1)}%</td></tr>
                <tr><td>Expansion Prob</td><td>${(r.expansion*100).toFixed(1)}%</td></tr>
                <tr><td>Toxicity</td><td>${r.tox.toFixed(3)}</td></tr>
            </table>
            `
        ));

        container.classList.add("sa-pop");
        setTimeout(() => container.classList.remove("sa-pop"), 300);
    }

    function dispatch(event) {
        if (event.type === "smart_alert") {
            render(event.payload);
        }
    }

    return { init, dispatch };

})();
