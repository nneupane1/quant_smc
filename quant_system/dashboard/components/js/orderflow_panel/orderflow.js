// orderflow.js
// DOM renderer for L2 orderbook + delta bars.

window.OrderflowPanel = (function () {

    let bidsEl = null;
    let asksEl = null;
    let deltaEl = null;

    function init(bidsId, asksId, deltaId) {
        bidsEl = document.getElementById(bidsId);
        asksEl = document.getElementById(asksId);
        deltaEl = document.getElementById(deltaId);
    }

    function levelColor(size, maxSize) {
        const t = Math.min(size / maxSize, 1);
        return `rgba(0, 180, 255, ${0.15 + 0.85 * t})`;
    }

    function renderOrderbook(payload) {
        const { bids, asks } = payload;

        const maxBid = Math.max(...bids.map(x => x[1]), 1);
        const maxAsk = Math.max(...asks.map(x => x[1]), 1);

        bidsEl.innerHTML = "";
        asksEl.innerHTML = "";

        bids.forEach(([p, s]) => {
            const row = document.createElement("div");
            row.className = "of-row";
            row.style.background = levelColor(s, maxBid);
            row.innerHTML = `<span>${p.toFixed(1)}</span><span>${s.toFixed(3)}</span>`;
            bidsEl.appendChild(row);
        });

        asks.forEach(([p, s]) => {
            const row = document.createElement("div");
            row.className = "of-row";
            row.style.background = levelColor(s, maxAsk);
            row.innerHTML = `<span>${p.toFixed(1)}</span><span>${s.toFixed(3)}</span>`;
            asksEl.appendChild(row);
        });
    }

    function renderDelta(payload) {
        const { timestamp, buy, sell, delta } = payload;

        const bar = document.createElement("div");
        bar.className = "delta-bar";
        bar.style.height = "6px";
        bar.style.marginBottom = "2px";
        bar.style.background = delta >= 0
            ? `rgba(0,255,0,${Math.min(delta/500,1)})`
            : `rgba(255,0,0,${Math.min(Math.abs(delta)/500,1)})`;

        deltaEl.prepend(bar);
        if (deltaEl.children.length > 120) {
            deltaEl.removeChild(deltaEl.lastChild);
        }
    }

    function dispatch(event) {
        switch(event.type) {
            case "orderbook_update": return renderOrderbook(event.payload);
            case "delta_update": return renderDelta(event.payload);
        }
    }

    return { init, dispatch };

})();
