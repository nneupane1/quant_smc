// exec_panel.js
// Live Execution Panel UI. Zero-refresh. Bloomberg-style aesthetics.

window.ExecPanel = (function () {

    let container = null;
    let sendAction = null;

    function init(elementId, callback) {
        container = document.getElementById(elementId);
        sendAction = callback;
        renderBaseUI();
    }

    function renderBaseUI() {
        container.innerHTML = `
        <div class="exec-header">Execution Control</div>

        <div class="exec-block">
            <label>Leverage</label>
            <input id="exec-lev" type="range" min="1" max="5" step="1" value="1"/>
            <span id="exec-lev-val">1×</span>
        </div>

        <div class="exec-block">
            <label>Position Size (USDT)</label>
            <input id="exec-size" type="number" value="20000" min="10" max="1000000" />
        </div>

        <div class="exec-block">
            <label>Take Profit (TP)</label>
            <input id="exec-tp" type="number" step="0.1" />
        </div>

        <div class="exec-block">
            <label>Stop Loss (SL)</label>
            <input id="exec-sl" type="number" step="0.1" />
        </div>

        <div class="exec-block">
            <label>Risk Mode</label>
            <select id="exec-risk">
                <option value="low">Low (0.5%)</option>
                <option value="normal" selected>Normal (1%)</option>
                <option value="high">High (1.5%)</option>
            </select>
        </div>

        <div class="exec-block">
            <label>Hedge Position</label>
            <label class="switch">
                <input id="exec-hedge" type="checkbox">
                <span class="slider"></span>
            </label>
        </div>

        <button id="exec-force-exit" class="exec-btn-danger">Force Market Exit</button>
        `;

        wireEvents();
    }

    function wireEvents() {
        document.getElementById("exec-lev").oninput = (e) => {
            const val = e.target.value;
            document.getElementById("exec-lev-val").innerText = `${val}×`;
            sendAction({ action: "set_leverage", data: { leverage: parseInt(val) } });
        };

        document.getElementById("exec-size").onchange = (e) =>
            sendAction({ action: "set_size", data: { size: parseFloat(e.target.value) } });

        document.getElementById("exec-tp").onchange = (e) =>
            sendAction({ action: "set_tp", data: { tp: parseFloat(e.target.value) } });

        document.getElementById("exec-sl").onchange = (e) =>
            sendAction({ action: "set_sl", data: { sl: parseFloat(e.target.value) } });

        document.getElementById("exec-risk").onchange = (e) =>
            sendAction({ action: "risk_mode", data: { risk: e.target.value } });

        document.getElementById("exec-hedge").onchange = (e) =>
            sendAction({ action: "toggle_hedge", data: { enabled: e.target.checked } });

        document.getElementById("exec-force-exit").onclick = () =>
            sendAction({ action: "force_exit", data: {} });
    }

    function updateState(state) {
        // state = { leverage, size, tp, sl, risk_mode, hedge_enabled }
        document.getElementById("exec-lev").value = state.leverage;
        document.getElementById("exec-lev-val").innerText = `${state.leverage}×`;

        document.getElementById("exec-size").value = state.size;
        document.getElementById("exec-tp").value = state.tp || "";
        document.getElementById("exec-sl").value = state.sl || "";
        document.getElementById("exec-risk").value = state.risk_mode;
        document.getElementById("exec-hedge").checked = state.hedge_enabled;
    }

    function dispatch(event) {
        if (event.type === "exec_state") {
            updateState(event.payload);
        }
    }

    return { init, dispatch };

})();
