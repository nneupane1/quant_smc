// explainer.js
// Renders hierarchical strategy reasoning and handles animation.

(function() {

    window.addEventListener("message", (event) => {
        if (!event.data || event.data.type !== "strategy_explainer_update") return;
        const tree = event.data.tree;
        renderExplainer(tree);
    });

    function renderExplainer(tree) {
        let root = document.getElementById("explainer-root");
        if (!root) {
            root = document.createElement("div");
            root.id = "explainer-root";
            root.className = "explainer-container";
            document.body.appendChild(root);
        }
        root.innerHTML = ""; // rebuild completely

        buildSection(root, "SMC Context", tree.smc);
        buildSection(root, "Flow Validation (1h)", tree.flow);
        buildSection(root, "ML Signals", tree.ml);
        buildSection(root, "Confluence Breakdown", tree.confluence);
        buildSection(root, "EVR Computation", tree.evr);
        buildSection(root, "Hazard & Trailing Logic", tree.hazard);
        buildSection(root, "MPC Risk Decisions", tree.mpc);

        attachListeners();
    }

    function buildSection(root, title, obj) {
        const sectionTitle = document.createElement("div");
        sectionTitle.className = "section-title";
        sectionTitle.innerText = title;
        root.appendChild(sectionTitle);

        const node = makeNode("Details", obj);
        root.appendChild(node);
    }

    function makeNode(label, obj) {
        const wrapper = document.createElement("div");
        wrapper.className = "node";

        const header = document.createElement("div");
        header.className = "node-header";
        header.innerText = label;

        const body = document.createElement("div");
        body.className = "node-body";

        if (typeof obj === "object") {
            Object.keys(obj).forEach(k => {
                const val = obj[k];
                const div = document.createElement("div");
                div.className = "metric";
                div.innerHTML = `<span class="key">${k}</span>: <span class="val">${format(val)}</span>`;
                body.appendChild(div);
            });
        }

        wrapper.appendChild(header);
        wrapper.appendChild(body);
        return wrapper;
    }

    function format(v) {
        if (v == null) return "";
        if (typeof v === "number") return Number(v).toFixed(4);
        return v;
    }

    function attachListeners() {
        const headers = document.querySelectorAll(".node-header");
        headers.forEach(h => {
            h.onclick = () => {
                const body = h.nextElementSibling;
                body.classList.toggle("open");
            };
        });
    }

})();
