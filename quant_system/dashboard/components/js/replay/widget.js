// widget.js
// Bi-directional communication layer between Streamlit <-> Replay JS.
//
// Responsibilities:
//  - Provide a clean event bus.
//  - Allow Streamlit to send structured commands to replay.js
//  - Allow replay.js to emit UI events back to Streamlit.
//  - No flicker, no reloading.
//  - Works with Streamlit's postMessage API.
//
// Usage from Python:
//
//   st.markdown(f"""
//       <script>
//           ReplayWidget.send("jump", {{ timestamp: {ts} }});
//       </script>
//   """, unsafe_allow_html=True)
//
// Usage from JS:
//
//   ReplayWidget.on("cursor_move", data => {...});
//   ReplayWidget.emit("cursor_move", payload);
//
// All messages flow through window.postMessage() namespaced by `replay:`
//

class ReplayWidgetClass {
    constructor() {
        this.handlers = {};     // event_name -> [callbacks]
        this.ready = false;

        window.addEventListener("message", (event) => {
            if (!event || !event.data) return;

            const msg = event.data;

            // Accept only namespaced messages from Streamlit or self
            if (typeof msg !== "string" || !msg.startsWith("replay:")) return;

            let payload;
            try { payload = JSON.parse(msg.slice("replay:".length)); }
            catch { return; }

            this.routeIncoming(payload);
        });

        // Notify Python the widget is ready
        setTimeout(() => {
            this.ready = true;
            this.emit("ready", { status: "widget_ready" });
        }, 50);
    }

    // Register event handler
    on(eventName, callback) {
        if (!this.handlers[eventName]) this.handlers[eventName] = [];
        this.handlers[eventName].push(callback);
    }

    // Emit event (JS -> Python)
    emit(eventName, data) {
        const payload = {
            event: eventName,
            data: data || {},
        };
        const msg = "replay:" + JSON.stringify(payload);
        window.parent.postMessage(msg, "*");
    }

    // Send command (Python -> JS)
    send(command, data) {
        // interpreted by replay.js rather than routed back to Python
        const payload = {
            command: command,
            data: data || {},
        };
        this.routeIncoming(payload);
    }

    // Route incoming payload
    routeIncoming(payload) {
        if (payload.command) {
            this.handleCommand(payload.command, payload.data);
        } else if (payload.event) {
            this.handleEvent(payload.event, payload.data);
        }
    }

    // Handle events emitted by Python
    handleCommand(cmd, data) {
        switch (cmd) {
            case "jump":
                if (window.replay_jump) replay_jump(data.timestamp);
                break;

            case "next":
                if (window.replay_next) replay_next();
                break;

            case "prev":
                if (window.replay_prev) replay_prev();
                break;

            case "play":
                if (window.replay_play) replay_play();
                break;

            case "load":
                if (window.replay_load) replay_load(JSON.stringify(data.payload));
                break;

            case "theme":
                document.body.className = "";
                document.body.classList.add(data.theme || "default");
                break;

            default:
                console.warn("Unknown command:", cmd, data);
                break;
        }
    }

    // Handle JS → Python events
    handleEvent(eventName, data) {
        const listeners = this.handlers[eventName] || [];
        listeners.forEach(cb => cb(data));
    }
}


// Create global singleton
window.ReplayWidget = new ReplayWidgetClass();


// -------------------------------------------------------------
// Optional UX side features
// -------------------------------------------------------------

// Notify Python when playback finishes
if (window.replay_next) {
    const oldNext = window.replay_next;

    window.replay_next = function() {
        oldNext();

        // Detect end-of-replay
        if (typeof replayData !== "undefined" && typeof ptr !== "undefined") {
            if (ptr === replayData.length - 1) {
                ReplayWidget.emit("finished", { idx: ptr });
            }
        }
    };
}


// Notify Python when user hovers on chart
if (window.chart) {
    window.chart.subscribeCrosshairMove(param => {
        if (!param || !param.time) return;
        ReplayWidget.emit("cursor_move", { time: param.time });
    });
}
