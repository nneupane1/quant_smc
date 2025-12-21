// profit_heatmap.js
// GitHub-style heatmap tiles for daily, weekly, monthly PnL.

window.ProfitHeatmap = (function () {

    let dailyContainer = null;
    let weeklyContainer = null;
    let monthlyContainer = null;

    let dailyMap = {};
    let weeklyMap = {};
    let monthlyMap = {};

    function init(dailyId, weeklyId, monthlyId) {
        dailyContainer = document.getElementById(dailyId);
        weeklyContainer = document.getElementById(weeklyId);
        monthlyContainer = document.getElementById(monthlyId);
    }

    // Convert pnl → color
    function pnlToColor(val) {
        if (val > 0) {
            const t = Math.min(val / 1000, 1);
            return `rgba(0, 230, 118, ${0.2 + 0.8 * t})`;
        }
        if (val < 0) {
            const t = Math.min(Math.abs(val) / 1000, 1);
            return `rgba(229, 57, 53, ${0.2 + 0.8 * t})`;
        }
        return "rgba(255,255,255,0.15)";
    }

    function animateTile(tile) {
        tile.style.transform = "scale(1.15)";
        tile.style.opacity = 1;
        setTimeout(() => {
            tile.style.transform = "scale(1)";
        }, 120);
    }

    function renderDaily(payload) {
        const { timestamp, pnl } = payload;
        dailyMap[timestamp] = pnl;

        dailyContainer.innerHTML = "";
        Object.keys(dailyMap).forEach(ts => {
            const val = dailyMap[ts];
            const div = document.createElement("div");
            div.className = "heat-tile";
            div.style.background = pnlToColor(val);
            div.title = `${ts} : ${val.toFixed(2)}`;
            dailyContainer.appendChild(div);
            animateTile(div);
        });
    }

    function renderWeekly(payload) {
        const { week, pnl } = payload;
        weeklyMap[week] = pnl;

        weeklyContainer.innerHTML = "";
        Object.keys(weeklyMap).forEach(wk => {
            const val = weeklyMap[wk];
            const div = document.createElement("div");
            div.className = "heat-tile-week";
            div.style.background = pnlToColor(val);
            div.title = `Week ${wk} : ${val.toFixed(2)}`;
            weeklyContainer.appendChild(div);
            animateTile(div);
        });
    }

    function renderMonthly(payload) {
        const { month, pnl } = payload;
        monthlyMap[month] = pnl;

        monthlyContainer.innerHTML = "";
        Object.keys(monthlyMap).forEach(m => {
            const val = monthlyMap[m];
            const div = document.createElement("div");
            div.className = "heat-tile-month";
            div.style.background = pnlToColor(val);
            div.title = `${m} : ${val.toFixed(2)}`;
            monthlyContainer.appendChild(div);
            animateTile(div);
        });
    }

    function dispatch(event) {
        switch (event.type) {
            case "heatmap_daily": return renderDaily(event.payload);
            case "heatmap_weekly": return renderWeekly(event.payload);
            case "heatmap_monthly": return renderMonthly(event.payload);
        }
    }

    return { init, dispatch };

})();
