// bos_choch.js
// Break of Structure (BOS) and CHOCH overlays.
// BOS = continuation line
// CHOCH = reversal line

export function createBOSCHOCHSeries(chart) {
    return {
        bos: chart.addLineSeries({
            color: "rgba(0,168,255,0.85)",
            lineWidth: 2,
            lineStyle: 1
        }),
        choch: chart.addLineSeries({
            color: "rgba(255,0,128,0.85)",
            lineWidth: 2,
            lineStyle: 2
        })
    };
}

export function updateBOSCHOCH(series, bosList, chochList) {
    if (!bosList) bosList = [];
    if (!chochList) chochList = [];

    const bosData = bosList.map(p => ({
        time: p.t,
        value: p.level
    }));

    const chochData = chochList.map(p => ({
        time: p.t,
        value: p.level
    }));

    series.bos.setData(bosData);
    series.choch.setData(chochData);
}
