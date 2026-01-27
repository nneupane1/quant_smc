/*
 * Regime Panel Indicator
 *
 * This module displays regime probabilities derived from a Hidden Markov
 * Model (HMM) on a TradingView chart. Each regime represents a different
 * market condition: trend, range, expansion and collapse. For each bar
 * we display the dominant regime as a small coloured square above the
 * candle. Colours are assigned as follows:
 *   trend    -> blue
 *   range    -> orange
 *   expansion-> green
 *   collapse -> red
 *
 * Input:
 *   timestamps  - Array<number> of bar timestamps (seconds)
 *   regimes     - Array<Object> where each element is an object with
 *                 probabilities for each regime, e.g.
 *                 { trend: 0.5, range: 0.3, expansion: 0.1, collapse: 0.1 }
 *
 * Example usage:
 *   const markers = buildRegimeMarkers(times, probs);
 *   const cleanup = applyRegimePanel(tvWidget, markers);
 */

/**
 * Determine the dominant regime and its colour.
 * @param {Object} prob - Regime probability object
 * @returns {{label: string, color: string}}
 */
export function regimeColour(prob) {
  const entries = Object.entries(prob);
  entries.sort((a, b) => b[1] - a[1]);
  const [regime] = entries[0];
  let color;
  switch (regime) {
    case 'trend':
      color = '#0094FF';
      break;
    case 'range':
      color = '#FFA500';
      break;
    case 'expansion':
      color = '#00C49A';
      break;
    case 'collapse':
      color = '#FF4D4D';
      break;
    default:
      color = '#888888';
  }
  return { label: regime, color };
}

/**
 * Build marker definitions for regime probabilities.
 * Each marker is placed slightly above the candle high and uses a small
 * square shape. The size parameter can be adjusted as needed.
 * @param {Array<number>} times - Bar timestamps
 * @param {Array<Object>} probs - Regime probability objects
 * @returns {Array<Object>} marker definitions
 */
export function buildRegimeMarkers(times, probs) {
  const markers = [];
  for (let i = 0; i < times.length; i++) {
    const { label, color } = regimeColour(probs[i]);
    markers.push({
      time: times[i],
      color: color,
      text: label,
    });
  }
  return markers;
}

/**
 * Render regime markers on the active chart.
 * Each marker is drawn as a square at the candle high plus an offset.
 * The offset ensures the marker does not overlap with price bars.
 * @param {Object} tv - TradingView widget instance
 * @param {Array<Object>} markers - Marker definitions
 * @param {number} [offset=0.5] - Price offset in chart units
 * @returns {Function} cleanup function to remove shapes
 */
export function applyRegimePanel(tv, markers, offset = 0.5) {
  const chart = tv.activeChart();
  const shapeIds = [];
  markers.forEach(m => {
    const id = chart.createShape({
      shape: 'square',
      time: m.time,
      price: null, // null ensures the shape anchors to bar's high
      color: m.color,
      text: m.text.toUpperCase(),
      size: 3,
      yOffset: offset,
    });
    shapeIds.push(id);
  });
  return function remove() {
    shapeIds.forEach(id => chart.removeShape(id));
  };
}
