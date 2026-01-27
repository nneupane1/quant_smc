/*
 * Liquidity Sweep Indicator
 *
 * Liquidity sweeps occur when price aggressively hunts resting liquidity
 * above recent highs (sell-side liquidity) or below recent lows (buy-side
 * liquidity). These events often precede significant reversals or
 * continuation moves. This module visualises sweep events on a TradingView
 * chart by placing directional arrow markers at the sweep price.
 *
 * Each event in the input array should have the structure:
 * {
 *   time: <number> Unix timestamp (seconds) at which the sweep occurred,
 *   price: <number> Price level of the sweep,
 *   side: <string> 'buy' if sweeping sell-side (stops above highs),
 *                   'sell' if sweeping buy-side (stops below lows)
 * }
 *
 * Example usage:
 *   const markers = buildSweepMarkers(events);
 *   const cleanup = applySweeps(tvWidget, markers);
 */

/**
 * Build sweep marker definitions from events.
 * Each marker defines its shape ('arrowUp' for buy, 'arrowDown' for sell),
 * colour and text label.
 * @param {Array<Object>} events - Liquidity sweep events
 * @returns {Array<Object>} Marker definitions
 */
export function buildSweepMarkers(events) {
  return events.map(ev => ({
    time: ev.time,
    price: ev.price,
    side: ev.side,
    shape: ev.side === 'buy' ? 'arrowUp' : 'arrowDown',
    color: ev.side === 'buy' ? '#00FF7F' : '#FF4500',
    text: ev.side === 'buy' ? 'Sweep Buy' : 'Sweep Sell',
  }));
}

/**
 * Render sweep markers on the active TradingView chart.
 * Markers are placed slightly outside the candle body by default. You can
 * adjust the `yOffset` to change the vertical offset.
 * @param {Object} tv - TradingView widget instance
 * @param {Array<Object>} markers - Marker definitions
 * @param {number} [yOffset=0.2] - Vertical offset added to price
 * @returns {Function} cleanup function to remove shapes
 */
export function applySweeps(tv, markers, yOffset = 0.2) {
  const chart = tv.activeChart();
  const shapeIds = [];
  markers.forEach(m => {
    const id = chart.createShape({
      shape: m.shape,
      time: m.time,
      price: m.price + (m.side === 'buy' ? yOffset : -yOffset),
      color: m.color,
      text: m.text,
      size: 2,
    });
    shapeIds.push(id);
  });
  return function remove() {
    shapeIds.forEach(id => chart.removeShape(id));
  };
}
