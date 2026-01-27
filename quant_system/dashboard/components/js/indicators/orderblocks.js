/*
 * Order Blocks Indicator
 *
 * This module visualises detected order block zones on a TradingView chart.
 * Order blocks are areas where large institutional orders have left
 * unfilled resting liquidity. When price revisits these zones they often
 * act as support (bullish order block) or resistance (bearish order block).
 *
 * Each event in the input array should have the following shape:
 * {
 *   start: <number> Unix timestamp in seconds marking the first bar of the zone,
 *   end:   <number> Unix timestamp in seconds marking the last bar of the zone,
 *   low:   <number> Lowest price of the zone,
 *   high:  <number> Highest price of the zone,
 *   side:  <string> Either 'buy' for a bullish block or 'sell' for a bearish block,
 *   filled: <boolean> Whether price has already traded through the zone.
 * }
 *
 * Usage:
 *   const zones = buildOrderBlockZones(events);
 *   // zones is an array of {time, to, price, toPrice, color}
 *   const cleanup = applyOrderBlocks(tvWidget, zones);
 *   // call cleanup() to remove shapes when no longer needed
 */

/**
 * Compute the display colour for an order block.
 * Bullish blocks are green, bearish blocks are red. Filled blocks have
 * reduced opacity.
 * @param {Object} ob - order block event
 * @returns {string} CSS rgba colour string
 */
export function obColour(ob) {
  const alpha = ob.filled ? 0.15 : 0.3;
  return ob.side === 'buy'
    ? `rgba(0, 200, 0, ${alpha})`
    : `rgba(200, 0, 0, ${alpha})`;
}

/**
 * Build a list of zone descriptors from order block events.
 * @param {Array<Object>} events - Array of order block events
 * @returns {Array<Object>} List of zone definitions for rendering
 */
export function buildOrderBlockZones(events) {
  return events.map(ob => ({
    time: ob.start,
    to: ob.end,
    price: ob.high,
    toPrice: ob.low,
    color: obColour(ob),
    side: ob.side,
    filled: ob.filled,
  }));
}

/**
 * Render order block zones on a TradingView chart.
 * This function iterates through the list of zone descriptors and creates
 * rectangular shapes spanning the time and price range of each block. The
 * shapes are drawn on the main pane. A cleanup function is returned
 * allowing all created shapes to be removed.
 *
 * @param {Object} tv - TradingView widget instance
 * @param {Array<Object>} zones - Zone descriptors as returned by buildOrderBlockZones
 * @returns {Function} cleanup function to remove created shapes
 */
export function applyOrderBlocks(tv, zones) {
  const chart = tv.activeChart();
  const shapeIds = [];
  zones.forEach(z => {
    const id = chart.createShape({
      shape: 'box',
      time: z.time,
      to: z.to,
      price: z.price,
      toPrice: z.toPrice,
      color: z.color,
      borderColor: z.side === 'buy' ? 'rgba(0,100,0,0.6)' : 'rgba(100,0,0,0.6)',
      transparency: 50,
      text: z.side === 'buy' ? 'OB (Buy)' : 'OB (Sell)',
    });
    shapeIds.push(id);
  });
  // return cleanup
  return function remove() {
    shapeIds.forEach(id => chart.removeShape(id));
  };
}
