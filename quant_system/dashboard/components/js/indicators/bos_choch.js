/*
 * BOS/CHOCH indicator module
 *
 * This module provides helper functions for rendering Break of Structure (BOS)
 * and Change of Character (CHOCH) events on a TradingView chart or other
 * charting library. BOS occurs when price breaks a significant structural
 * high/low in the direction of the prevailing trend. CHOCH occurs when price
 * breaks the previous high/low against the prevailing trend, signalling a
 * potential reversal.
 *
 * The exported functions consume arrays of BOS/CHOCH event objects from the
 * back-end (e.g. the ML engine) and return drawing instructions. Each event
 * object should have at minimum:
 *   - timestamp: ISO string or epoch ms of the candle where the event occurred.
 *   - price: the price level at which the break occurred.
 *   - type: either 'BOS' or 'CHOCH'.
 * Optional fields can include 'timeframe', 'strength', 'structureHighLow' etc.
 *
 * Example usage:
 * import { buildBosChochLines } from './bos_choch.js';
 * const overlays = buildBosChochLines(events);
 * overlays.forEach(overlay => chart.addLine(overlay));
 */

/**
 * Convert a BOS/CHOCH event into a line definition for TradingView or similar
 * chart. Returns an object describing the line.
 *
 * @param {Object} event - BOS or CHOCH event object.
 * @returns {Object} A line definition with coordinates, style and text label.
 */
export function buildLineFromEvent(event) {
  const { timestamp, price, type, timeframe } = event;
  // Use different colours for BOS and CHOCH
  const colour = type === 'BOS' ? '#00C49A' : '#FF4444';
  const label = `${type}${timeframe ? ` (${timeframe})` : ''}`;

  return {
    time: new Date(timestamp).getTime(),
    price: price,
    colour: colour,
    width: 1,
    style: 'dashed',
    text: label,
    textAlign: 'right',
  };
}

/**
 * Create line overlays for an array of BOS/CHOCH events.
 *
 * @param {Array<Object>} events - Array of event objects.
 * @returns {Array<Object>} Array of line definitions ready for rendering.
 */
export function buildBosChochLines(events = []) {
  return events
    .filter(evt => evt && (evt.type === 'BOS' || evt.type === 'CHOCH'))
    .map(evt => buildLineFromEvent(evt));
}

/**
 * Apply BOS/CHOCH overlays to a TradingView widget instance.
 *
 * Note: This function assumes that the passed widget implements an API
 * compatible with TradingView's charting library (e.g. tvWidget instance).
 * If you are using another charting library, adapt this function accordingly.
 *
 * @param {Object} chart - TradingView widget instance.
 * @param {Array<Object>} events - Array of BOS/CHOCH events.
 */
export function applyBosChochOverlay(chart, events = []) {
  const lines = buildBosChochLines(events);
  lines.forEach(line => {
    if (chart && typeof chart.createShape === 'function') {
      chart.createShape(
        { time: line.time / 1000, price: line.price },
        {
          shape: 'line',
          shapeProperties: {
            linecolor: line.colour,
            linestyle: 2,
            linewidth: line.width,
            text: line.text,
            textalign: line.textAlign,
          },
        },
      );
    }
  });
}
