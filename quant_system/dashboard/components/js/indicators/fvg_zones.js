/*
 * Fair Value Gap (FVG) zones indicator module
 *
 * This module provides functions to build and render FVG zones on charts.
 * An FVG zone is a price gap created when a large imbalance of orders
 * leaves a void between the high of one candle and the low of the next (in an
 * uptrend) or vice versa in a downtrend. Price often returns to fill these
 * gaps before continuing the trend, making them important areas of interest.
 *
 * Each FVG zone object should include:
 *  - startTime: timestamp when the gap begins (ms)
 *  - endTime: timestamp when the gap ends or is considered invalidated (ms)
 *  - upper: upper price of the gap (low for downtrend FVG)
 *  - lower: lower price of the gap (high for downtrend FVG)
 *  - filled: boolean flag indicating if price has filled the gap
 *
 * Example usage:
 * import { buildFvgZones, applyFvgZones } from './fvg_zones.js';
 * const zones = buildFvgZones(fvgEvents);
 * applyFvgZones(chart, zones);
 */

/**
 * Convert raw FVG events into zone definitions for rendering.
 *
 * @param {Array<Object>} events - Array of objects describing FVGs with properties
 *   { startTime, endTime, upper, lower, filled }.
 * @returns {Array<Object>} Array of FVG zone objects with computed colours.
 */
export function buildFvgZones(events = []) {
  return events.map(evt => {
    const { startTime, endTime, upper, lower, filled } = evt;
    return {
      start: new Date(startTime).getTime(),
      end: new Date(endTime).getTime(),
      upper: upper,
      lower: lower,
      filled: !!filled,
    };
  });
}

/**
 * Compute a semi-transparent colour for an FVG zone based on whether it is filled.
 *
 * Filled zones are shown with a subdued colour, while open zones are more vivid.
 *
 * @param {boolean} filled - Whether the zone has been filled by price.
 * @returns {string} RGBA colour string.
 */
export function fvgColour(filled) {
  return filled ? 'rgba(255, 193, 7, 0.2)' : 'rgba(255, 193, 7, 0.4)';
}

/**
 * Apply FVG zones to a TradingView chart widget.
 *
 * This will draw rectangular regions covering the price gap for the duration
 * of the zone. Each rectangle is drawn between the start and end times and
 * between the upper and lower prices.
 *
 * @param {Object} chart - TradingView chart widget.
 * @param {Array<Object>} zones - Array of FVG zone definitions.
 */
export function applyFvgZones(chart, zones = []) {
  zones.forEach(zone => {
    const colour = fvgColour(zone.filled);
    if (chart && typeof chart.createShape === 'function') {
      chart.createShape(
        {
          time: zone.start / 1000,
          price: zone.upper,
        },
        {
          shape: 'box',
          shapeProperties: {
            backgroundColor: colour,
            borderColor: colour,
            borderWidth: 0,
            transparency: 80,
            points: [
              { time: zone.start / 1000, price: zone.upper },
              { time: zone.end / 1000, price: zone.lower },
            ],
          },
        },
      );
    }
  });
}
