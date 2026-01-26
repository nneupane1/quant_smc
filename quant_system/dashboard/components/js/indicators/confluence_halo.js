/*
 * Confluence halo indicator module
 *
 * This module provides functions to build and render confluence halo overlays
 * on trading charts. A confluence halo is a semi-transparent band drawn
 * around price bars to visualise the aggregated confluence score from multiple
 * models and SMC signals. The score should be normalised between 0 and 1.
 *
 * Each halo segment is defined by:
 *  - time: timestamp of the bar (ms)
 *  - high: high price for the halo band
 *  - low: low price for the halo band
 *  - score: confluence score between 0 and 1
 *
 * The rendering function will map the score to an RGBA colour; darker
 * intensity indicates higher confluence.
 *
 * Example usage:
 * import { buildHaloSegments, applyConfluenceHalo } from './confluence_halo.js';
 * const segments = buildHaloSegments(series, priceRange);
 * applyConfluenceHalo(tvWidget, segments);
 */

/**
 * Build halo segments from a time-series of confluence scores.
 *
 * @param {Array<Object>} series - Array of objects with timestamp, score,
 *   and optionally high and low values. If high/low are omitted, the priceRange
 *   argument will be used instead.
 * @param {Object} priceRange - Optional object with properties { min, max }
 *   specifying the vertical range (e.g. day's low/high). Required when series
 *   lacks high/low.
 * @returns {Array<Object>} Array of halo segment definitions.
 */
export function buildHaloSegments(series = [], priceRange = null) {
  return series.map(point => {
    const { timestamp, score, high, low } = point;
    const top = high !== undefined ? high : (priceRange ? priceRange.max : null);
    const bottom = low !== undefined ? low : (priceRange ? priceRange.min : null);
    return {
      time: new Date(timestamp).getTime(),
      high: top,
      low: bottom,
      score: Math.max(0, Math.min(1, score ?? 0)),
    };
  });
}

/**
 * Convert a confluence score to an RGBA colour string.
 *
 * @param {number} score - Normalised confluence score [0,1].
 * @returns {string} RGBA colour.
 */
export function scoreToColour(score) {
  // Base colour for confluence halo (blue). Adjust alpha based on score.
  const alpha = Math.max(0.05, Math.min(0.6, score));
  return `rgba(0, 123, 255, ${alpha})`;
}

/**
 * Apply confluence halos to a TradingView widget.
 *
 * For each halo segment, draw a filled rectangle on the chart between
 * the specified high and low prices. This function assumes the widget
 * supports createShape with a 'box' shape type.
 *
 * @param {Object} chart - TradingView chart widget.
 * @param {Array<Object>} segments - Array of halo segment definitions.
 */
export function applyConfluenceHalo(chart, segments = []) {
  segments.forEach(seg => {
    const colour = scoreToColour(seg.score);
    if (chart && typeof chart.createShape === 'function') {
      chart.createShape(
        {
          time: seg.time / 1000,
          price: seg.high,
        },
        {
          shape: 'box',
          shapeProperties: {
            backgroundColor: colour,
            borderColor: colour,
            borderWidth: 0,
            transparency: 80,
            // Coordinates: top price is seg.high, bottom is seg.low
            points: [
              { time: seg.time / 1000, price: seg.high },
              { time: seg.time / 1000, price: seg.low },
            ],
          },
        },
      );
    }
  });
}
