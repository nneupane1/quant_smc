/*
 * Hazard Panel indicator module
 *
 * This module provides functions to transform hazard probabilities from the ML hazard model into visual indicators
 * that can be rendered on a TradingView chart. The hazard curve is a discrete-time survival function that measures
 * the conditional probability of a stop-loss event in the next bar given survival up to the current bar.
 *
 * The hazard panel renders each value as a coloured square or bar above the price chart, with colour intensity
 * indicating the magnitude of the hazard. A simple palette is used: green for low hazard, orange for medium hazard and red for high hazard.
 *
 * Example usage:
 *   import { buildHazardMarkers, applyHazardPanel } from './hazard_panel.js';
 *   const markers = buildHazardMarkers(timestamps, hazardCurve);
 *   applyHazardPanel(chart, markers);
 */

/**
 * Determine a colour for a given hazard probability.
 *
 * @param {number} hazard - Hazard probability between 0 and 1.
 * @returns {string} A hex colour representing low, medium or high hazard.
 */
export function hazardColour(hazard) {
  if (hazard == null || isNaN(hazard)) return '#999999'; // fallback grey
  if (hazard >= 0.7) return '#e74c3c'; // high hazard - red
  if (hazard >= 0.4) return '#f39c12'; // medium hazard - orange
  return '#2ecc71'; // low hazard - green
}

/**
 * Build an array of hazard marker objects for rendering.
 *
 * @param {Array<number>} timestamps - An array of unix epoch timestamps (ms) corresponding to each bar.
 * @param {Array<number>} hazardCurve - An array of hazard probabilities aligned with timestamps.
 * @returns {Array<Object>} An array of objects containing time, hazard value and colour.
 */
export function buildHazardMarkers(timestamps, hazardCurve) {
  const markers = [];
  if (!Array.isArray(timestamps) || !Array.isArray(hazardCurve)) return markers;
  const n = Math.min(timestamps.length, hazardCurve.length);
  for (let i = 0; i < n; i++) {
    const hazard = hazardCurve[i];
    const time = timestamps[i];
    markers.push({
      time,
      hazard,
      color: hazardColour(hazard),
    });
  }
  return markers;
}

/**
 * Render hazard markers on a TradingView chart instance.
 *
 * Each marker is drawn as a small square shape above the bar. The marker's colour encodes the hazard level
 * and optional text shows the hazard percentage. This helper will automatically assign an id to each shape
 * to avoid duplicates; previously drawn shapes are removed before rendering new ones.
 *
 * @param {Object} chart - A TradingView chart widget. Must support createShape() and removeShape().
 * @param {Array<Object>} markers - Hazard markers built via buildHazardMarkers().
 */
export function applyHazardPanel(chart, markers) {
  if (!chart || typeof chart.createShape !== 'function') {
    console.warn('applyHazardPanel: invalid chart instance');
    return;
  }
  // Initialise storage for shape ids on the chart instance
  if (!chart.__hazardShapeIds) {
    chart.__hazardShapeIds = [];
  }
  // Remove any previously created hazard shapes
  chart.__hazardShapeIds.forEach(id => {
    try {
      chart.removeShape(id);
    } catch (err) {
      // ignore missing shapes
    }
  });
  chart.__hazardShapeIds = [];

  markers.forEach(marker => {
    // Use new Date() to convert timestamp to JS Date; some chart libraries accept Date directly
    const shapeId = chart.createShape({
      time: new Date(marker.time),
      shape: 'square',
      color: marker.color,
      borderColor: marker.color,
      textColor: '#ffffff',
      text: (marker.hazard * 100).toFixed(0) + '%',
      placement: 'aboveBar',
      disableSelection: true,
      lock: true,
      zIndex: 100,
    });
    chart.__hazardShapeIds.push(shapeId);
  });
}
