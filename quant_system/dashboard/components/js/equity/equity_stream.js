/*
 * Equity Stream module
 *
 * This module handles streaming of equity data from the backend to the front-end.
 * It uses Server-Sent Events (EventSource) to subscribe to a stream of JSON
 * objects containing equity metrics (timestamp, equity, pnl, drawdown, etc.).
 * The consumer passes a callback which will be invoked for each event with the
 * parsed data. The module also exposes a function to disconnect.
 */

export function subscribeEquityStream(callback, options = {}) {
  const endpoint = options.endpoint || '/api/equity_stream';
  const reconnectInterval = options.reconnectInterval || 5000;

  let eventSource = null;
  let closed = false;

  function connect() {
    if (closed) {
      return;
    }
    eventSource = new EventSource(endpoint);

    eventSource.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        callback(data);
      } catch (err) {
        console.error('Failed to parse equity stream event:', err);
      }
    };

    eventSource.onerror = () => {
      console.warn('Equity stream disconnected, attempting to reconnect...');
      if (eventSource) {
        eventSource.close();
        eventSource = null;
      }
      setTimeout(connect, reconnectInterval);
    };
  }

  connect();

  return function unsubscribe() {
  closed = true;
    if (eventSource) {
      eventSource.close();
      eventSource = null;
    }
  };
}

/*
 * Optionally, if SSE is not available or disabled, you can fetch the equity
 * data periodically by using the pollEquity function below.
 */
export async function pollEquity(callback, options = {}) {
  const endpoint = options.endpoint || '/api/equity_snapshot';
  const interval = options.interval || 5000;
  let stopped = false;

  async function poll() {
    if (stopped) {
      return;
    }
    try {
      const resp = await fetch(endpoint);
      if (resp.ok) {
        const data = await resp.json();
        callback(data);
      } else {
        console.error('Failed to fetch equity snapshot:', resp.status);
      }
    } catch (err) {
      console.error('Error polling equity snapshot:', err);
    } finally {
      setTimeout(poll, interval);
    }
  }

  poll();

  return function stop() {
    stopped = true;
  };
}

}
