"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  HistogramSeries,
  LineStyle,
  createSeriesMarkers,
  createChart,
} from "lightweight-charts";

import type { TerminalSnapshot } from "@/lib/terminal-types";

type MarketPayload = TerminalSnapshot["market"];

type HoverState = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
} | null;

type ZoneRect = {
  key: string;
  left: number;
  width: number;
  top: number;
  height: number;
  color: string;
  border: string;
  label: string;
};

function zonePalette(zone: MarketPayload["zones"][number]): { fill: string; border: string } {
  if (zone.kind === "ob") {
    return zone.side === "bullish"
      ? { fill: "rgba(42, 230, 184, 0.16)", border: "rgba(42, 230, 184, 0.62)" }
      : { fill: "rgba(255, 107, 136, 0.16)", border: "rgba(255, 107, 136, 0.62)" };
  }
  if (zone.kind === "fvg") {
    return { fill: "rgba(246, 182, 60, 0.16)", border: "rgba(246, 182, 60, 0.62)" };
  }
  return { fill: "rgba(82, 215, 255, 0.16)", border: "rgba(82, 215, 255, 0.62)" };
}

export function MarketCanvas({
  market,
  onHoverTime,
  focusTime,
}: {
  market: MarketPayload;
  onHoverTime?: (time: number | null) => void;
  focusTime?: number | null;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<any>(null);
  const candleRef = useRef<any>(null);
  const volumeRef = useRef<any>(null);
  const markerApiRef = useRef<any>(null);
  const recalcRef = useRef<() => void>(() => {});
  const zonesRef = useRef(market.zones ?? []);
  const hoverCallbackRef = useRef(onHoverTime);
  const hasFitRef = useRef(false);
  const [zoneRects, setZoneRects] = useState<ZoneRect[]>([]);
  const [hover, setHover] = useState<HoverState>(null);

  const focusLabel = useMemo(() => {
    if (!focusTime) return null;
    const ts = new Date(focusTime * 1000);
    if (Number.isNaN(ts.getTime())) return null;
    return ts.toLocaleString("en-GB", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  }, [focusTime]);

  useEffect(() => {
    zonesRef.current = market.zones ?? [];
    recalcRef.current();
  }, [market.zones]);

  useEffect(() => {
    hoverCallbackRef.current = onHoverTime;
  }, [onHoverTime]);

  useEffect(() => {
    const host = containerRef.current;
    if (!host) return;

    const chart = createChart(host, {
      width: host.clientWidth,
      height: Math.max(560, host.clientHeight || 620),
      layout: {
        background: { type: ColorType.Solid, color: "rgba(6, 14, 24, 0.98)" },
        textColor: "rgba(222, 231, 246, 0.88)",
        attributionLogo: false,
      },
      grid: {
        vertLines: { color: "rgba(82, 215, 255, 0.08)", style: LineStyle.Dotted },
        horzLines: { color: "rgba(82, 215, 255, 0.08)", style: LineStyle.Dotted },
      },
      rightPriceScale: {
        borderColor: "rgba(82, 215, 255, 0.20)",
        scaleMargins: { top: 0.06, bottom: 0.24 },
      },
      timeScale: {
        borderColor: "rgba(82, 215, 255, 0.20)",
        timeVisible: true,
        secondsVisible: false,
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "rgba(82, 215, 255, 0.45)", width: 1, style: LineStyle.Solid, labelBackgroundColor: "#0f2438" },
        horzLine: { color: "rgba(82, 215, 255, 0.45)", width: 1, style: LineStyle.Solid, labelBackgroundColor: "#0f2438" },
      },
      localization: {
        locale: "en-GB",
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#2ae6b8",
      downColor: "#ff6b88",
      borderUpColor: "#2ae6b8",
      borderDownColor: "#ff6b88",
      wickUpColor: "#2ae6b8",
      wickDownColor: "#ff6b88",
      priceLineVisible: true,
      lastValueVisible: true,
    });

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceScaleId: "",
      priceFormat: { type: "volume" },
      base: 0,
      color: "rgba(82, 215, 255, 0.22)",
    });
    volumeSeries.priceScale().applyOptions({
      scaleMargins: { top: 0.82, bottom: 0.0 },
      borderVisible: false,
    });

    chart.subscribeCrosshairMove((param) => {
      const candleData = param.seriesData.get(candleSeries) as
        | { open?: number; high?: number; low?: number; close?: number }
        | undefined;
      const volumeData = param.seriesData.get(volumeSeries) as { value?: number } | undefined;
      const ts = typeof param.time === "number" ? param.time : null;
      if (!candleData || ts === null) {
        setHover(null);
        hoverCallbackRef.current?.(null);
        return;
      }
      setHover({
        time: ts,
        open: Number(candleData.open ?? 0),
        high: Number(candleData.high ?? 0),
        low: Number(candleData.low ?? 0),
        close: Number(candleData.close ?? 0),
        volume: Number(volumeData?.value ?? 0),
      });
      hoverCallbackRef.current?.(ts);
    });

    chartRef.current = chart;
    candleRef.current = candleSeries;
    volumeRef.current = volumeSeries;
    markerApiRef.current = createSeriesMarkers(candleSeries, []);
    recalcRef.current = () => {
      const zones = zonesRef.current;
      if (!zones.length) {
        setZoneRects([]);
        return;
      }
      const rows: ZoneRect[] = [];
      for (const zone of zones) {
        const x1 = chart.timeScale().timeToCoordinate(zone.start as never);
        const x2 = chart.timeScale().timeToCoordinate(zone.end as never);
        const y1 = candleSeries.priceToCoordinate(zone.top);
        const y2 = candleSeries.priceToCoordinate(zone.bottom);
        if ([x1, x2, y1, y2].some((value) => value === null || value === undefined)) continue;
        const left = Math.min(Number(x1), Number(x2));
        const width = Math.max(2, Math.abs(Number(x2) - Number(x1)));
        const top = Math.min(Number(y1), Number(y2));
        const height = Math.max(2, Math.abs(Number(y2) - Number(y1)));
        const colors = zonePalette(zone);
        rows.push({
          key: `${zone.kind}-${zone.side}-${zone.start}-${zone.end}-${zone.label}`,
          left,
          width,
          top,
          height,
          color: colors.fill,
          border: colors.border,
          label: zone.label,
        });
      }
      setZoneRects(rows);
    };

    const onVisibleRange = () => recalcRef.current();
    chart.timeScale().subscribeVisibleTimeRangeChange(onVisibleRange);

    const observer = new ResizeObserver((entries) => {
      const width = entries[0]?.contentRect.width ?? host.clientWidth;
      const height = Math.max(560, entries[0]?.contentRect.height || host.clientHeight || 620);
      chart.applyOptions({ width, height });
      recalcRef.current();
    });
    observer.observe(host);

    return () => {
      chart.timeScale().unsubscribeVisibleTimeRangeChange(onVisibleRange);
      observer.disconnect();
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      volumeRef.current = null;
      markerApiRef.current = null;
      recalcRef.current = () => {};
      hasFitRef.current = false;
      setZoneRects([]);
      hoverCallbackRef.current?.(null);
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleRef.current;
    const volumeSeries = volumeRef.current;
    if (!chart || !candleSeries || !volumeSeries) return;

    const candleData = market.candles.map((row) => ({
      time: row.time as never,
      open: row.open,
      high: row.high,
      low: row.low,
      close: row.close,
    }));
    const volumeData = market.candles.map((row) => ({
      time: row.time as never,
      value: row.volume,
      color: row.close >= row.open ? "rgba(42,230,184,0.35)" : "rgba(255,107,136,0.35)",
    }));

    candleSeries.setData(candleData);
    volumeSeries.setData(volumeData);
    markerApiRef.current?.setMarkers(market.markers.map((row) => ({ ...row, time: row.time as never })) as never);
    if (!hasFitRef.current && candleData.length > 10) {
      chart.timeScale().fitContent();
      hasFitRef.current = true;
    }
    recalcRef.current();
  }, [market]);

  const latest = market.candles[market.candles.length - 1];
  const active = hover ?? (latest ? {
    time: latest.time,
    open: latest.open,
    high: latest.high,
    low: latest.low,
    close: latest.close,
    volume: latest.volume,
  } : null);
  const ohlcDelta = active ? active.close - active.open : 0;
  const ohlcPct = active && active.open ? (ohlcDelta / active.open) * 100 : 0;

  return (
    <div className="relative overflow-hidden rounded-[20px] border border-white/10 bg-[#060e18]">
      <div className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan/70 to-transparent" />
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
        <div className="text-xs uppercase tracking-[0.2em] text-slate-400">
          {market.symbol} {market.timeframe} • drag to pan • wheel to zoom
        </div>
        <div className="flex flex-wrap items-center gap-3 font-mono text-xs">
          <span className="text-slate-400">O</span><span className="text-white">{active ? active.open.toFixed(2) : "-"}</span>
          <span className="text-slate-400">H</span><span className="text-teal">{active ? active.high.toFixed(2) : "-"}</span>
          <span className="text-slate-400">L</span><span className="text-rose">{active ? active.low.toFixed(2) : "-"}</span>
          <span className="text-slate-400">C</span><span className="text-cyan">{active ? active.close.toFixed(2) : "-"}</span>
          <span className={ohlcDelta >= 0 ? "text-teal" : "text-rose"}>
            {ohlcDelta >= 0 ? "+" : ""}{ohlcDelta.toFixed(2)} ({ohlcPct >= 0 ? "+" : ""}{ohlcPct.toFixed(2)}%)
          </span>
          <span className="text-slate-400">V</span><span className="text-amber">{active ? active.volume.toFixed(0) : "-"}</span>
          {focusLabel ? (
            <>
              <span className="text-slate-400">Focus</span><span className="text-cyan">{focusLabel}</span>
            </>
          ) : null}
        </div>
      </div>
      <div className="relative">
        <div ref={containerRef} className="h-[640px] w-full" />
        <div className="pointer-events-none absolute inset-0">
          {zoneRects.map((zone) => (
            <div
              key={zone.key}
              className="absolute rounded-md border"
              style={{
                left: `${zone.left}px`,
                width: `${zone.width}px`,
                top: `${zone.top}px`,
                height: `${zone.height}px`,
                backgroundColor: zone.color,
                borderColor: zone.border,
              }}
            >
              <span className="absolute left-1 top-0 -translate-y-full rounded bg-black/60 px-1 py-0.5 text-[10px] uppercase tracking-[0.12em] text-slate-200">
                {zone.label}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
