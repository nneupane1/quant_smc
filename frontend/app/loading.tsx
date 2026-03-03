export default function Loading() {
  return (
    <main className="terminal-shell flex min-h-screen items-center justify-center">
      <div className="glass-panel flex items-center gap-4 px-8 py-6">
        <div className="h-3 w-3 animate-pulse rounded-full bg-cyan shadow-glow" />
        <div>
          <div className="section-kicker">Booting</div>
          <div className="mt-1 text-lg font-semibold text-white">Loading terminal state</div>
        </div>
      </div>
    </main>
  );
}
