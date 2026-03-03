import { TerminalApp } from "@/components/terminal-app";
import { loadTerminalSnapshot } from "@/lib/terminal-data";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const snapshot = await loadTerminalSnapshot();
  return <TerminalApp initialSnapshot={snapshot} />;
}
