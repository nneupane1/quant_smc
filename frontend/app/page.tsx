import { TerminalApp } from "@/components/terminal-app";
import { loadTerminalSnapshot } from "@/lib/terminal-data";
import type { TerminalMode } from "@/lib/terminal-types";

export const dynamic = "force-dynamic";

export default async function HomePage({
  searchParams,
}: {
  searchParams?: Promise<{ mode?: string | string[] | undefined }>;
}) {
  const params = searchParams ? await searchParams : {};
  const rawMode = Array.isArray(params?.mode) ? params.mode[0] : params?.mode;
  const mode = (rawMode ?? "auto") as TerminalMode;
  const snapshot = await loadTerminalSnapshot(mode);
  return <TerminalApp initialSnapshot={snapshot} initialMode={mode} />;
}
