import { loadTerminalSnapshot } from "@/lib/terminal-data";
import type { TerminalMode } from "@/lib/terminal-types";

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const mode = (url.searchParams.get("mode") ?? "auto") as TerminalMode;
  const snapshot = await loadTerminalSnapshot(mode);
  return Response.json(snapshot, {
    headers: {
      "Cache-Control": "no-store, max-age=0",
    },
  });
}
