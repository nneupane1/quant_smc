import { loadTerminalSnapshot } from "@/lib/terminal-data";

export const dynamic = "force-dynamic";

export async function GET() {
  const snapshot = await loadTerminalSnapshot();
  return Response.json(snapshot, {
    headers: {
      "Cache-Control": "no-store, max-age=0",
    },
  });
}
