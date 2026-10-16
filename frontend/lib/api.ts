import type { ChatRequestBody, ChunkDetail, DocumentInfo } from "./types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function getJSON<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, { signal, headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`Request failed (${res.status})`);
  return (await res.json()) as T;
}

export async function fetchDocuments(signal?: AbortSignal): Promise<DocumentInfo[]> {
  const data = await getJSON<{ documents: DocumentInfo[] }>("/api/v1/documents", signal);
  return data.documents;
}

export async function fetchChunk(id: string, signal?: AbortSignal): Promise<ChunkDetail> {
  return getJSON<ChunkDetail>(`/api/v1/sources/${encodeURIComponent(id)}`, signal);
}

export class ChatHTTPError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export async function openChatStream(body: ChatRequestBody, signal: AbortSignal): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch(`${API_BASE}/api/v1/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let detail = `The server returned ${res.status}.`;
    try {
      const j = (await res.json()) as { detail?: unknown };
      if (typeof j.detail === "string") detail = j.detail;
    } catch {
      /* ignore */
    }
    throw new ChatHTTPError(detail, res.status);
  }
  return res.body;
}
