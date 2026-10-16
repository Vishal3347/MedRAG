import type { Conversation } from "./types";

const KEY = "medrag.conversations.v1";
const MAX_CONVERSATIONS = 50;

export function loadConversations(): Conversation[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // A reload mid-stream leaves a message stuck in "streaming": mark it stopped.
    return (parsed as Conversation[]).map((c) => ({
      ...c,
      messages: c.messages.map((m) => (m.status === "streaming" ? { ...m, status: "stopped" as const } : m)),
    }));
  } catch {
    return [];
  }
}

export function saveConversations(conversations: Conversation[]): void {
  try {
    const trimmed = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CONVERSATIONS);
    window.localStorage.setItem(KEY, JSON.stringify(trimmed));
  } catch {
    /* storage full or unavailable: history is a convenience, never fatal */
  }
}
