"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { openChatStream } from "@/lib/api";
import { readSSE } from "@/lib/sse";
import { loadConversations, saveConversations } from "@/lib/storage";
import type { ChatMessage, Conversation, Source } from "@/lib/types";

const HISTORY_TURNS = 8;

const uid = (): string => `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;

const REASONS: Record<string, string> = {
  rate_limit: "a rate limit",
  unavailable: "an outage",
  timeout: "a timeout",
  connection: "a connection error",
  empty: "an empty reply",
  blocked: "a content filter",
  auth: "a rejected API key",
  not_found: "an unknown model",
  bad_request: "a rejected request",
  protocol: "a malformed reply",
  internal: "an internal error",
};

function historyFrom(messages: ChatMessage[]): { role: "user" | "assistant"; content: string }[] {
  return messages
    .filter((m) => m.status === "done" && m.confident !== false && m.content.trim().length > 0)
    .slice(-HISTORY_TURNS)
    .map((m) => ({ role: m.role, content: m.content }));
}

function newAssistant(): ChatMessage {
  return { id: uid(), role: "assistant", content: "", status: "streaming", createdAt: Date.now() };
}

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [hydrated, setHydrated] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);

  const abortRef = useRef<AbortController | null>(null);
  const conversationsRef = useRef<Conversation[]>([]);
  const activeIdRef = useRef<string | null>(null);
  conversationsRef.current = conversations;
  activeIdRef.current = activeId;

  useEffect(() => {
    setConversations(loadConversations());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    const t = setTimeout(() => saveConversations(conversations), 400);
    return () => clearTimeout(t);
  }, [conversations, hydrated]);

  const patchMessage = useCallback(
    (convId: string, msgId: string, patch: Partial<ChatMessage> | ((m: ChatMessage) => Partial<ChatMessage>)) => {
      setConversations((prev) =>
        prev.map((c) =>
          c.id !== convId
            ? c
            : {
                ...c,
                updatedAt: Date.now(),
                messages: c.messages.map((m) =>
                  m.id !== msgId ? m : { ...m, ...(typeof patch === "function" ? patch(m) : patch) },
                ),
              },
        ),
      );
    },
    [],
  );

  const runStream = useCallback(
    async (convId: string, assistantId: string, question: string, history: { role: "user" | "assistant"; content: string }[]) => {
      const controller = new AbortController();
      abortRef.current = controller;
      setIsStreaming(true);

      let pending = "";
      let raf: number | null = null;
      let finished = false;

      const flush = () => {
        raf = null;
        if (!pending) return;
        const text = pending;
        pending = "";
        patchMessage(convId, assistantId, (m) => ({ content: m.content + text }));
      };
      const schedule = () => {
        if (raf === null) raf = requestAnimationFrame(flush);
      };
      const cancelFrame = () => {
        if (raf !== null) {
          cancelAnimationFrame(raf);
          raf = null;
        }
      };
      const settle = (patch: Partial<ChatMessage>) => {
        cancelFrame();
        const rest = pending;
        pending = "";
        patchMessage(convId, assistantId, (m) => ({ ...patch, content: m.content + rest }));
      };

      try {
        const body = await openChatStream({ question, history }, controller.signal);
        for await (const ev of readSSE(body)) {
          let data: Record<string, unknown>;
          try {
            data = JSON.parse(ev.data) as Record<string, unknown>;
          } catch {
            continue;
          }
          switch (ev.event) {
            case "sources":
              patchMessage(convId, assistantId, {
                sources: data.sources as Source[],
                confident: data.confident as boolean,
              });
              break;
            case "provider":
              patchMessage(convId, assistantId, { provider: `${String(data.provider)} · ${String(data.model)}` });
              break;
            case "token":
              pending += String(data.text ?? "");
              schedule();
              break;
            case "restart": {
              cancelFrame();
              pending = "";
              const why = REASONS[String(data.reason)] ?? "an error";
              patchMessage(convId, assistantId, {
                content: "",
                notice: `Switched to ${String(data.to)} after ${why}.`,
              });
              break;
            }
            case "done":
              finished = true;
              settle({ status: "done" });
              break;
            case "error":
              finished = true;
              settle({ status: "error", error: String(data.message ?? "Something went wrong.") });
              break;
            default:
              break;
          }
        }
        if (!finished) settle({ status: "error", error: "The connection closed before the answer finished." });
      } catch (err) {
        if ((err as { name?: string }).name === "AbortError") {
          settle({ status: "stopped" });
        } else {
          settle({
            status: "error",
            error: err instanceof Error && err.message ? err.message : "Could not reach the server.",
          });
        }
      } finally {
        if (abortRef.current === controller) abortRef.current = null;
        setIsStreaming(false);
      }
    },
    [patchMessage],
  );

  const send = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || abortRef.current) return;

      const existing = conversationsRef.current.find((c) => c.id === activeIdRef.current);
      const history = existing ? historyFrom(existing.messages) : [];
      const user: ChatMessage = { id: uid(), role: "user", content: question, status: "done", createdAt: Date.now() };
      const assistant = newAssistant();

      if (existing) {
        const id = existing.id;
        setConversations((prev) =>
          prev.map((c) =>
            c.id === id ? { ...c, updatedAt: Date.now(), messages: [...c.messages, user, assistant] } : c,
          ),
        );
        await runStream(id, assistant.id, question, history);
      } else {
        const conv: Conversation = {
          id: uid(),
          title: question.length > 48 ? `${question.slice(0, 47)}…` : question,
          messages: [user, assistant],
          updatedAt: Date.now(),
        };
        setConversations((prev) => [conv, ...prev]);
        setActiveId(conv.id);
        activeIdRef.current = conv.id;
        await runStream(conv.id, assistant.id, question, history);
      }
    },
    [runStream],
  );

  const regenerate = useCallback(async () => {
    if (abortRef.current) return;
    const conv = conversationsRef.current.find((c) => c.id === activeIdRef.current);
    if (!conv) return;
    let idx = -1;
    conv.messages.forEach((m, i) => {
      if (m.role === "user") idx = i;
    });
    if (idx === -1) return;
    const question = conv.messages[idx].content;
    const history = historyFrom(conv.messages.slice(0, idx));
    const assistant = newAssistant();
    setConversations((prev) =>
      prev.map((c) =>
        c.id === conv.id ? { ...c, messages: [...c.messages.slice(0, idx + 1), assistant], updatedAt: Date.now() } : c,
      ),
    );
    await runStream(conv.id, assistant.id, question, history);
  }, [runStream]);

  const stop = useCallback(() => abortRef.current?.abort(), []);

  const newChat = useCallback(() => {
    abortRef.current?.abort();
    setActiveId(null);
  }, []);

  const selectChat = useCallback((id: string) => {
    abortRef.current?.abort();
    setActiveId(id);
  }, []);

  const deleteChat = useCallback((id: string) => {
    if (activeIdRef.current === id) {
      abortRef.current?.abort();
      setActiveId(null);
    }
    setConversations((prev) => prev.filter((c) => c.id !== id));
  }, []);

  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  const active = conversations.find((c) => c.id === activeId) ?? null;

  return { conversations: sorted, active, activeId, isStreaming, hydrated, send, regenerate, stop, newChat, selectChat, deleteChat };
}
