"use client";

import { PanelLeft, SquarePen } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { useChat } from "@/hooks/useChat";
import { fetchDocuments } from "@/lib/api";
import type { DocumentInfo, Source } from "@/lib/types";
import Composer from "./Composer";
import MessageView from "./MessageView";
import Sidebar from "./Sidebar";
import SourcePanel from "./SourcePanel";

export default function ChatApp() {
  const chat = useChat();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [source, setSource] = useState<Source | null>(null);
  const [documents, setDocuments] = useState<DocumentInfo[] | null>(null);
  const [documentsFailed, setDocumentsFailed] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const stickRef = useRef(true);

  useEffect(() => {
    setSidebarOpen(window.matchMedia("(min-width: 1024px)").matches);
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    fetchDocuments(controller.signal)
      .then(setDocuments)
      .catch((e: unknown) => {
        if ((e as { name?: string }).name !== "AbortError") setDocumentsFailed(true);
      });
    return () => controller.abort();
  }, []);

  const messages = chat.active?.messages ?? [];
  const last = messages[messages.length - 1];

  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickRef.current) el.scrollTop = el.scrollHeight;
  }, [messages.length, last?.content, last?.status]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (el) stickRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };

  const closeSource = useCallback(() => setSource(null), []);

  const send = (text: string) => {
    stickRef.current = true;
    void chat.send(text);
  };

  const selectChat = (id: string) => {
    chat.selectChat(id);
    setSource(null);
    stickRef.current = true;
    if (window.matchMedia("(max-width: 1023px)").matches) setSidebarOpen(false);
  };

  const newChat = () => {
    chat.newChat();
    setSource(null);
    if (window.matchMedia("(max-width: 1023px)").matches) setSidebarOpen(false);
  };

  const doc = documents && documents.length > 0 ? documents[0] : null;

  return (
    <div className="flex h-dvh overflow-hidden bg-bg">
      <Sidebar
        open={sidebarOpen}
        onToggle={() => setSidebarOpen((o) => !o)}
        conversations={chat.conversations}
        activeId={chat.activeId}
        onSelect={selectChat}
        onNew={newChat}
        onDelete={chat.deleteChat}
        documents={documents}
        documentsFailed={documentsFailed}
      />

      <main className="relative flex min-w-0 flex-1 flex-col">
        <header className="flex h-12 shrink-0 items-center gap-1 px-3">
          {!sidebarOpen && (
            <>
              <button
                type="button"
                onClick={() => setSidebarOpen(true)}
                aria-label="Open sidebar"
                className="rounded-md p-2 text-muted hover:bg-panel hover:text-ink"
              >
                <PanelLeft size={18} />
              </button>
              <button
                type="button"
                onClick={newChat}
                aria-label="New chat"
                className="rounded-md p-2 text-muted hover:bg-panel hover:text-ink"
              >
                <SquarePen size={18} />
              </button>
            </>
          )}
          <h1 className="ml-1 truncate text-sm text-muted">{chat.active?.title ?? "New chat"}</h1>
        </header>

        <div ref={scrollRef} onScroll={onScroll} className="thin-scroll flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-chat px-4 pb-6">
            {messages.length === 0 ? (
              <div className="pt-[18vh] text-center">
                <h2 className="font-serif text-3xl font-semibold tracking-tight">Ask the textbook</h2>
                <p className="mx-auto mt-3 max-w-md text-base leading-relaxed text-muted">
                  {doc ? (
                    <>
                      Answers are drawn from <span className="text-ink">{doc.title}</span> and cite the chapter and PDF
                      page.
                    </>
                  ) : (
                    "Answers are drawn from your indexed textbook and cite the chapter and PDF page."
                  )}{" "}
                  Select a citation to read the exact passage.
                </p>
              </div>
            ) : (
              messages.map((m, i) => (
                <MessageView
                  key={m.id}
                  message={m}
                  isLast={i === messages.length - 1}
                  isStreaming={chat.isStreaming}
                  onCite={setSource}
                  onRetry={() => void chat.regenerate()}
                />
              ))
            )}
          </div>
        </div>

        <Composer onSend={send} onStop={chat.stop} isStreaming={chat.isStreaming} />
      </main>

      <SourcePanel source={source} onClose={closeSource} />
    </div>
  );
}
