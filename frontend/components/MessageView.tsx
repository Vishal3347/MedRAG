"use client";

import { Check, Copy, RotateCcw } from "lucide-react";
import { useState } from "react";
import { citationLabel } from "@/lib/citations";
import type { ChatMessage, Source } from "@/lib/types";
import Markdown from "./Markdown";

interface Props {
  message: ChatMessage;
  isLast: boolean;
  isStreaming: boolean;
  onCite: (s: Source) => void;
  onRetry: () => void;
}

function Pending({ message }: { message: ChatMessage }) {
  const label = message.sources
    ? message.sources.length > 0
      ? `Reading ${message.sources.length} passages`
      : "Looking for evidence"
    : "Searching the textbook";
  return (
    <p className="flex items-center gap-2 font-sans text-sm text-muted" role="status">
      <span className="h-2 w-2 animate-pulse rounded-full bg-accent" aria-hidden="true" />
      {label}…
    </p>
  );
}

export default function MessageView({ message, isLast, isStreaming, onCite, onRetry }: Props) {
  const [copied, setCopied] = useState(false);

  if (message.role === "user") {
    return (
      <div className="flex justify-end py-3">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-3xl bg-bubble px-4 py-2.5 text-base leading-6">
          {message.content}
        </div>
      </div>
    );
  }

  const streaming = message.status === "streaming";
  const showSources = !streaming && message.sources && message.sources.length > 0;
  const canRetry = isLast && !isStreaming && message.status !== "streaming";

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(message.content.replace(/\s?\[S\d+(?:\s*[,;]\s*S\d+)*\]/g, ""));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <article className="py-4" aria-busy={streaming}>
      {message.notice && streaming && (
        <p className="mb-2 font-sans text-xs text-muted" role="status">
          {message.notice}
        </p>
      )}

      {message.content ? (
        <div className="font-serif text-[17px] leading-[1.7]">
          <Markdown text={message.content} sources={message.sources} onCite={onCite} />
        </div>
      ) : (
        streaming && <Pending message={message} />
      )}

      {message.status === "error" && (
        <p className="mt-3 font-sans text-sm text-danger" role="alert">
          {message.error ?? "Something went wrong."}
        </p>
      )}
      {message.status === "stopped" && <p className="mt-2 font-sans text-xs text-muted">Stopped.</p>}

      {showSources && (
        <div className="mt-4 font-sans">
          <p className="mb-1.5 text-xs text-muted">
            {message.confident === false ? "Closest passages (low relevance)" : "Sources"}
          </p>
          <ul className="flex flex-wrap gap-1.5">
            {message.sources!.map((s) => (
              <li key={s.chunk_id}>
                <button
                  type="button"
                  onClick={() => onCite(s)}
                  className="rounded-lg border border-line px-2.5 py-1 text-xs hover:border-accent/60 hover:text-accent"
                  title={s.snippet}
                >
                  <span className="mr-1.5 font-semibold text-accent">{s.ref}</span>
                  {citationLabel(s)}
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {!streaming && (
        <div className="mt-3 flex items-center gap-1 font-sans text-xs text-muted">
          {message.content && (
            <button
              type="button"
              onClick={copy}
              aria-label="Copy answer"
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 hover:bg-panel hover:text-ink"
            >
              {copied ? <Check size={14} /> : <Copy size={14} />}
              {copied ? "Copied" : "Copy"}
            </button>
          )}
          {canRetry && (
            <button
              type="button"
              onClick={onRetry}
              aria-label="Regenerate answer"
              className="inline-flex items-center gap-1 rounded-md px-2 py-1 hover:bg-panel hover:text-ink"
            >
              <RotateCcw size={14} />
              Retry
            </button>
          )}
          {message.provider && <span className="ml-2">{message.provider}</span>}
          {message.notice && <span className="ml-2">{message.notice}</span>}
        </div>
      )}
    </article>
  );
}
