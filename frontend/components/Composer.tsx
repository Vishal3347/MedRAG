"use client";

import { ArrowUp, Square } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";

interface Props {
  onSend: (text: string) => void;
  onStop: () => void;
  isStreaming: boolean;
}

export default function Composer({ onSend, onStop, isStreaming }: Props) {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [text]);

  const submit = () => {
    const value = text.trim();
    if (!value || isStreaming) return;
    onSend(value);
    setText("");
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      submit();
    }
  };

  return (
    <div className="px-4 pb-3 pt-2">
      <div className="mx-auto w-full max-w-chat">
        <div className="flex items-end gap-2 rounded-3xl border border-line bg-panel py-2 pl-5 pr-2 shadow-sm focus-within:border-accent/60">
          <textarea
            ref={ref}
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={onKeyDown}
            rows={1}
            maxLength={2000}
            placeholder="Ask a question about the textbook"
            aria-label="Message"
            className="max-h-[200px] flex-1 resize-none bg-transparent py-1.5 text-base leading-6 outline-none placeholder:text-muted"
          />
          {isStreaming ? (
            <button
              type="button"
              onClick={onStop}
              aria-label="Stop generating"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-ink text-bg"
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              type="button"
              onClick={submit}
              disabled={!text.trim()}
              aria-label="Send message"
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-accent text-bg disabled:opacity-30"
            >
              <ArrowUp size={18} />
            </button>
          )}
        </div>
        <p className="mt-2 text-center text-xs text-muted">
          Answers come only from the indexed textbook and can still contain errors. Verify before any clinical use.
        </p>
      </div>
    </div>
  );
}
