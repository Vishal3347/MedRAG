"use client";

import { Check, Copy, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { fetchChunk } from "@/lib/api";
import { citationLabel } from "@/lib/citations";
import type { ChunkDetail, Source } from "@/lib/types";

interface Props {
  source: Source | null;
  onClose: () => void;
}

export default function SourcePanel({ source, onClose }: Props) {
  const [detail, setDetail] = useState<ChunkDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const cache = useRef(new Map<string, ChunkDetail>());
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!source) return;
    setError(null);
    setCopied(false);
    const hit = cache.current.get(source.chunk_id);
    if (hit) {
      setDetail(hit);
      setLoading(false);
      return;
    }
    setDetail(null);
    setLoading(true);
    const controller = new AbortController();
    fetchChunk(source.chunk_id, controller.signal)
      .then((d) => {
        cache.current.set(source.chunk_id, d);
        setDetail(d);
      })
      .catch((e: unknown) => {
        if ((e as { name?: string }).name !== "AbortError") setError("Could not load the passage text.");
      })
      .finally(() => setLoading(false));
    return () => controller.abort();
  }, [source]);

  useEffect(() => {
    if (!source) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    closeRef.current?.focus();
    return () => window.removeEventListener("keydown", onKey);
  }, [source, onClose]);

  const copy = async () => {
    if (!detail) return;
    try {
      await navigator.clipboard.writeText(detail.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard unavailable */
    }
  };

  const open = source !== null;
  const paragraphs = detail ? detail.text.split(/\n{2,}/) : [];

  return (
    <>
      <div
        className={`fixed inset-0 z-30 bg-ink/30 transition-opacity lg:hidden ${open ? "opacity-100" : "pointer-events-none opacity-0"}`}
        onClick={onClose}
        aria-hidden="true"
      />
      <aside
        role="dialog"
        aria-label="Source passage"
        aria-hidden={!open}
        className={`fixed inset-y-0 right-0 z-40 flex w-full max-w-[460px] flex-col border-l border-line bg-bg shadow-xl transition-transform duration-200 ${open ? "translate-x-0" : "pointer-events-none translate-x-full"}`}
      >
        <div className="flex items-start gap-3 border-b border-line px-5 py-4">
          <div className="min-w-0 flex-1">
            <p className="text-xs text-muted">Source passage</p>
            <h2 className="mt-0.5 text-base font-semibold leading-snug">{source ? citationLabel(source) : ""}</h2>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            aria-label="Close source panel"
            className="rounded-md p-1.5 text-muted hover:bg-panel hover:text-ink"
          >
            <X size={18} />
          </button>
        </div>

        <div className="thin-scroll flex-1 overflow-y-auto px-5 py-4">
          {source && (
            <dl className="mb-4 space-y-1 text-sm">
              <div className="flex gap-2">
                <dt className="w-16 shrink-0 text-muted">Book</dt>
                <dd>{source.document_title}</dd>
              </div>
              {source.chapter && (
                <div className="flex gap-2">
                  <dt className="w-16 shrink-0 text-muted">Chapter</dt>
                  <dd>{source.chapter}</dd>
                </div>
              )}
              {source.section_heading && (
                <div className="flex gap-2">
                  <dt className="w-16 shrink-0 text-muted">Section</dt>
                  <dd>{source.section_heading}</dd>
                </div>
              )}
            </dl>
          )}

          {loading && <p className="text-sm text-muted">Loading passage…</p>}
          {error && <p className="text-sm text-danger">{error}</p>}
          {detail && (
            <div className="rounded-lg border border-line bg-panel p-4 font-serif text-[15px] leading-[1.7]">
              {paragraphs.map((p, i) =>
                p.startsWith("## ") ? (
                  <p key={i} className="mt-3 font-sans text-sm font-semibold first:mt-0">
                    {p.slice(3)}
                  </p>
                ) : (
                  <p key={i} className="mt-3 whitespace-pre-wrap first:mt-0">
                    {p}
                  </p>
                ),
              )}
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 border-t border-line px-5 py-3">
          <p className="flex-1 text-xs leading-snug text-muted">
            Exact text stored in the index. Page numbers are PDF page positions, which can differ from the printed
            page numbers.
          </p>
          <button
            type="button"
            onClick={copy}
            disabled={!detail}
            className="inline-flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-xs font-medium hover:bg-panel disabled:opacity-40"
          >
            {copied ? <Check size={14} /> : <Copy size={14} />}
            {copied ? "Copied" : "Copy"}
          </button>
        </div>
      </aside>
    </>
  );
}
