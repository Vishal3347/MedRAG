"use client";

import { BookOpen, PanelLeft, SquarePen, Trash2 } from "lucide-react";
import type { Conversation, DocumentInfo } from "@/lib/types";

interface Props {
  open: boolean;
  onToggle: () => void;
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  documents: DocumentInfo[] | null;
  documentsFailed: boolean;
}

export default function Sidebar({
  open,
  onToggle,
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  documents,
  documentsFailed,
}: Props) {
  return (
    <>
      <div
        className={`fixed inset-0 z-30 bg-ink/30 transition-opacity lg:hidden ${open ? "opacity-100" : "pointer-events-none opacity-0"}`}
        onClick={onToggle}
        aria-hidden="true"
      />
      <aside
        aria-label="Conversations"
        aria-hidden={!open}
        className={`fixed inset-y-0 left-0 z-40 flex w-[280px] flex-col border-r border-line bg-panel transition-transform duration-200 lg:static lg:z-auto lg:shrink-0 lg:transition-[margin] ${
          open ? "translate-x-0 lg:ml-0" : "pointer-events-none -translate-x-full lg:-ml-[280px] lg:translate-x-0"
        }`}
      >
        <div className="flex items-center justify-between px-3 py-3">
          <button
            type="button"
            onClick={onToggle}
            aria-label="Close sidebar"
            className="rounded-md p-2 text-muted hover:bg-line/50 hover:text-ink"
          >
            <PanelLeft size={18} />
          </button>
          <button
            type="button"
            onClick={onNew}
            aria-label="New chat"
            className="rounded-md p-2 text-muted hover:bg-line/50 hover:text-ink"
          >
            <SquarePen size={18} />
          </button>
        </div>

        <nav className="thin-scroll flex-1 overflow-y-auto px-2 pb-2">
          {conversations.length === 0 ? (
            <p className="px-3 py-4 text-sm text-muted">Your conversations will appear here.</p>
          ) : (
            <ul className="space-y-0.5">
              {conversations.map((c) => (
                <li key={c.id} className="group relative">
                  <button
                    type="button"
                    onClick={() => onSelect(c.id)}
                    aria-current={c.id === activeId ? "true" : undefined}
                    className={`w-full truncate rounded-lg px-3 py-2 pr-9 text-left text-sm ${
                      c.id === activeId ? "bg-line/70 font-medium" : "hover:bg-line/40"
                    }`}
                  >
                    {c.title}
                  </button>
                  <button
                    type="button"
                    onClick={() => onDelete(c.id)}
                    aria-label={`Delete conversation: ${c.title}`}
                    className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded-md p-1.5 text-muted opacity-0 hover:text-danger focus-visible:opacity-100 group-hover:opacity-100"
                  >
                    <Trash2 size={15} />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </nav>

        <div className="border-t border-line p-3">
          <div className="flex items-start gap-2.5 rounded-lg bg-bg/60 p-3">
            <BookOpen size={18} className="mt-0.5 shrink-0 text-accent" />
            <div className="min-w-0 text-sm">
              {documents === null && !documentsFailed && <p className="text-muted">Loading document…</p>}
              {documentsFailed && <p className="text-danger">Could not reach the server.</p>}
              {documents !== null && documents.length === 0 && (
                <p className="text-muted">No document is indexed yet. Run the ingestion script.</p>
              )}
              {documents?.map((d) => (
                <div key={d.document_id}>
                  <p className="line-clamp-2 font-medium leading-snug">{d.title}</p>
                  <p className="mt-0.5 text-xs text-muted">
                    PDF pp. {d.page_start}–{d.page_end} · {d.chapters} chapters · {d.chunks} passages
                  </p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
