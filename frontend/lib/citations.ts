import type { Source } from "./types";

/** Matches [S1], [S1, S2] and [S1; S3]. */
export const CITATION_RE = /\[(S\d+(?:\s*[,;]\s*S\d+)*)\]/;

export function splitRefs(group: string): string[] {
  return group.split(/\s*[,;]\s*/).map((s) => s.trim());
}

function chapterLabel(chapter: string): string {
  const m = /^(chapter|part|unit)\s+([\w.]+)/i.exec(chapter.trim());
  if (m) {
    const kind = m[1].charAt(0).toUpperCase() + m[1].slice(1).toLowerCase();
    return `${kind} ${m[2].replace(/[.:]$/, "")}`;
  }
  const t = chapter.trim();
  if (!t) return "";
  return t.length > 30 ? `${t.slice(0, 29)}…` : t;
}

export function pageLabel(s: Pick<Source, "page_start" | "page_end">): string {
  return s.page_start === s.page_end ? `PDF p. ${s.page_start}` : `PDF pp. ${s.page_start}–${s.page_end}`;
}

/** "Chapter 12, PDF p. 345" */
export function citationLabel(s: Source): string {
  const ch = chapterLabel(s.chapter);
  return ch ? `${ch}, ${pageLabel(s)}` : pageLabel(s);
}
