import { Fragment, type ReactNode } from "react";
import { splitRefs } from "@/lib/citations";
import type { Source } from "@/lib/types";
import CitationBadge from "./CitationBadge";

type Block =
  | { type: "heading"; level: number; text: string }
  | { type: "paragraph"; text: string }
  | { type: "code"; text: string }
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "quote"; text: string }
  | { type: "table"; header: string[]; rows: string[][] }
  | { type: "hr" };

const FENCE = /^\s*```/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const HR = /^\s*([-*_])(?:\s*\1){2,}\s*$/;
const LIST_ITEM = /^\s*([-*•]|\d+[.)])\s+(.*)$/;
const TABLE_SEP = /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/;

function splitRow(line: string): string[] {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((c) => c.trim());
}

function startsBlock(line: string, next: string | undefined): boolean {
  return (
    FENCE.test(line) ||
    HEADING.test(line) ||
    HR.test(line) ||
    LIST_ITEM.test(line) ||
    line.trimStart().startsWith(">") ||
    (line.includes("|") && next !== undefined && TABLE_SEP.test(next))
  );
}

/** Tolerant of half-written markdown: anything unfinished simply renders as plain text. */
export function parseBlocks(source: string): Block[] {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i++;
      continue;
    }
    if (FENCE.test(line)) {
      const body: string[] = [];
      i++;
      while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++]);
      i++; // closing fence (or end of input while streaming)
      blocks.push({ type: "code", text: body.join("\n") });
      continue;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2].trim() });
      i++;
      continue;
    }
    if (HR.test(line)) {
      blocks.push({ type: "hr" });
      i++;
      continue;
    }
    if (line.includes("|") && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
      const header = splitRow(line);
      const rows: string[][] = [];
      i += 2;
      while (i < lines.length && lines[i].trim() && lines[i].includes("|")) rows.push(splitRow(lines[i++]));
      blocks.push({ type: "table", header, rows });
      continue;
    }
    if (line.trimStart().startsWith(">")) {
      const quote: string[] = [];
      while (i < lines.length && lines[i].trimStart().startsWith(">")) quote.push(lines[i++].replace(/^\s*>\s?/, ""));
      blocks.push({ type: "quote", text: quote.join(" ") });
      continue;
    }
    const first = LIST_ITEM.exec(line);
    if (first) {
      const ordered = /\d/.test(first[1]);
      const items: string[] = [];
      while (i < lines.length) {
        const m = LIST_ITEM.exec(lines[i]);
        if (m && /\d/.test(m[1]) === ordered) {
          items.push(m[2]);
          i++;
        } else if (m === null && lines[i].trim() && /^\s{2,}\S/.test(lines[i]) && items.length) {
          items[items.length - 1] += ` ${lines[i].trim()}`;
          i++;
        } else break;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }
    const para: string[] = [];
    while (i < lines.length && lines[i].trim() && !(para.length > 0 && startsBlock(lines[i], lines[i + 1]))) {
      para.push(lines[i++].trim());
    }
    blocks.push({ type: "paragraph", text: para.join(" ") });
  }
  return blocks;
}

const INLINE =
  /(`[^`\n]+`)|(\*\*[^*\n]+?\*\*)|(\*[^*\s][^*\n]*?\*)|(\[S\d+(?:\s*[,;]\s*S\d+)*\])|(\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\))/g;

interface InlineCtx {
  sources: Map<string, Source>;
  onCite: (s: Source) => void;
}

function renderInline(text: string, ctx: InlineCtx, keyPrefix: string): ReactNode[] {
  const out: ReactNode[] = [];
  let last = 0;
  let n = 0;
  const re = new RegExp(INLINE.source, "g");
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index));
    const key = `${keyPrefix}-${n++}`;
    if (m[1]) {
      out.push(
        <code key={key} className="rounded bg-bubble px-1 py-0.5 font-mono text-[0.85em]">
          {m[1].slice(1, -1)}
        </code>,
      );
    } else if (m[2]) {
      out.push(<strong key={key}>{renderInline(m[2].slice(2, -2), ctx, key)}</strong>);
    } else if (m[3]) {
      out.push(<em key={key}>{renderInline(m[3].slice(1, -1), ctx, key)}</em>);
    } else if (m[4]) {
      const refs = splitRefs(m[4].slice(1, -1));
      out.push(
        <Fragment key={key}>
          {refs.map((ref) => {
            const src = ctx.sources.get(ref);
            return src ? <CitationBadge key={ref} source={src} onClick={ctx.onCite} /> : <span key={ref}>[{ref}]</span>;
          })}
        </Fragment>,
      );
    } else if (m[5]) {
      out.push(
        <a
          key={key}
          href={m[7]}
          target="_blank"
          rel="noopener noreferrer"
          className="text-accent underline underline-offset-2"
        >
          {m[6]}
        </a>,
      );
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

interface Props {
  text: string;
  sources?: Source[];
  onCite: (s: Source) => void;
}

export default function Markdown({ text, sources, onCite }: Props) {
  const ctx: InlineCtx = { sources: new Map((sources ?? []).map((s) => [s.ref, s])), onCite };
  const blocks = parseBlocks(text);
  return (
    <div className="space-y-3.5">
      {blocks.map((b, i) => {
        const key = `b${i}`;
        switch (b.type) {
          case "heading":
            return b.level <= 2 ? (
              <h2 key={key} className="mt-5 font-sans text-lg font-semibold leading-snug">
                {renderInline(b.text, ctx, key)}
              </h2>
            ) : (
              <h3 key={key} className="mt-4 font-sans text-base font-semibold leading-snug">
                {renderInline(b.text, ctx, key)}
              </h3>
            );
          case "paragraph":
            return <p key={key}>{renderInline(b.text, ctx, key)}</p>;
          case "code":
            return (
              <pre key={key} className="thin-scroll overflow-x-auto rounded-lg bg-bubble p-3 font-mono text-[13px] leading-relaxed">
                {b.text}
              </pre>
            );
          case "list": {
            const Tag = b.ordered ? "ol" : "ul";
            return (
              <Tag key={key} className={`space-y-1.5 pl-6 ${b.ordered ? "list-decimal" : "list-disc"} marker:text-muted`}>
                {b.items.map((item, j) => (
                  <li key={j} className="pl-1">
                    {renderInline(item, ctx, `${key}-${j}`)}
                  </li>
                ))}
              </Tag>
            );
          }
          case "quote":
            return (
              <blockquote key={key} className="border-l-2 border-accent/40 pl-4 text-muted">
                {renderInline(b.text, ctx, key)}
              </blockquote>
            );
          case "table":
            return (
              <div key={key} className="thin-scroll overflow-x-auto">
                <table className="w-full border-collapse font-sans text-sm">
                  <thead>
                    <tr>
                      {b.header.map((h, j) => (
                        <th key={j} className="border border-line bg-panel px-3 py-1.5 text-left font-semibold">
                          {renderInline(h, ctx, `${key}-h${j}`)}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {b.rows.map((row, r) => (
                      <tr key={r}>
                        {row.map((cell, c) => (
                          <td key={c} className="border border-line px-3 py-1.5 align-top">
                            {renderInline(cell, ctx, `${key}-${r}-${c}`)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          case "hr":
            return <hr key={key} className="border-line" />;
        }
      })}
    </div>
  );
}
