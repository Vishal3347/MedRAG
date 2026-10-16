import { citationLabel } from "@/lib/citations";
import type { Source } from "@/lib/types";

interface Props {
  source: Source;
  onClick: (source: Source) => void;
}

/** Inline citation: "Chapter 12, PDF p. 345". Opens the source inspector. */
export default function CitationBadge({ source, onClick }: Props) {
  const label = citationLabel(source);
  return (
    <button
      type="button"
      onClick={() => onClick(source)}
      title={source.section_heading ? `${source.section_heading}` : source.document_title}
      aria-label={`Open source: ${label}`}
      className="mx-0.5 inline-flex items-center whitespace-nowrap rounded-md border border-accent/30 bg-accent-soft px-1.5 py-px align-baseline font-sans text-[11px] font-medium leading-[1.15rem] text-accent transition-colors hover:border-accent/70"
    >
      {label}
    </button>
  );
}
