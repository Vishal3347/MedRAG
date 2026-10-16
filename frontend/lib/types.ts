export type Role = "user" | "assistant";

export interface Source {
  ref: string; // "S1"
  chunk_id: string;
  document_id: string;
  document_title: string;
  chapter: string;
  section_heading: string;
  page_start: number;
  page_end: number;
  snippet: string;
  score: number | null;
}

export type MessageStatus = "streaming" | "done" | "error" | "stopped";

export interface ChatMessage {
  id: string;
  role: Role;
  content: string;
  status: MessageStatus;
  createdAt: number;
  sources?: Source[];
  /** false when retrieval found no sufficiently relevant passage (no model was called) */
  confident?: boolean;
  provider?: string;
  /** transient notice, e.g. "Switched to gemini after a rate limit" */
  notice?: string;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  updatedAt: number;
}

export interface DocumentInfo {
  document_id: string;
  title: string;
  chunks: number;
  page_start: number;
  page_end: number;
  chapters: number;
}

export interface ChunkDetail {
  id: string;
  text: string;
  document_title: string;
  chapter: string;
  section_heading: string;
  page_start: number;
  page_end: number;
}

export interface ChatRequestBody {
  question: string;
  history: { role: Role; content: string }[];
  document_id?: string;
}
