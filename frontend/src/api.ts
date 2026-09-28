export type Model = "gpt-6-sol" | "gpt-6-luna";
export type Variant = "V0" | "V1" | "V2" | "V3";
export type SourceStatus =
  | "pending"
  | "processing"
  | "ready"
  | "failed"
  | "deleted";
export type RunStatus =
  | "queued"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled"
  | "interrupted";

export interface Collection {
  id: string;
  title: string;
  description: string;
  created_at: string;
  revision: number;
  source_count: number;
  ready_count: number;
  unavailable_count: number;
  has_pending_sources: boolean;
}
export interface Source {
  id: string;
  collection_id: string;
  title: string;
  filename: string;
  media_type: string;
  document_class: string;
  status: SourceStatus;
  sha256: string;
  byte_count: number;
  page_count: number | null;
  error: string | null;
  created_at: string;
  deleted_at: string | null;
  file_key: string;
}
export interface Span {
  id: string;
  source_id: string;
  source_title?: string;
  source_hash?: string;
  page: number | null;
  line_start: number | null;
  line_end: number | null;
  text: string;
  bbox: [number, number, number, number] | null;
  page_width: number | null;
  page_height: number | null;
  sha256: string;
  section: string | null;
}
export interface Message {
  id: string;
  conversation_id: string;
  run_id: string | null;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}
export interface Conversation {
  id: string;
  collection_id: string;
  title: string;
  created_at: string;
  revision: number;
}
export interface Answer {
  kind: string;
  status:
    | "supported"
    | "partial"
    | "conflicting"
    | "not_documented"
    | "needs_clarification";
  answer: string;
  claims: { text: string; evidence_ids: string[] }[];
  citations?: Span[];
  limitations: string[];
  coverage_requirement: string;
}
export interface Run {
  id: string;
  conversation_id: string;
  request_id: string;
  question: string;
  model: Model;
  variant: Variant;
  status: RunStatus;
  scope: {
    id: string;
    collection_id: string;
    revision: number;
    source_count: number;
    ready_count: number;
    unavailable_count: number;
    snapshot_hash: string;
  } | null;
  answer: Answer | null;
  error: { code?: string; message?: string } | null;
  trace_id: string | null;
  trace_url: string | null;
  trace_status: string | null;
  metrics: Record<string, unknown> | null;
  created_at: string;
  finished_at: string | null;
  retry_of_run_id: string | null;
}
export interface RunPage {
  items: Run[];
  next_cursor: string | null;
}
export interface RunEvent {
  run_id: string;
  seq: number;
  type: string;
  at: string;
  payload: Record<string, unknown>;
}
export interface Experiment {
  id: string;
  campaign_id: string;
  name: string;
  status: string;
  planned: number;
  completed: number;
  model: string;
  variant: string;
  metrics: Record<string, unknown> | null;
  latest_correction: {
    attempted_rows: number;
    completed: number;
    state_counts: Record<string, number>;
    confirmed_grounded_success: {
      numerator: number;
      denominator: number;
      value: number | null;
    };
  };
  created_at: string;
  cases?: ExperimentCase[];
}
export interface ExperimentCase {
  case_id: string;
  family_id: string;
  status: string;
  primary: {
    attempt_id: string | null;
    run_ids: string[];
    judge_request_id: string | null;
    judge_verdict: string | null;
  };
  latest: {
    attempt_id: string | null;
    state: string;
    judge_status: string | null;
    judge_verdict: string | null;
    retry_reason: string | null;
    run_ids: string[];
    judge_request_id: string | null;
  };
}
export interface Stats {
  total_runs: number;
  runtime_status_counts: Record<string, number>;
  answer_status_counts: Record<string, number>;
  trace_status_counts: Record<string, number>;
  terminal_trace_status_counts: Record<string, number>;
  latency_p50_ms: number | null;
  latency_n: number;
  latency_measure: string;
  observed_at: string;
}
export interface Health {
  status: string;
  provider: string;
  model: string;
  langfuse_url: string | null;
  version: string;
}
export interface Readiness {
  provider: string;
  ready: boolean;
  connection: string;
  configuration: string;
  inference: "unverified";
  reason: string;
}

let csrfToken = "";

export class ApiError extends Error {
  constructor(
    public code: string,
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (
    init.body &&
    !(init.body instanceof FormData) &&
    !headers.has("Content-Type")
  )
    headers.set("Content-Type", "application/json");
  if (init.method && !["GET", "HEAD"].includes(init.method.toUpperCase()))
    headers.set("X-CSRF-Token", csrfToken);
  const response = await fetch(`/api${path}`, {
    ...init,
    headers,
    credentials: "include",
  });
  if (!response.ok) {
    if (response.status === 401 && path !== "/session")
      window.dispatchEvent(new Event("evidence-lab:session-expired"));
    let detail: { code?: string; message?: string } | string | undefined;
    try {
      detail = (await response.json()).detail;
    } catch {
      detail = undefined;
    }
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message || response.statusText;
    const code =
      typeof detail === "string"
        ? "request_failed"
        : detail?.code || "request_failed";
    throw new ApiError(code, message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function fetchReadiness(signal: AbortSignal): Promise<Readiness> {
  const response = await fetch("/api/readiness", { credentials: "include", signal });
  if (!response.ok && response.status !== 503)
    throw new Error(response.statusText || "Readiness request failed");
  const result = await response.json();
  if (typeof result?.ready !== "boolean" ||
      typeof result.provider !== "string" ||
      typeof result.reason !== "string")
    throw new Error("Readiness response is invalid");
  return result as Readiness;
}

export async function bootstrapSession(): Promise<{
  authenticated: boolean;
}> {
  const token = new URLSearchParams(window.location.hash.slice(1)).get("token");
  if (token) {
    const response = await api<{ csrf_token: string }>(
      "/session",
      { method: "POST", body: JSON.stringify({ token }) },
    );
    csrfToken = response.csrf_token;
  }
  const session = await api<{
    authenticated: boolean;
    csrf_token: string;
  }>("/session");
  csrfToken = session.csrf_token || "";
  if (token && session.authenticated)
    history.replaceState(null, "", `${location.pathname}${location.search}`);
  return {
    authenticated: session.authenticated,
  };
}

export function setCsrfToken(value: string) {
  csrfToken = value;
}

export function localeDate(
  value: string | null | undefined,
): string {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.valueOf())
    ? "—"
    : new Intl.DateTimeFormat("en-GB", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      }).format(date);
}

export function shortId(id: string) {
  return id.slice(0, 8);
}
export function isActive(status: RunStatus) {
  return status === "queued" || status === "running";
}
