import {
  ExperimentBreakdown,
  ExperimentComparison,
  ExperimentSummary,
  type Breakdown,
} from "./ExperimentMetrics";
import {
  lazy,
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type Dispatch,
  type ReactNode,
  type SetStateAction,
} from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Select from "@radix-ui/react-select";
import ReactMarkdown from "react-markdown";
import {
  Activity,
  ArrowDownToLine,
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  CircleAlert,
  ExternalLink,
  FileText,
  FlaskConical,
  FolderOpen,
  Layers3,
  Link2,
  Menu,
  MessageCircle,
  PanelRightClose,
  Plus,
  RefreshCw,
  Send,
  Square,
  UploadCloud,
  X,
} from "lucide-react";
import {
  Link,
  Navigate,
  NavLink,
  Route,
  Routes,
  useLocation,
  useNavigate,
  useParams,
} from "react-router-dom";
import {
  api,
  ApiError,
  bootstrapSession,
  fetchReadiness,
  isActive,
  localeDate,
  shortId,
  type Collection,
  type Conversation,
  type Experiment,
  type ExperimentCase,
  type Health,
  type Message,
  type Model,
  type Readiness,
  type Run,
  type RunPage,
  type RunEvent,
  type RunStatus,
  type Source,
  type Span,
  type Stats,
  type Variant,
} from "./api";
import { t, type LabelKey } from "./i18n";
import { documentClassLabels } from "./taxonomy";

const PdfDocument = lazy(() => import("./PdfDocument"));

const terminal = new Set(["succeeded", "failed", "cancelled", "interrupted"]);
const eventNames = [
  "run.created",
  "stage.started",
  "stage.completed",
  "tool.started",
  "tool.completed",
  "answer.ready",
  "run.failed",
  "run.cancelled",
  "run.interrupted",
];
const roleStages: Record<string, string> = {
  retrieval: "Retrieving source text",
  generation: "Composing answer",
  answer: "Composing answer",
  validate: "Checking evidence",
  validation: "Checking evidence",
  search_evidence: "Searching evidence",
  read_evidence: "Reading source passages",
  collect_scope: "Collecting source scope",
};

type I18n = (key: LabelKey) => string;
type ConversationDetail = Conversation & {
  messages: Message[];
  runs: Run[];
  messages_next_cursor: string | null;
  runs_next_cursor: string | null;
};
type ExperimentDetail = Experiment & {
  cases: ExperimentCase[];
  breakdown?: Breakdown;
};
type PendingRunRequest = {
  conversationId: string;
  question: string;
  model: Model;
  variant: Variant;
  retryOf: string | null;
  requestId: string;
};
type UploadRequest = {
  requestId: string;
  confirmed: boolean;
  uncertain: boolean;
};
type UploadDraft = {
  files: File[];
  documentClass: string;
  submitted: boolean;
};
type UploadState = {
  drafts: Map<string, UploadDraft>;
  setDrafts: (update: (drafts: Map<string, UploadDraft>) => Map<string, UploadDraft>) => void;
  requests: Map<string, Map<File, UploadRequest>>;
  revision: number;
  refresh: () => void;
};

function safeTrace(url: string | null, base: string | null): string | null {
  if (!url) return null;
  try {
    const target = new URL(url);
    const allowed = base ? new URL(base).origin : null;
    if (target.protocol !== "http:" && target.protocol !== "https:")
      return null;
    if (allowed && target.origin !== allowed) return null;
    if (
      !allowed &&
      !["localhost", "127.0.0.1", "[::1]"].includes(target.hostname)
    )
      return null;
    return target.href;
  } catch {
    return null;
  }
}

function safeAnswerLink(value: string | undefined): string | null {
  if (!value) return null;
  try {
    const url = new URL(value);
    return ["https:", "http:", "mailto:"].includes(url.protocol) ? url.href : null;
  } catch {
    return null;
  }
}

function runErrorMessage(run: Run, tr: I18n): string {
  if (run.error?.code === "scope_expired") return tr("scopeChangedError");
  if (run.error?.code === "source_unavailable") return tr("sourceChangedError");
  if (run.error?.code === "process_restart") return tr("processRestartError");
  return run.error?.message || tr("noAnswer");
}

function traceStatusLabel(run: Run, tr: I18n): string {
  if (run.trace_status === "unrecoverable") return tr("traceUnrecoverable");
  if (run.trace_status === "incomplete") return tr("traceIncomplete");
  if (run.trace_status === "pending") return tr("tracePending");
  if (!run.trace_status) return tr("traceMissing");
  return tr("traceUnavailable");
}

function uncertainPost(error: unknown): boolean {
  return !(error instanceof ApiError) || error.status === 408 || error.status >= 500;
}

function SelectField({
  value,
  onChange,
  options,
  label,
  compact = false,
  disabled = false,
}: {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  label: string;
  compact?: boolean;
  disabled?: boolean;
}) {
  return (
    <label className={`field ${compact ? "field-compact" : ""}`}>
      <span className="field-label">{label}</span>
      <Select.Root value={value} onValueChange={onChange} disabled={disabled}>
        <Select.Trigger className="select-trigger" aria-label={label}>
          <Select.Value />
          <Select.Icon>
            <ChevronDown size={14} />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Content
            className="select-content"
            position="popper"
            sideOffset={4}
          >
            <Select.Viewport>
              {options.map((option) => (
                <Select.Item
                  className="select-item"
                  value={option.value}
                  key={option.value}
                >
                  <Select.ItemText>{option.label}</Select.ItemText>
                  <Select.ItemIndicator>
                    <Check size={14} />
                  </Select.ItemIndicator>
                </Select.Item>
              ))}
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
    </label>
  );
}

function Modal({
  open,
  onOpenChange,
  title,
  children,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  title: string;
  children: ReactNode;
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Overlay className="dialog-overlay" />
        <Dialog.Content className="dialog-content">
          <div className="dialog-head">
            <Dialog.Title>{title}</Dialog.Title>
            <Dialog.Close type="button" className="icon-button" aria-label="Close">
              <X size={18} />
            </Dialog.Close>
          </div>
          {children}
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}

function StatusTag({ status, tr }: { status: string; tr: I18n }) {
  const display =
    status in
    {
      pending: 1,
      processing: 1,
      ready: 1,
      failed: 1,
      deleted: 1,
      supported: 1,
      partial: 1,
      conflicting: 1,
      not_documented: 1,
      needs_clarification: 1,
      interrupted: 1,
      queued: 1,
      running: 1,
      succeeded: 1,
      cancelled: 1,
    }
      ? tr(status as LabelKey)
      : status.replaceAll("_", " ");
  return (
    <span className={`status-tag status-${status}`}>
      <span className="status-dot" />
      {display}
    </span>
  );
}

function markStage(event: RunEvent): string {
  const p = event.payload || {};
  const name = String(p.stage || p.tool || p.name || "");
  return (
    roleStages[name] ||
    name.replaceAll("_", " ") ||
    event.type.replaceAll(".", " ")
  );
}

function useRunEvents(
  run: Run | null,
  onUpdate: (run: Run) => void,
  onTerminal: () => void,
) {
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [connecting, setConnecting] = useState(false);
  const highest = useRef(0);
  const updateRef = useRef(onUpdate);
  const terminalRef = useRef(onTerminal);
  updateRef.current = onUpdate;
  terminalRef.current = onTerminal;
  const runId = run?.id;
  const active = !!run && isActive(run.status);
  useEffect(() => {
    highest.current = 0;
    setEvents([]);
    setConnecting(false);
  }, [runId]);
  useEffect(() => {
    if (!runId || !active) return;
    let closed = false;
    let inFlight = false;
    const stream = new EventSource(
      `/api/runs/${encodeURIComponent(runId)}/events`,
      { withCredentials: true },
    );
    const stop = () => {
      if (closed) return;
      closed = true;
      stream.close();
      window.clearInterval(interval);
      setConnecting(false);
    };
    const reconcile = async () => {
      if (closed || inFlight) return;
      inFlight = true;
      try {
        const current = await api<Run>(`/runs/${runId}`);
        if (closed) return;
        updateRef.current(current);
        if (terminal.has(current.status)) {
          stop();
          terminalRef.current();
        }
      } catch {
        if (!closed) setConnecting(true);
      } finally {
        inFlight = false;
      }
    };
    const handle = (event: MessageEvent<string>) => {
      try {
        const item = JSON.parse(event.data) as RunEvent;
        if (item.run_id !== runId || item.seq <= highest.current || closed)
          return;
        highest.current = item.seq;
        setEvents((previous) => [...previous, item].slice(-80));
        setConnecting(false);
        if (
          [
            "answer.ready",
            "run.failed",
            "run.cancelled",
            "run.interrupted",
          ].includes(item.type)
        ) {
          void reconcile();
        }
      } catch {
        setConnecting(true);
      }
    };
    eventNames.forEach((name) =>
      stream.addEventListener(name, handle as EventListener),
    );
    stream.onopen = () => setConnecting(false);
    stream.onerror = () => {
      if (!closed) {
        setConnecting(true);
        void reconcile();
      }
    };
    const interval = window.setInterval(() => {
      void reconcile();
    }, 7000);
    return stop;
  }, [runId, active]);
  return { events, connecting };
}

function usePendingTraceRefresh(
  pending: boolean,
  refresh: (signal: AbortSignal) => Promise<void>,
) {
  useEffect(() => {
    if (!pending) return;
    let stopped = false;
    let delay = 2000;
    let timer: number | undefined;
    let controller: AbortController | null = null;
    const schedule = () => {
      if (!stopped && document.visibilityState === "visible" && !controller && timer === undefined)
        timer = window.setTimeout(() => {
          timer = undefined;
          void tick();
        }, delay);
    };
    const tick = async () => {
      controller = new AbortController();
      try {
        await refresh(controller.signal);
      } catch {
      } finally {
        controller = null;
        delay = Math.min(delay * 2, 30000);
        schedule();
      }
    };
    const visibility = () => {
      window.clearTimeout(timer);
      timer = undefined;
      if (document.visibilityState === "visible") {
        delay = 2000;
        schedule();
      } else {
        controller?.abort();
      }
    };
    document.addEventListener("visibilitychange", visibility);
    schedule();
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", visibility);
    };
  }, [pending, refresh]);
}

function tracePending(run: Run) {
  return terminal.has(run.status) &&
    (run.trace_status === "pending" || run.trace_status === "incomplete");
}

let sessionPromise: ReturnType<typeof bootstrapSession> | null = null;

function App() {
  const [session, setSession] = useState<
    "loading" | "ready" | "unauthorized" | "error"
  >("loading");
  const [health, setHealth] = useState<Health | null>(null);
  const [readiness, setReadiness] = useState<Readiness | null>(null);
  const [readinessChecking, setReadinessChecking] = useState(false);
  const [readinessError, setReadinessError] = useState(false);
  const [collections, setCollections] = useState<Collection[]>([]);
  const [collectionsLoaded, setCollectionsLoaded] = useState(false);
  const [collectionId, setCollectionId] = useState<string>(
    () => localStorage.getItem("papertrail.collection") || "",
  );
  const [sources, setSources] = useState<Source[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [loadedCollectionId, setLoadedCollectionId] = useState("");
  const [loadedConversationCollectionId, setLoadedConversationCollectionId] = useState("");
  const [notice, setNotice] = useState<string | null>(null);
  const [mobilePane, setMobilePane] = useState<"library" | "answer" | "source">(
    "answer",
  );
  const [sourcePages, setSourcePages] = useState<{ collectionId: string; cursors: string[] }>({ collectionId: "", cursors: [""] });
  const [sourceNextCursor, setSourceNextCursor] = useState<string | null>(null);
  const [sourceCollection, setSourceCollection] = useState<Collection | null>(null);
  const [sourcesLoading, setSourcesLoading] = useState(false);
  const [pendingUploadCollectionId, setPendingUploadCollectionId] = useState<string | null>(null);
  const [conversationPages, setConversationPages] = useState<{ collectionId: string; cursors: string[] }>({ collectionId: "", cursors: [""] });
  const [conversationNextCursor, setConversationNextCursor] = useState<string | null>(null);
  const [conversationsLoading, setConversationsLoading] = useState(false);
  const [uploadDrafts, setUploadDrafts] = useState<Map<string, UploadDraft>>(() => new Map());
  const uploadRequests = useRef<Map<string, Map<File, UploadRequest>>>(new Map());
  const [uploadRequestRevision, setUploadRequestRevision] = useState(0);
  const pendingRunRequests = useRef<PendingRunRequest[]>([]);
  const uploadState: UploadState = {
    drafts: uploadDrafts,
    setDrafts: setUploadDrafts,
    requests: uploadRequests.current,
    revision: uploadRequestRevision,
    refresh: () => setUploadRequestRevision((value) => value + 1),
  };
  const conversationCursors = conversationPages.collectionId === collectionId ? conversationPages.cursors : [""];
  const conversationCursor = conversationCursors.at(-1) || "";
  const sourceCursors = sourcePages.collectionId === collectionId ? sourcePages.cursors : [""];
  const sourceCursor = sourceCursors.at(-1) || "";
  const showSourcePage = (cursors: string[]) => {
    setSources([]);
    setSourceNextCursor(null);
    setSourcesLoading(true);
    setSourcePages({ collectionId, cursors });
  };
  const sourcePagination = {
    page: sourceCursors.length,
    loading: sourcesLoading,
    hasNext: !!sourceNextCursor,
    previous: () => showSourcePage(sourceCursors.slice(0, -1)),
    next: () => {
      if (sourceNextCursor)
        showSourcePage([...sourceCursors, sourceNextCursor]);
    },
    reset: () => showSourcePage([""]),
  };
  const showConversationPage = (cursors: string[]) => {
    setConversations([]);
    setConversationNextCursor(null);
    setConversationsLoading(true);
    setConversationPages({ collectionId, cursors });
  };
  const conversationPagination = {
    page: conversationCursors.length,
    loading: conversationsLoading,
    hasNext: !!conversationNextCursor,
    previous: () => showConversationPage(conversationCursors.slice(0, -1)),
    next: () => {
      if (conversationNextCursor)
        showConversationPage([...conversationCursors, conversationNextCursor]);
    },
    reset: () => showConversationPage([""]),
  };
  const collectionController = useRef<AbortController | null>(null);
  const conversationListController = useRef<AbortController | null>(null);
  const baseController = useRef<AbortController | null>(null);
  const readinessController = useRef<AbortController | null>(null);
  const collectionLoading = useRef(false);
  const uploadGeneration = useRef(0);
  const selectedCollectionRef = useRef(collectionId);
  selectedCollectionRef.current = collectionId;
  const tr = t;
  const navigate = useNavigate();
  const appLocation = useLocation();
  const workspaceRoute = appLocation.pathname.startsWith("/workspace");
  useEffect(() => {
    if (!workspaceRoute) setMobilePane("answer");
  }, [workspaceRoute]);
  const selectedCollection =
    (sourceCollection?.id === collectionId ? sourceCollection : null) ||
    collections.find((item) => item.id === collectionId) || null;
  const visibleSources = loadedCollectionId === collectionId ? sources : [];
  const visibleConversations = loadedConversationCollectionId === collectionId ? conversations : [];

  const refreshReadiness = useCallback(async () => {
    if (readinessController.current) return;
    const controller = new AbortController();
    readinessController.current = controller;
    setReadinessChecking(true);
    try {
      const result = await fetchReadiness(controller.signal);
      if (controller.signal.aborted) return;
      setReadiness(result);
      setReadinessError(false);
    } catch {
      if (!controller.signal.aborted) {
        setReadiness(null);
        setReadinessError(true);
      }
    } finally {
      if (readinessController.current === controller) {
        readinessController.current = null;
        setReadinessChecking(false);
      }
    }
  }, []);

  useEffect(() => {
    if (session !== "ready" || !workspaceRoute) {
      readinessController.current?.abort();
      readinessController.current = null;
      setReadiness(null);
      setReadinessError(false);
      setReadinessChecking(false);
      return;
    }
    const check = () => {
      if (document.visibilityState === "visible") refreshReadiness().catch(() => undefined);
    };
    check();
    const timer = window.setInterval(check, 30000);
    window.addEventListener("focus", check);
    document.addEventListener("visibilitychange", check);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener("focus", check);
      document.removeEventListener("visibilitychange", check);
      readinessController.current?.abort();
      readinessController.current = null;
    };
  }, [session, workspaceRoute, refreshReadiness]);

  const loadBase = useCallback(async () => {
    baseController.current?.abort();
    const controller = new AbortController();
    baseController.current = controller;
    try {
      const [list, status] = await Promise.all([
        api<{ items: Collection[] }>("/collections", { signal: controller.signal }),
        api<Health>("/health", { signal: controller.signal }),
      ]);
      if (controller.signal.aborted) return;
      setCollections(list.items);
      setCollectionsLoaded(true);
      setHealth(status);
      setCollectionId((previous) => {
        const valid = list.items.some((item) => item.id === previous);
        const next = valid ? previous : list.items[0]?.id || "";
        if (next) localStorage.setItem("papertrail.collection", next);
        else localStorage.removeItem("papertrail.collection");
        selectedCollectionRef.current = next;
        return next;
      });
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    } finally {
      if (baseController.current === controller) baseController.current = null;
    }
  }, []);

  useEffect(() => {
    let active = true;
    const authenticate = (renew = false) => {
      if (renew || !sessionPromise) sessionPromise = bootstrapSession();
      sessionPromise
        .then((value) => {
          if (!active) return;
          if (!value.authenticated) {
            setSession("unauthorized");
            return;
          }
          setSession("ready");
          loadBase().catch((error) => {
            if (!active) return;
            setSession(
              error instanceof ApiError && error.status === 401
                ? "unauthorized"
                : "error",
            );
            setNotice(
              error instanceof Error ? error.message : "Request failed",
            );
          });
        })
        .catch((error) => {
          if (!active) return;
          setSession(
            error instanceof ApiError && error.status === 401
              ? "unauthorized"
              : "error",
          );
          setNotice(error instanceof Error ? error.message : "Request failed");
        });
    };
    const launchLink = () => {
      if (new URLSearchParams(window.location.hash.slice(1)).has("token")) {
        setSession("loading");
        authenticate(true);
      }
    };
    window.addEventListener("hashchange", launchLink);
    authenticate();
    return () => {
      active = false;
      window.removeEventListener("hashchange", launchLink);
    };
  }, [loadBase]);

  useEffect(() => {
    const expired = () => {
      sessionPromise = null;
      setSession("unauthorized");
      setNotice(tr("expired"));
    };
    window.addEventListener("evidence-lab:session-expired", expired);
    return () =>
      window.removeEventListener("evidence-lab:session-expired", expired);
  }, [tr]);

  const loadCollection = useCallback(async () => {
    collectionController.current?.abort();
    if (!collectionsLoaded ||
        (collectionId && !collections.some((item) => item.id === collectionId))) return;
    if (!collectionId || session !== "ready") {
      collectionLoading.current = false;
      setSources([]);
      setLoadedCollectionId("");
      setSourceCollection(null);
      setSourceNextCursor(null);
      setSourcesLoading(false);
      return;
    }
    const controller = new AbortController();
    collectionController.current = controller;
    const requestUploadGeneration = uploadGeneration.current;
    collectionLoading.current = true;
    setSourcesLoading(true);
    let resettingPage = false;
    try {
      const sourceList = await api<{ items: Source[]; next_cursor: string | null; collection: Collection }>(`/collections/${collectionId}/sources?limit=100&cursor=${encodeURIComponent(sourceCursor)}`, { signal: controller.signal });
      if (controller.signal.aborted || selectedCollectionRef.current !== collectionId) return;
      if (sourceCursor && sourceList.items.length === 0) {
        resettingPage = true;
        setSourceCollection(sourceList.collection);
        if (!sourceList.collection.has_pending_sources && requestUploadGeneration === uploadGeneration.current)
          setPendingUploadCollectionId((previous) => previous === collectionId ? null : previous);
        setSources([]);
        setSourceNextCursor(null);
        setLoadedCollectionId(collectionId);
        setSourcePages({ collectionId, cursors: [""] });
        return;
      }
      setSources(sourceList.items);
      setSourceNextCursor(sourceList.next_cursor);
      setSourceCollection(sourceList.collection);
      if (!sourceList.collection.has_pending_sources && requestUploadGeneration === uploadGeneration.current)
        setPendingUploadCollectionId((previous) => previous === collectionId ? null : previous);
      setLoadedCollectionId(collectionId);
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    } finally {
      if (collectionController.current === controller) {
        collectionController.current = null;
        collectionLoading.current = false;
        if (!resettingPage) setSourcesLoading(false);
      }
    }
  }, [collectionId, collections, collectionsLoaded, session, sourceCursor]);

  const loadConversations = useCallback(async () => {
    conversationListController.current?.abort();
    if (!collectionId || session !== "ready" || !workspaceRoute) {
      setConversations([]);
      setLoadedConversationCollectionId("");
      setConversationNextCursor(null);
      setConversationsLoading(false);
      return;
    }
    const controller = new AbortController();
    conversationListController.current = controller;
    setConversationsLoading(true);
    let resettingPage = false;
    try {
      const result = await api<{ items: Conversation[]; next_cursor: string | null }>(
        `/conversations?collection_id=${encodeURIComponent(collectionId)}&limit=100&cursor=${encodeURIComponent(conversationCursor)}`,
        { signal: controller.signal },
      );
      if (controller.signal.aborted || selectedCollectionRef.current !== collectionId) return;
      if (conversationCursor && result.items.length === 0) {
        resettingPage = true;
        setConversationPages({ collectionId, cursors: [""] });
        return;
      }
      setConversations(result.items);
      setConversationNextCursor(result.next_cursor);
      setLoadedConversationCollectionId(collectionId);
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    } finally {
      if (conversationListController.current === controller) {
        conversationListController.current = null;
        if (!resettingPage) setConversationsLoading(false);
      }
    }
  }, [collectionId, conversationCursor, session, workspaceRoute]);

  useEffect(() => {
    if (!workspaceRoute) {
      collectionController.current?.abort();
      return;
    }
    loadCollection().catch((error) =>
      setNotice(error instanceof Error ? error.message : tr("error")),
    );
    return () => collectionController.current?.abort();
  }, [loadCollection, tr, workspaceRoute]);
  useEffect(() => {
    loadConversations().catch((error) =>
      setNotice(error instanceof Error ? error.message : tr("error")),
    );
    return () => conversationListController.current?.abort();
  }, [loadConversations, tr]);
  useEffect(() => {
    if (session !== "ready") return;
    const refresh = () => {
      if (document.visibilityState !== "visible") return;
      loadBase().catch(() => undefined);
    };
    window.addEventListener("focus", refresh);
    document.addEventListener("visibilitychange", refresh);
    return () => {
      window.removeEventListener("focus", refresh);
      document.removeEventListener("visibilitychange", refresh);
    };
  }, [session, loadBase]);
  useEffect(() => {
    if (
      !workspaceRoute ||
      !(selectedCollection?.has_pending_sources || pendingUploadCollectionId === collectionId || visibleSources.some(
        (source) => source.status === "pending" || source.status === "processing",
      ))
    )
      return;
    const timer = window.setInterval(() => {
      if (!collectionLoading.current) loadCollection().catch(() => undefined);
    }, 3500);
    return () => window.clearInterval(timer);
  }, [workspaceRoute, selectedCollection?.has_pending_sources, pendingUploadCollectionId, collectionId, visibleSources, loadCollection]);

  const chooseCollection = (id: string) => {
    if (selectedCollectionRef.current !== id) {
      selectedCollectionRef.current = id;
      collectionController.current?.abort();
      setCollectionId(id);
      localStorage.setItem("papertrail.collection", id);
    }
    navigate("/workspace");
    setMobilePane("answer");
  };
  const markSourceUploadConfirmed = (id: string) => {
    uploadGeneration.current += 1;
    setPendingUploadCollectionId(id);
  };
  const ensureCollection = useCallback((id: string) => {
    if (selectedCollectionRef.current === id) return;
    selectedCollectionRef.current = id;
    collectionController.current?.abort();
    setCollectionId(id);
    localStorage.setItem("papertrail.collection", id);
  }, []);

  if (session === "loading")
    return (
      <div className="boot-screen">
        <div className="brand-mark">
          E<span>.</span>
        </div>
        <p>{tr("workInProgress")}</p>
      </div>
    );
  if (session !== "ready")
    return (
      <div className="auth-screen">
        <div className="brand-mark">
          E<span>.</span>
        </div>
        <span className="eyebrow">LOCAL RESEARCH WORKSPACE</span>
        <h1>{tr("authentication")}</h1>
        <p>{tr("authHint")}</p>
        <button
          className="primary-button"
          onClick={() => {
            sessionPromise = null;
            window.location.reload();
          }}
        >
          {tr("retryConnect")}
        </button>
        {notice && (
          <p className="error-text" role="alert">
            {notice}
          </p>
        )}
      </div>
    );

  return (
    <div className="app-shell">
      <header className="topbar">
        <Link
          className="brand"
          to="/workspace"
          aria-label="Evidence Lab workspace"
        >
          <span className="brand-mark">
            E<span>.</span>
          </span>
          <span className="brand-word">
            {tr("appName")}
            <small>{tr("appTagline")}</small>
          </span>
        </Link>
        <nav className="main-nav" aria-label="Primary">
          <NavLink
            to="/workspace"
            className={({ isActive }) => (isActive ? "nav-active" : "")}
          >
            <BookOpen size={16} />
            {tr("workspace")}
          </NavLink>
          <NavLink
            to="/runs"
            className={({ isActive }) => (isActive ? "nav-active" : "")}
          >
            <Activity size={16} />
            {tr("runs")}
          </NavLink>
          <NavLink
            to="/experiments"
            className={({ isActive }) => (isActive ? "nav-active" : "")}
          >
            <FlaskConical size={16} />
            {tr("experiments")}
          </NavLink>
        </nav>
        <div className="top-actions">
          <span className="local-badge">
            <span className="pulse-dot" />
            {tr("local")}
          </span>
        </div>
      </header>
      {notice && (
        <div className="notice" role="alert">
          <CircleAlert size={16} />
          <span>{notice}</span>
          <button onClick={() => setNotice(null)} aria-label={tr("close")}>
            <X size={16} />
          </button>
        </div>
      )}
      <Routes>
        <Route path="/" element={<Navigate to="/workspace" replace />} />
        <Route
          path="/workspace"
          element={
            <Workspace
              collection={selectedCollection}
              collections={collections}
              sources={visibleSources}
              sourcePagination={sourcePagination}
              conversations={visibleConversations}
              conversationPagination={conversationPagination}
              chooseCollection={chooseCollection}
              ensureCollection={ensureCollection}
              refreshCollection={loadCollection}
              refreshConversations={loadConversations}
              onSourceUploadConfirmed={markSourceUploadConfirmed}
              uploadState={uploadState}
              pendingRunRequests={pendingRunRequests}
              refreshBase={loadBase}
              tr={tr}
              health={health}
              readiness={readiness}
              readinessChecking={readinessChecking}
              readinessError={readinessError}
              refreshReadiness={refreshReadiness}
              setNotice={setNotice}
              mobilePane={mobilePane}
              setMobilePane={setMobilePane}
            />
          }
        />
        <Route
          path="/workspace/:conversationId"
          element={
            <Workspace
              collection={selectedCollection}
              collections={collections}
              sources={visibleSources}
              sourcePagination={sourcePagination}
              conversations={visibleConversations}
              conversationPagination={conversationPagination}
              chooseCollection={chooseCollection}
              ensureCollection={ensureCollection}
              refreshCollection={loadCollection}
              refreshConversations={loadConversations}
              onSourceUploadConfirmed={markSourceUploadConfirmed}
              uploadState={uploadState}
              pendingRunRequests={pendingRunRequests}
              refreshBase={loadBase}
              tr={tr}
              health={health}
              readiness={readiness}
              readinessChecking={readinessChecking}
              readinessError={readinessError}
              refreshReadiness={refreshReadiness}
              setNotice={setNotice}
              mobilePane={mobilePane}
              setMobilePane={setMobilePane}
            />
          }
        />
        <Route
          path="/runs"
          element={
            <RunsPage
              tr={tr}
              health={health}
              setNotice={setNotice}
            />
          }
        />
        <Route path="/runs/:runId" element={<RunLink tr={tr} />} />
        <Route
          path="/experiments"
          element={
            <ExperimentsPage
              tr={tr}
              setNotice={setNotice}
            />
          }
        />
        <Route
          path="/experiments/:experimentId"
          element={
            <ExperimentPage tr={tr} setNotice={setNotice} />
          }
        />
        <Route
          path="*"
          element={
            <div className="not-found">
              <h1>404</h1>
              <Link to="/workspace">
                {tr("workspace")} <ArrowRight size={16} />
              </Link>
            </div>
          }
        />
      </Routes>
    </div>
  );
}

function RunLink({ tr }: { tr: I18n }) {
  const { runId } = useParams();
  const [run, setRun] = useState<Run | null>(null);
  const [error, setError] = useState<{ runId: string; message: string } | null>(null);
  useEffect(() => {
    let active = true;
    if (runId)
      api<Run>(`/runs/${runId}`)
        .then((value) => {
          if (active && value.id === runId) setRun(value);
        })
        .catch((reason) => {
          if (active)
            setError({ runId, message: reason instanceof Error ? reason.message : "Request failed" });
        });
    return () => {
      active = false;
    };
  }, [runId]);
  if (runId && run?.id === runId)
    return (
      <Navigate
        to={`/workspace/${run.conversation_id}?run=${run.id}`}
        replace
      />
    );
  return (
    <main className="report-page">
      <p>{error && error.runId === runId ? error.message : tr("dataPending")}</p>
      <Link to="/runs">{tr("runs")}</Link>
    </main>
  );
}

interface WorkspaceProps {
  collection: Collection | null;
  collections: Collection[];
  sources: Source[];
  sourcePagination: {
    page: number;
    loading: boolean;
    hasNext: boolean;
    previous: () => void;
    next: () => void;
    reset: () => void;
  };
  conversations: Conversation[];
  conversationPagination: {
    page: number;
    loading: boolean;
    hasNext: boolean;
    previous: () => void;
    next: () => void;
    reset: () => void;
  };
  chooseCollection: (id: string) => void;
  ensureCollection: (id: string) => void;
  refreshCollection: () => Promise<void>;
  refreshConversations: () => Promise<void>;
  onSourceUploadConfirmed: (collectionId: string) => void;
  uploadState: UploadState;
  pendingRunRequests: { current: PendingRunRequest[] };
  refreshBase: () => Promise<void>;
  tr: I18n;
  health: Health | null;
  readiness: Readiness | null;
  readinessChecking: boolean;
  readinessError: boolean;
  refreshReadiness: () => Promise<void>;
  setNotice: (value: string | null) => void;
  mobilePane: "library" | "answer" | "source";
  setMobilePane: Dispatch<SetStateAction<"library" | "answer" | "source">>;
}

function Workspace(props: WorkspaceProps) {
  const {
    collection,
    collections,
    sources,
    sourcePagination,
    conversations,
    conversationPagination,
    chooseCollection,
    ensureCollection,
    refreshCollection,
    refreshConversations,
    onSourceUploadConfirmed,
    uploadState,
    pendingRunRequests,
    refreshBase,
    tr,
    health,
    readiness,
    readinessChecking,
    readinessError,
    refreshReadiness,
    setNotice,
    mobilePane,
    setMobilePane,
  } = props;
  const { conversationId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const params = new URLSearchParams(location.search);
  const sourceId = params.get("source");
  const spanId = params.get("span");
  const requestedRunId = params.get("run");
  const [storedConversationDetail, setConversationDetail] = useState<ConversationDetail | null>(null);
  const [currentRun, setCurrentRun] = useState<Run | null>(null);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [selectedSpan, setSelectedSpan] = useState<Span | null>(null);
  const [detachedSource, setDetachedSource] = useState<Source | null>(null);
  const [newCollectionOpen, setNewCollectionOpen] = useState(false);
  const [newConversationOpen, setNewConversationOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [question, setQuestion] = useState("");
  const [model, setModel] = useState<Model>("gpt-6-sol");
  const [variant, setVariant] = useState<Variant>("V3");
  const [advanced, setAdvanced] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);
  const conversationController = useRef<AbortController | null>(null);
  const uploadCollectionId = collection?.id || "";
  const uploadDraft = uploadState.drafts.get(uploadCollectionId) || { files: [], documentClass: "other", submitted: false };
  const setUploadDraft = (update: (draft: UploadDraft) => UploadDraft) => {
    if (!uploadCollectionId) return;
    uploadState.setDrafts((previous) => {
      const next = new Map(previous);
      next.set(uploadCollectionId, update(previous.get(uploadCollectionId) || { files: [], documentClass: "other", submitted: false }));
      return next;
    });
  };
  const clearUploadDraft = () => {
    uploadState.setDrafts((previous) => {
      const next = new Map(previous);
      next.delete(uploadCollectionId);
      return next;
    });
    uploadState.requests.delete(uploadCollectionId);
    uploadState.refresh();
  };
  const uncertainUploadFiles = useMemo(() => new Set(
    [...(uploadState.requests.get(uploadCollectionId) || [])]
      .filter(([, request]) => request.uncertain)
      .map(([file]) => file),
  ), [uploadCollectionId, uploadState.requests, uploadState.revision]);
  const routeKey = `${conversationId || ""}:${requestedRunId || ""}`;
  const routeKeyRef = useRef(routeKey);
  routeKeyRef.current = routeKey;
  const conversationIdRef = useRef(conversationId);
  conversationIdRef.current = conversationId;
  const detail = storedConversationDetail?.id === conversationId ? storedConversationDetail : null;
  const activeRun = detail?.runs.find((item) => isActive(item.status)) ||
    (currentRun && currentRun.conversation_id === conversationId && isActive(currentRun.status)
      ? currentRun
      : null);
  const providerUnavailable = readiness?.ready === false;

  const loadConversation = useCallback(async (signal?: AbortSignal) => {
    if (routeKeyRef.current !== routeKey ||
        window.location.pathname !== (conversationId ? `/workspace/${conversationId}` : "/workspace")) return;
    conversationController.current?.abort();
    if (!conversationId) {
      setConversationDetail(null);
      setCurrentRun(null);
      return;
    }
    const controller = new AbortController();
    conversationController.current = controller;
    const abort = () => controller.abort();
    signal?.addEventListener("abort", abort, { once: true });
    try {
      const result = await api<ConversationDetail>(
        `/conversations/${conversationId}?messages_limit=50&runs_limit=50`,
        { signal: controller.signal },
      );
      if (controller.signal.aborted || routeKeyRef.current !== routeKey || result.id !== conversationId) return;
      const knownRequests = new Set(result.runs.map((item) => item.request_id));
      pendingRunRequests.current = pendingRunRequests.current.filter(
        (request) => request.conversationId !== conversationId || !knownRequests.has(request.requestId),
      );
      if (result.collection_id !== collection?.id)
        ensureCollection(result.collection_id);
      setConversationDetail((previous) => {
        if (previous?.id !== result.id) return result;
        const newRuns = new Set(result.runs.map((item) => item.id));
        const newMessages = new Set(result.messages.map((item) => item.id));
        return {
          ...result,
          runs: [...result.runs, ...previous.runs.filter((item) => !newRuns.has(item.id))],
          messages: [...result.messages, ...previous.messages.filter((item) => !newMessages.has(item.id))],
          runs_next_cursor: previous.runs.length > result.runs.length ? previous.runs_next_cursor : result.runs_next_cursor,
          messages_next_cursor: previous.messages.length > result.messages.length ? previous.messages_next_cursor : result.messages_next_cursor,
        };
      });
      const selected = result.runs.find((item) => item.id === requestedRunId) ||
        (requestedRunId ? await api<Run>(`/runs/${requestedRunId}`, { signal: controller.signal }) : null);
      if (controller.signal.aborted || routeKeyRef.current !== routeKey) return;
      setCurrentRun(selected?.conversation_id === conversationId ? selected : result.runs[0] || null);
    } catch (error) {
      if (!controller.signal.aborted) throw error;
    } finally {
      signal?.removeEventListener("abort", abort);
    }
  }, [conversationId, collection?.id, ensureCollection, pendingRunRequests, requestedRunId, routeKey]);

  const loadOlderHistory = async (kind: "runs" | "messages") => {
    if (!conversationId || !detail || historyLoading) return;
    const cursor = kind === "runs" ? detail.runs_next_cursor : detail.messages_next_cursor;
    if (!cursor) return;
    setHistoryLoading(true);
    try {
      const query = kind === "runs"
        ? `runs_limit=50&runs_cursor=${encodeURIComponent(cursor)}&messages_limit=1`
        : `messages_limit=50&messages_cursor=${encodeURIComponent(cursor)}&runs_limit=1`;
      const result = await api<ConversationDetail>(`/conversations/${conversationId}?${query}`);
      if (routeKeyRef.current !== routeKey || result.id !== conversationId) return;
      setConversationDetail((previous) => {
        if (previous?.id !== conversationId) return previous;
        const existing = new Set((kind === "runs" ? previous.runs : previous.messages).map((item) => item.id));
        return kind === "runs"
          ? { ...previous, runs: [...previous.runs, ...result.runs.filter((item) => !existing.has(item.id))], runs_next_cursor: result.runs_next_cursor }
          : { ...previous, messages: [...previous.messages, ...result.messages.filter((item) => !existing.has(item.id))], messages_next_cursor: result.messages_next_cursor };
      });
    } catch (error) {
      setNotice(error instanceof Error ? error.message : tr("error"));
    } finally {
      setHistoryLoading(false);
    }
  };

  useEffect(() => {
    loadConversation().catch((error) =>
      setNotice(error instanceof Error ? error.message : tr("error")),
    );
    return () => conversationController.current?.abort();
  }, [loadConversation, tr]);
  const pendingTrace = !!detail?.runs.slice(0, 50).some(tracePending) ||
    (!!currentRun && tracePending(currentRun));
  const refreshPendingTraces = useCallback(
    (signal: AbortSignal) => loadConversation(signal),
    [loadConversation],
  );
  usePendingTraceRefresh(pendingTrace, refreshPendingTraces);
  useEffect(() => {
    let active = true;
    setSelectedSpan(null);
    if (!spanId) {
      return;
    }
    const saved = detail?.runs
      .flatMap((run) => run.answer?.citations || [])
      .find((span) => span.id === spanId && span.source_id === sourceId);
    if (saved) {
      setSelectedSpan(saved);
      return;
    }
    api<Span>(`/spans/${spanId}`)
      .then((span) => {
        if (active && span.source_id === sourceId) setSelectedSpan(span);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [spanId, sourceId, detail?.runs]);
  useEffect(() => {
    let active = true;
    setDetachedSource(null);
    if (!sourceId || sources.some((item) => item.id === sourceId)) {
      return;
    }
    api<Source>(`/sources/${sourceId}`)
      .then((source) => {
        if (active) setDetachedSource(source);
      })
      .catch(() => undefined);
    return () => {
      active = false;
    };
  }, [sourceId, sources]);
  useEffect(() => {
    setMobilePane((previous) => sourceId ? "source" : previous === "source" ? "answer" : previous);
  }, [sourceId]);
  useEffect(() => {
    const selected = requestedRunId && currentRun?.id === requestedRunId
      ? document.getElementById(`run-${requestedRunId}`)
      : null;
    if (selected) selected.scrollIntoView({ block: "center", behavior: "smooth" });
    else bottom.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [conversationId, currentRun?.id, requestedRunId]);

  const openSource = (id: string, span?: string) => {
    const next = new URLSearchParams(location.search);
    next.set("source", id);
    if (span) next.set("span", span);
    else next.delete("span");
    navigate(`${location.pathname}?${next.toString()}`);
    setMobilePane("source");
  };
  const closeSource = () => {
    const next = new URLSearchParams(location.search);
    next.delete("source");
    next.delete("span");
    navigate(`${location.pathname}${next.size ? `?${next.toString()}` : ""}`);
    setMobilePane("answer");
  };

  const createCollection = async (title: string, description: string) => {
    setBusy(true);
    try {
      const created = await api<Collection>("/collections", {
        method: "POST",
        body: JSON.stringify({ title, description }),
      });
      chooseCollection(created.id);
      setNewCollectionOpen(false);
      refreshBase().catch((error) =>
        setNotice(error instanceof Error ? error.message : tr("error")),
      );
    } catch (error) {
      if (uncertainPost(error)) {
        setNewCollectionOpen(false);
        setNotice("Collection creation outcome is unknown. Check the refreshed library before creating it again.");
        refreshBase().catch(() => undefined);
      } else setNotice(error instanceof Error ? error.message : tr("createError"));
    } finally {
      setBusy(false);
    }
  };
  const createConversation = async (title: string) => {
    if (!collection) return;
    setBusy(true);
    try {
      const created = await api<Conversation>("/conversations", {
        method: "POST",
        body: JSON.stringify({ collection_id: collection.id, title }),
      });
      navigate(`/workspace/${created.id}`);
      setNewConversationOpen(false);
      setMobilePane("answer");
      if (conversationPagination.page > 1) conversationPagination.reset();
      else refreshConversations().catch((error) =>
        setNotice(error instanceof Error ? error.message : tr("error")),
      );
    } catch (error) {
      if (uncertainPost(error)) {
        setNewConversationOpen(false);
        setNotice("Conversation creation outcome is unknown. Check the refreshed list before creating it again.");
        if (conversationPagination.page > 1) conversationPagination.reset();
        else refreshConversations().catch(() => undefined);
      } else setNotice(error instanceof Error ? error.message : tr("createError"));
    } finally {
      setBusy(false);
    }
  };
  const upload = async (
    files: File[],
    documentClass: string,
  ) => {
    if (!collection) return;
    setUploadError(null);
    setBusy(true);
    let requests = uploadState.requests.get(collection.id);
    if (!requests) {
      requests = new Map();
      uploadState.requests.set(collection.id, requests);
    }
    try {
      const failures: string[] = [];
      for (const file of files) {
        const previous = requests.get(file);
        if (previous?.confirmed) continue;
        const request = previous || { requestId: crypto.randomUUID(), confirmed: false, uncertain: false };
        requests.set(file, request);
        const form = new FormData();
        form.append("file", file);
        form.append("document_class", documentClass);
        form.append("request_id", request.requestId);
        try {
          await api<Source>(`/collections/${collection.id}/sources`, {
            method: "POST",
            body: form,
          });
          request.confirmed = true;
          request.uncertain = false;
          onSourceUploadConfirmed(collection.id);
        } catch (error) {
          if (uncertainPost(error)) request.uncertain = true;
          else requests.delete(file);
          failures.push(`${file.name}: ${uncertainPost(error)
            ? tr("uploadOutcomeUncertain")
            : error instanceof Error ? error.message : tr("uploadError")}`);
        }
      }
      uploadState.refresh();
      if (sourcePagination.page > 1) sourcePagination.reset();
      else refreshCollection().catch((error) =>
        setNotice(error instanceof Error ? error.message : tr("error")),
      );
      if (failures.length) setUploadError(failures.join("; "));
      else {
        clearUploadDraft();
        setUploadError(null);
        setUploadOpen(false);
      }
    } finally {
      setBusy(false);
    }
  };
  const submit = async (
    text: string,
    retryOf: string | null = null,
    retryModel?: Model,
    retryVariant?: Variant,
  ) => {
    if (
      !conversationId ||
      !text.trim() ||
      activeRun || busy || providerUnavailable
    )
      return;
    setNotice(null);
    setBusy(true);
    const nextQuestion = text.trim();
    const nextModel = retryModel || model;
    const nextVariant = retryVariant || variant;
    const previous = pendingRunRequests.current.find((request) =>
      request.conversationId === conversationId && request.question === nextQuestion &&
      request.model === nextModel && request.variant === nextVariant && request.retryOf === retryOf,
    );
    const requestId = previous
      ? previous.requestId
      : crypto.randomUUID();
    if (!previous)
      pendingRunRequests.current.push({
        conversationId, question: nextQuestion, model: nextModel,
        variant: nextVariant, retryOf, requestId,
      });
    const forgetRequest = () => {
      pendingRunRequests.current = pendingRunRequests.current.filter((request) => request.requestId !== requestId);
    };
    try {
      const created = await api<Run>(`/conversations/${conversationId}/runs`, {
        method: "POST",
        body: JSON.stringify({
          request_id: requestId,
          question: nextQuestion,
          model: nextModel,
          variant: nextVariant,
          retry_of_run_id: retryOf,
        }),
      });
      forgetRequest();
      if (conversationIdRef.current !== conversationId ||
          window.location.pathname !== `/workspace/${conversationId}`) return;
      setCurrentRun(created);
      setQuestion("");
      navigate(`/workspace/${conversationId}?run=${created.id}`);
      setMobilePane("answer");
    } catch (error) {
      if (!uncertainPost(error)) forgetRequest();
      if (conversationIdRef.current === conversationId &&
          window.location.pathname === `/workspace/${conversationId}`)
        setNotice(uncertainPost(error)
          ? tr("runOutcomeUncertain")
          : error instanceof Error ? error.message : tr("error"));
    } finally {
      setBusy(false);
    }
  };
  const cancel = async (runId: string) => {
    const cancelRoute = routeKeyRef.current;
    const cancelPath = window.location.pathname;
    try {
      const stopped = await api<Run>(`/runs/${runId}/cancel`, {
        method: "POST",
      });
      if (routeKeyRef.current !== cancelRoute || window.location.pathname !== cancelPath) return;
      if (currentRun?.id === runId) setCurrentRun(stopped);
      setConversationDetail((previous) => previous && previous.id === conversationId
        ? { ...previous, runs: previous.runs.map((item) => item.id === runId ? stopped : item) }
        : previous);
      await loadConversation();
    } catch (error) {
      if (routeKeyRef.current === cancelRoute && window.location.pathname === cancelPath)
        setNotice(error instanceof Error ? error.message : tr("error"));
    }
  };
  const removeSource = async (id: string) => {
    if (!window.confirm(tr("removeConfirm"))) return;
    try {
      await api<Source>(`/sources/${id}`, { method: "DELETE" });
      await refreshCollection();
      if (sourceId === id) closeSource();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : tr("removeFailed"));
    }
  };
  const { events, connecting } = useRunEvents(activeRun, (updated) => {
    if (currentRun?.id === updated.id) setCurrentRun(updated);
    setConversationDetail((previous) => previous && previous.id === conversationId
      ? { ...previous, runs: previous.runs.map((item) => item.id === updated.id ? updated : item) }
      : previous);
  }, () => {
    loadConversation().catch(() => undefined);
  });
  const sourceCandidate =
    sources.find((item) => item.id === sourceId) ||
    (detachedSource?.id === sourceId ? detachedSource : null);
  const expectedCollectionId = conversationId ? detail?.collection_id : collection?.id;
  const sourceMismatch = !!sourceCandidate && !!expectedCollectionId &&
    sourceCandidate.collection_id !== expectedCollectionId;
  const source = sourceCandidate && !sourceMismatch && expectedCollectionId
    ? sourceCandidate : null;
  const readyCount = collection?.ready_count || 0;
  const sourceCount = collection?.source_count || 0;

  return (
    <main
      className={`workspace ${mobilePane === "source" ? "workspace-source-open" : ""}`}
      aria-label={tr("workspace")}
    >
      <aside className={`library-panel pane-${mobilePane}`}>
        <div className="panel-head">
          <div>
            <span className="eyebrow">01 / {tr("library")}</span>
            <h2>{tr("library")}</h2>
          </div>
          <button
            className="icon-button"
            aria-label={tr("refreshLibrary")}
            title={tr("refreshLibrary")}
            onClick={() => {
              sourcePagination.reset();
              refreshBase().catch((error) => setNotice(error instanceof Error ? error.message : tr("error")));
            }}
          >
            <RefreshCw size={18} />
          </button>
          <button
            className="icon-button"
            aria-label={tr("newCollection")}
            onClick={() => setNewCollectionOpen(true)}
          >
            <Plus size={18} />
          </button>
        </div>
        <div className="library-body">
          <SelectField
            value={collection?.id || "none"}
            onChange={(value) =>
              value === "new"
                ? setNewCollectionOpen(true)
                : chooseCollection(value)
            }
            options={[
              ...collections.map((item) => ({
                value: item.id,
                label: item.title,
              })),
              ...(collections.length
                ? []
                : [{ value: "none", label: tr("selectCollection") }]),
              { value: "new", label: `+ ${tr("newCollection")}` },
            ]}
            label={tr("collection")}
          />
          {collection ? (
            <>
              <div className="library-section-heading">
                <span>{tr("sources")}</span>
                <span className="section-count">
                  {sourceCount.toString().padStart(2, "0")}
                </span>
              </div>
              <button
                className="add-source"
                onClick={() => setUploadOpen(true)}
              >
                <UploadCloud size={17} />
                <span>{tr("upload")}</span>
                <Plus size={15} />
              </button>
              <div className="source-list">
                {sources.length === 0 ? (
                  <p className="muted-message">
                    {sourcePagination.loading
                      ? "Loading documents…"
                      : sourceCount === 0
                        ? tr("noSources")
                        : "Documents are unavailable. Refresh the library."}
                  </p>
                ) : (
                  sources.map((item) => (
                    <div
                      className={`source-row ${sourceId === item.id ? "source-selected" : ""}`}
                      key={item.id}
                    >
                      <button
                        className="source-main"
                        onClick={() => openSource(item.id)}
                      >
                        <span className="file-icon">
                          <FileText size={17} />
                        </span>
                        <span className="source-details">
                          <strong title={item.title || item.filename}>
                            {item.title || item.filename}
                          </strong>
                          <small>
                            {item.media_type.includes("pdf") ? "PDF" : "TXT"} ·{" "}
                            {item.page_count
                              ? `${item.page_count} ${tr("page").toLowerCase()}`
                              : item.document_class}
                          </small>
                        </span>
                      </button>
                      <StatusTag status={item.status} tr={tr} />
                      <button
                        className="source-remove"
                        onClick={() => removeSource(item.id)}
                        aria-label={`${tr("remove")}: ${item.filename}`}
                        title={tr("remove")}
                      >
                        <X size={13} />
                      </button>
                      {item.status === "failed" && item.error && (
                        <p className="source-error">{item.error}</p>
                      )}
                    </div>
                  ))
                )}
              </div>
              {(sourceCount > 100 || sourcePagination.page > 1) && (
                <div className="source-pagination" aria-label="Source pages">
                  <button className="text-button" disabled={sourcePagination.loading || sourcePagination.page === 1} onClick={sourcePagination.previous}>Previous</button>
                  <span className="mono">{sourcePagination.page}</span>
                  <button className="text-button" disabled={sourcePagination.loading || !sourcePagination.hasNext} onClick={sourcePagination.next}>Next</button>
                </div>
              )}
              <div className="library-section-heading conversation-heading">
                <span>{tr("conversations")}</span>
                <button
                  className="subtle-icon"
                  onClick={() => setNewConversationOpen(true)}
                  aria-label={tr("newConversation")}
                >
                  <Plus size={17} />
                </button>
              </div>
              {conversations.length === 0 ? (
                <p className="muted-message">{conversationPagination.loading ? tr("dataPending") : tr("noConversations")}</p>
              ) : (
                <div className="conversation-list">
                  {conversations.map((item) => (
                    <Link
                      className={`conversation-row ${conversationId === item.id ? "conversation-selected" : ""}`}
                      to={`/workspace/${item.id}`}
                      key={item.id}
                      onClick={() => setMobilePane("answer")}
                    >
                      <MessageCircle size={16} />
                      <span>
                        {item.title ||
                          `${tr("conversations")} ${shortId(item.id)}`}
                      </span>
                      <ChevronRight size={14} />
                    </Link>
                  ))}
                </div>
              )}
              {(conversationPagination.hasNext || conversationPagination.page > 1) && (
                <div className="source-pagination" aria-label="Conversation pages">
                  <button className="text-button" disabled={conversationPagination.loading || conversationPagination.page === 1} onClick={conversationPagination.previous}>Previous</button>
                  <span className="mono">{conversationPagination.page}</span>
                  <button className="text-button" disabled={conversationPagination.loading || !conversationPagination.hasNext} onClick={conversationPagination.next}>Next</button>
                </div>
              )}
            </>
          ) : (
            <div className="empty-library">
              <FolderOpen size={30} />
              <p>{tr("noCollection")}</p>
              <button
                className="secondary-button"
                onClick={() => setNewCollectionOpen(true)}
              >
                <Plus size={16} />
                {tr("newCollection")}
              </button>
            </div>
          )}
        </div>
        <div className="library-foot">
          <span className="mono">
            {String(readyCount).padStart(2, "0")} /{" "}
            {String(sourceCount).padStart(2, "0")}
          </span>
          <span>{tr("sourceReadyHint")}</span>
        </div>
      </aside>

      <section
        className={`chat-panel pane-${mobilePane}`}
        aria-label={tr("answer")}
      >
        <div className="chat-head">
          <div>
            <span className="eyebrow">02 / {tr("workspace")}</span>
            <h1>{detail?.title || collection?.title || tr("workspace")}</h1>
            <p>{collection?.description || tr("ask")}</p>
          </div>
          <div className="chat-head-actions">
            <span className="small-count">
              <Layers3 size={14} />
              {readyCount} {tr("sourceCount")}
            </span>
            <button
              className="icon-button mobile-menu"
              onClick={() => setMobilePane("library")}
              aria-label={tr("library")}
            >
              <Menu size={19} />
            </button>
          </div>
        </div>
        {(providerUnavailable || readinessError) && (
          <div
            role={providerUnavailable ? "alert" : "status"}
            style={{
              flexShrink: 0,
              padding: "10px 32px",
              background: providerUnavailable ? "#f9eee9" : "#f4f1ea",
              borderBottom: "1px solid #d8a898",
              color: providerUnavailable ? "#7f3522" : "#74675b",
              display: "flex",
              alignItems: "center",
              gap: 10,
              fontSize: 12,
            }}
          >
            <CircleAlert size={16} />
            <span>{tr(providerUnavailable ? "answerServiceUnavailable" : "availabilityUnknown")}</span>
            <button
              type="button"
              className="text-button"
              onClick={() => refreshReadiness().catch(() => undefined)}
              disabled={readinessChecking}
              style={{ marginLeft: "auto", whiteSpace: "nowrap" }}
            >
              <RefreshCw size={13} />
              {tr(readinessChecking ? "checkingAvailability" : "retryAvailability")}
            </button>
          </div>
        )}
        <div className="chat-scroll">
          {!collection ? (
            <EmptyWorkspace tr={tr} action={() => setNewCollectionOpen(true)} />
          ) : !detail ? (
            <div className="conversation-empty">
              <span className="eyebrow">RESEARCH / {collection.title}</span>
              <div className="empty-art">
                <div className="empty-curve" />
                <BookOpen size={42} strokeWidth={1} />
              </div>
              <h2>{tr("emptyWorkspaceTitle")}</h2>
              <p>{tr("emptyWorkspaceBody")}</p>
              <div className="empty-actions">
                <button
                  className="primary-button"
                  onClick={() =>
                    readyCount
                      ? setNewConversationOpen(true)
                      : setUploadOpen(true)
                  }
                >
                  {readyCount ? tr("newConversation") : tr("upload")}
                  <ArrowRight size={16} />
                </button>
              </div>
              <div className="empty-facts">
                <div>
                  <span>01</span>
                  {tr("uploadHint")}
                </div>
                <div>
                  <span>02</span>
                  {tr("sourceReadyHint")}
                </div>
                <div>
                  <span>03</span>
                  {tr("citations")}
                </div>
              </div>
            </div>
          ) : (
            <div className="turn-list">
              {(detail.runs.length ? detail.runs_next_cursor : detail.messages_next_cursor) && (
                <button className="text-button" disabled={historyLoading} onClick={() => loadOlderHistory(detail.runs.length ? "runs" : "messages")}>
                  {historyLoading ? tr("dataPending") : "Load older history"}
                </button>
              )}
              {detail.runs.length
                ? [...detail.runs, ...(currentRun && !detail.runs.some((item) => item.id === currentRun.id) ? [currentRun] : [])]
                    .sort((a, b) => a.created_at.localeCompare(b.created_at) || a.id.localeCompare(b.id))
                    .map((run) => (
                    <RunTurn
                      key={run.id}
                      run={run.id === currentRun?.id ? currentRun : run}
                      events={run.id === activeRun?.id ? events : []}
                      connecting={run.id === activeRun?.id && connecting}
                      tr={tr}

                      health={health}
                      answerUnavailable={providerUnavailable || !!activeRun || busy}
                      openSpan={(span) => openSource(span.source_id, span.id)}
                      submit={submit}
                      cancel={cancel}
                      setNotice={setNotice}
                    />
                  ))
                : [...detail.messages].reverse().map((message) => (
                    <div
                      className={`message message-${message.role}`}
                      key={message.id}
                    >
                      {message.content}
                    </div>
                  ))}
              <div ref={bottom} />
            </div>
          )}
        </div>
        {detail && (
          <div className="composer-wrap">
            <form
              className="composer"
              onSubmit={(event) => {
                event.preventDefault();
                submit(question);
              }}
            >
              <label htmlFor="question-input" className="sr-only">
                {tr("question")}
              </label>
              <textarea
                id="question-input"
                placeholder={
                  readyCount ? tr("askPlaceholder") : tr("noneReady")
                }
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                onKeyDown={(event) => {
                  if (
                    event.key === "Enter" &&
                    !event.shiftKey &&
                    !event.nativeEvent.isComposing
                  ) {
                    event.preventDefault();
                    submit(question);
                  }
                }}
                disabled={
                  !readyCount || !!activeRun || providerUnavailable
                }
                rows={2}
              />
              <div className="composer-foot">
                <div className="composer-controls">
                  <SelectField
                    compact
                    value={model}
                    onChange={(value) => setModel(value as Model)}
                    label={tr("model")}
                    options={[
                      { value: "gpt-6-sol", label: "GPT-6 Sol" },
                      { value: "gpt-6-luna", label: "GPT-6 Luna" },
                    ]}
                  />
                  <button
                    type="button"
                    className={`advanced-button ${advanced ? "advanced-on" : ""}`}
                    onClick={() => setAdvanced(!advanced)}
                    aria-expanded={advanced}
                  >
                    {tr("advanced")}
                    <ChevronDown size={14} />
                  </button>
                </div>
                <button
                  className="send-button"
                  type="submit"
                  disabled={
                    !question.trim() ||
                    !readyCount ||
                    busy ||
                    !!activeRun ||
                    providerUnavailable
                  }
                  aria-label={tr("send")}
                >
                  <Send size={17} />
                </button>
              </div>
              {advanced && (
                <div className="advanced-row">
                  <SelectField
                    value={variant}
                    onChange={(value) => setVariant(value as Variant)}
                    label={tr("variant")}
                    options={(["V0", "V1", "V2", "V3"] as Variant[]).map(
                      (value) => ({ value, label: value }),
                    )}
                  />
                  <p>{tr("modelRule")}</p>
                </div>
              )}
            </form>
            <p className="composer-hint">{tr("keyboardHint")}</p>
          </div>
        )}
      </section>

      <aside
        className={`evidence-panel pane-${mobilePane}`}
        aria-label={tr("evidence")}
      >
        <div className="panel-head evidence-head">
          <div>
            <span className="eyebrow">03 / {tr("evidence")}</span>
            <h2>{source ? source.title || source.filename : tr("evidence")}</h2>
          </div>
          <button
            className="icon-button"
            onClick={closeSource}
            aria-label={tr("close")}
          >
            <PanelRightClose size={18} />
          </button>
        </div>
        {sourceMismatch ? (
          <div className="evidence-empty" role="alert">
            <CircleAlert size={27} />
            <p>This source belongs to another collection.</p>
          </div>
        ) : sourceId && expectedCollectionId ? (
          <SourceViewer
            key={sourceId}
            source={source}
            sourceId={sourceId}
            span={selectedSpan}
            requestedSpanId={spanId}
            tr={tr}

          />
        ) : (
          <div className="evidence-empty">
            <div className="evidence-glyph">
              <Link2 size={27} />
            </div>
            <h3>{tr("evidence")}</h3>
            <p>{tr("sourceExcerpt")}</p>
            <div className="evidence-index">
              <span>01</span>
              <span>{tr("currentLibrary")}</span>
              <span>{readyCount}</span>
            </div>
            <div className="evidence-index">
              <span>02</span>
              <span>{tr("sourceStatus")}</span>
              <span>{collection?.unavailable_count || 0}</span>
            </div>
          </div>
        )}
      </aside>

      <div className="mobile-tabs" role="tablist" aria-label="Workspace panels">
        <button
          role="tab"
          aria-selected={mobilePane === "library"}
          onClick={() => setMobilePane("library")}
        >
          <BookOpen size={18} />
          {tr("mobileLibrary")}
        </button>
        <button
          role="tab"
          aria-selected={mobilePane === "answer"}
          onClick={() => setMobilePane("answer")}
        >
          <MessageCircle size={18} />
          {tr("mobileAnswer")}
        </button>
        <button
          role="tab"
          aria-selected={mobilePane === "source"}
          onClick={() => setMobilePane("source")}
        >
          <FileText size={18} />
          {tr("mobileSource")}
        </button>
      </div>

      <Modal
        open={newCollectionOpen}
        onOpenChange={setNewCollectionOpen}
        title={tr("newCollection")}
      >
        <form
          className="modal-form"
          onSubmit={(event) => {
            event.preventDefault();
            const data = new FormData(event.currentTarget);
            createCollection(
              String(data.get("title")).trim(),
              String(data.get("description")).trim(),
            );
          }}
        >
          <label>
            {tr("collectionName")}
            <input name="title" required maxLength={100} autoFocus />
          </label>
          <label>
            {tr("description")}
            <textarea name="description" rows={3} maxLength={500} />
          </label>
          <button className="primary-button" disabled={busy}>
            {tr("create")}
            <ArrowRight size={16} />
          </button>
        </form>
      </Modal>
      <Modal
        open={newConversationOpen}
        onOpenChange={setNewConversationOpen}
        title={tr("newConversation")}
      >
        <form
          className="modal-form"
          onSubmit={(event) => {
            event.preventDefault();
            const data = new FormData(event.currentTarget);
            createConversation(String(data.get("title")).trim());
          }}
        >
          <label>
            {tr("conversationName")}
            <input name="title" required maxLength={100} autoFocus />
          </label>
          <button className="primary-button" disabled={busy}>
            {tr("create")}
            <ArrowRight size={16} />
          </button>
        </form>
      </Modal>
      <Modal
        open={uploadOpen}
        onOpenChange={(open) => {
          setUploadOpen(open);
          if (!open) {
            setUploadError(null);
            if (!busy && !uncertainUploadFiles.size) clearUploadDraft();
          }
        }}
        title={tr("upload")}
      >
        <UploadForm tr={tr} busy={busy} error={uploadError} uncertainFiles={uncertainUploadFiles} draft={uploadDraft} setDraft={setUploadDraft} onSubmit={upload} />
      </Modal>
    </main>
  );
}

function UploadForm({
  tr,
  busy,
  error,
  uncertainFiles,
  draft,
  setDraft,
  onSubmit,
}: {
  tr: I18n;
  busy: boolean;
  error: string | null;
  uncertainFiles: Set<File>;
  draft: UploadDraft;
  setDraft: (update: (draft: UploadDraft) => UploadDraft) => void;
  onSubmit: (
    files: File[],
    documentClass: string,
  ) => void;
}) {
  const { files, documentClass, submitted } = draft;
  return (
    <form
      className="modal-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (files.length && !busy) {
          setDraft((current) => ({ ...current, submitted: true }));
          onSubmit(files, documentClass);
        }
      }}
    >
      <label className="upload-zone">
        <UploadCloud size={29} />
        <strong>
          {files.length
            ? files.map((file) => file.name).join(", ")
            : tr("chooseFile")}
        </strong>
        <span>{tr("uploadHint")}</span>
        <input
          type="file"
          accept=".pdf,.txt,application/pdf,text/plain"
          multiple
          onChange={(event) => {
            const selected = Array.from(event.target.files || []);
            setDraft((current) => ({ ...current, files: selected }));
            event.target.value = "";
          }}
          disabled={submitted || busy}
        />
      </label>
      {files.map((file, index) => (
        <div className="upload-file" key={`${file.name}-${file.lastModified}-${index}`}>
          <span>{file.name}</span>
          <button type="button" className="text-button" disabled={busy || uncertainFiles.has(file)} title={uncertainFiles.has(file) ? "Retry this file to resolve its uncertain outcome" : undefined} onClick={() => setDraft((current) => ({ ...current, files: current.files.filter((_, position) => position !== index) }))}>
            {tr("remove")}
          </button>
        </div>
      ))}
      <div className="modal-columns">
        <SelectField
          label={tr("documentClass")}
          value={documentClass}
          onChange={(value) => setDraft((current) => ({ ...current, documentClass: value }))}
          options={Object.entries(documentClassLabels).map(([value, label]) => ({ value, label }))}
          disabled={submitted || busy}
        />
      </div>
      <p className="muted">{tr("uploadBatchMetadataHint")}</p>
      {submitted && <p className="muted">The file chooser and class stay fixed for this batch. You can remove rejected files.</p>}
      {uncertainFiles.size > 0 && <p className="muted">Files with an unknown upload result stay selected when you close this window. Reopen it to retry safely.</p>}
      {error && <p className="error-text" role="alert">{error}</p>}
      <button className="primary-button" disabled={!files.length || busy}>
        <UploadCloud size={16} />
        {tr("upload")}
      </button>
    </form>
  );
}

function EmptyWorkspace({ tr, action }: { tr: I18n; action: () => void }) {
  return (
    <div className="conversation-empty">
      <span className="eyebrow">EVIDENCE LAB / 01</span>
      <div className="empty-art">
        <BookOpen size={42} strokeWidth={1} />
      </div>
      <h2>{tr("emptyWorkspaceTitle")}</h2>
      <p>{tr("emptyWorkspaceBody")}</p>
      <button className="primary-button" onClick={action}>
        {tr("newCollection")}
        <ArrowRight size={16} />
      </button>
    </div>
  );
}

function RunTurn({
  run,
  events,
  connecting,
  tr,
  health,
  answerUnavailable,
  openSpan,
  submit,
  cancel,
  setNotice,
}: {
  run: Run;
  events: RunEvent[];
  connecting: boolean;
  tr: I18n;
  health: Health | null;
  answerUnavailable: boolean;
  openSpan: (span: Span) => void;
  submit: (
    question: string,
    retryOf: string,
    model: Model,
    variant: Variant,
  ) => void;
  cancel: (runId: string) => void;
  setNotice: (value: string | null) => void;
}) {
  const [spans, setSpans] = useState<Record<string, Span>>({});
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!isActive(run.status)) return;
    const interval = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [run.status]);
  const evidenceIds = useMemo(
    () =>
      Array.from(
        new Set(
          run.answer?.claims.flatMap((claim) => claim.evidence_ids) || [],
        ),
      ),
    [run.answer],
  );
  useEffect(() => {
    let active = true;
    const saved = Object.fromEntries(
      (run.answer?.citations || []).map((span) => [span.id, span]),
    );
    setSpans(saved);
    Promise.all(
      evidenceIds
        .filter((id) => !saved[id])
        .map((id) =>
          api<Span>(`/spans/${id}`)
            .then((span) => [id, span] as const)
            .catch(() => null),
        ),
    ).then((values) => {
      if (active)
        setSpans({
          ...saved,
          ...Object.fromEntries(
            values.filter((value): value is readonly [string, Span] => !!value),
          ),
        });
    });
    return () => {
      active = false;
    };
  }, [evidenceIds.join("|"), run.answer]);
  const exportMarkdown = async () => {
    try {
      const response = await fetch(`/api/runs/${run.id}/export`, {
        credentials: "include",
      });
      if (!response.ok) throw new Error(response.statusText);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `evidence-lab-${shortId(run.id)}.md`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : tr("error"));
    }
  };
  const trace = safeTrace(run.trace_url, health?.langfuse_url || null);
  const currentStage = [...events]
    .reverse()
    .find(
      (event) =>
        event.type.startsWith("stage.") || event.type.startsWith("tool."),
    );
  const completeStages = events.filter(
    (event) =>
      event.type === "stage.completed" || event.type === "tool.completed",
  );
  const duration = run.finished_at
    ? Math.round(
        (new Date(run.finished_at).valueOf() -
          new Date(run.created_at).valueOf()) /
          1000,
      )
    : isActive(run.status)
      ? Math.max(
          0,
          Math.floor((now - new Date(run.created_at).valueOf()) / 1000),
        )
      : null;
  return (
    <article className="run-turn" id={`run-${run.id}`}>
      <div className="question-block">
        <div className="turn-meta">
          <span>{tr("question")}</span>
          <span className="mono">
            {localeDate(run.created_at)} · {shortId(run.id)}
          </span>
        </div>
        <p>{run.question}</p>
      </div>
      <div className="answer-block">
        <div className="answer-heading">
          <span className="answer-symbol">✳</span>
          <div>
            <span className="eyebrow">
              {tr("answer")} / {run.model === "gpt-6-sol" ? "SOL" : "LUNA"} /{" "}
              {run.variant}
            </span>
            <h3>
              {isActive(run.status)
                ? tr("workInProgress")
                : run.answer
                  ? tr("answer")
                  : tr("result")}
            </h3>
          </div>
          <div className="run-statuses">
            <StatusTag status={run.status} tr={tr} />
            {run.answer && <StatusTag status={run.answer.status} tr={tr} />}
          </div>
        </div>
        {run.scope && (
          <div className="run-scope-summary">
            <span>{tr("scopeLocked")}: {run.scope.source_count} {tr("sourceCount")}</span>
            <span>{tr("unavailable")}: {run.scope.unavailable_count}</span>
            <span>{localeDate(run.created_at)}</span>
          </div>
        )}
        {run.answer ? (
          <>
            <div className="answer-copy">
              <ReactMarkdown
                skipHtml
                components={{
                  img: () => null,
                  a: ({ href, children }) => {
                    const safe = safeAnswerLink(href);
                    return safe
                      ? <a href={safe} target="_blank" rel="noopener noreferrer">{children}</a>
                      : <span>{children}</span>;
                  },
                }}
              >{run.answer.answer}</ReactMarkdown>
            </div>
            {run.answer.claims.length > 0 && (
              <div className="claim-list">
                <div className="section-label">
                  {tr("citations")}{" "}
                  <span>
                    {run.answer.claims.length.toString().padStart(2, "0")}
                  </span>
                </div>
                {run.answer.claims.map((claim, index) => (
                  <div className="claim-row" key={`${run.id}-${index}`}>
                    <span className="claim-number">
                      {String(index + 1).padStart(2, "0")}
                    </span>
                    <div>
                      <p>{claim.text}</p>
                      <div className="citation-links">
                        {claim.evidence_ids.map((id) =>
                          spans[id] ? (
                            <button
                              key={id}
                              className="citation-link"
                              onClick={() => openSpan(spans[id])}
                            >
                              <Link2 size={13} />
                              <span>
                                {spans[id].source_title || tr("sourceExcerpt")}
                              </span>
                              <span className="mono">
                                {spans[id].page
                                  ? `p. ${spans[id].page}`
                                  : spans[id].line_start
                                    ? `L${spans[id].line_start}`
                                    : shortId(id)}
                              </span>
                            </button>
                          ) : (
                            <span className="citation-unavailable" key={id}>
                              {tr("evidenceMissing")} · {shortId(id)}
                            </span>
                          ),
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
            {run.answer.limitations?.length > 0 && (
              <div className="limitations">
                <CircleAlert size={15} />
                <div>
                  <strong>{tr("limitations")}</strong>
                  {run.answer.limitations.map((value, index) => (
                    <p key={index}>{value}</p>
                  ))}
                </div>
              </div>
            )}
          </>
        ) : isActive(run.status) ? (
          <div className="activity-card">
            <div className="activity-top">
              <div className="activity-loader" aria-hidden="true" />
              <div>
                <strong>
                  {connecting
                    ? tr("reconnecting")
                    : currentStage?.type.endsWith("started")
                      ? markStage(currentStage)
                      : tr("working")}
                </strong>
                <p>
                  {tr("stage")} · {shortId(run.id)}
                </p>
              </div>
            </div>
            {completeStages.length > 0 && (
              <details className="activity-log">
                <summary>
                  {tr("eventLog")} <span>{events.length}</span>
                </summary>
                <ol>
                  {events.map((event) => (
                    <li key={event.seq}>
                      <span className="mono">
                        {String(event.seq).padStart(2, "0")}
                      </span>
                      {markStage(event)}
                      <span className="mono">
                        {new Date(event.at).toLocaleTimeString("en-GB")}
                      </span>
                    </li>
                  ))}
                </ol>
              </details>
            )}
            <button className="text-button" onClick={() => cancel(run.id)}>
              <Square size={13} />
              {tr("stop")}
            </button>
          </div>
        ) : (
          <div className="run-error">
            <CircleAlert size={17} />
            <div>
              <strong>
                {run.status === "cancelled"
                  ? tr("runCancelled")
                  : run.status === "interrupted"
                    ? tr("interrupted")
                    : tr("runFailed")}
              </strong>
              <p>{runErrorMessage(run, tr)}</p>
            </div>
          </div>
        )}
        <div className="run-actions">
          <span className="mono">
            {tr("runId")} {shortId(run.id)}
            {duration !== null ? ` · ${duration}s` : ""}
          </span>
          <div>
            {trace && (
              <a
                href={trace}
                target="_blank"
                rel="noopener noreferrer"
                className="text-button"
              >
                {tr("trace")}
                <ExternalLink size={13} />
              </a>
            )}
            {trace && ["pending", "incomplete", "unrecoverable"].includes(run.trace_status || "") && (
              <span className="muted">{traceStatusLabel(run, tr)}</span>
            )}
            {!trace && terminal.has(run.status) && (
              <span className="muted">{traceStatusLabel(run, tr)}</span>
            )}
            {run.answer && (
              <button className="text-button" onClick={exportMarkdown}>
                <ArrowDownToLine size={14} />
                {tr("export")}
              </button>
            )}
            {!isActive(run.status) && (
              <button
                className="text-button"
                disabled={answerUnavailable}
                onClick={() =>
                  submit(run.question, run.id, run.model, run.variant)
                }
              >
                <RefreshCw size={13} />
                {tr("retry")}
              </button>
            )}
          </div>
        </div>
      </div>
    </article>
  );
}

function SourceViewer({
  source,
  sourceId,
  span,
  requestedSpanId,
  tr,
}: {
  source: Source | null;
  sourceId: string;
  span: Span | null;
  requestedSpanId: string | null;
  tr: I18n;
}) {
  const [pageCount, setPageCount] = useState(0);
  const [page, setPage] = useState(1);
  const [width, setWidth] = useState(400);
  const [blobUrl, setBlobUrl] = useState<string | null>(null);
  const [textSpans, setTextSpans] = useState<Span[]>([]);
  const [textPosition, setTextPosition] = useState<{ spanId: string | null; anchor: string | null; cursors: string[] }>({ spanId: null, anchor: null, cursors: [""] });
  const [textNextCursor, setTextNextCursor] = useState<string | null>(null);
  const [textLoading, setTextLoading] = useState(false);
  const textSpanId = span?.id || requestedSpanId;
  const textCursors = textPosition.spanId === textSpanId ? textPosition.cursors : [""];
  const textAnchor = textPosition.spanId === textSpanId ? textPosition.anchor : textSpanId;
  const textCursor = textCursors.at(-1) || "";
  const [error, setError] = useState<string | null>(null);
  const viewer = useRef<HTMLDivElement>(null);
  const pdf =
    source?.media_type.includes("pdf") ||
    source?.filename.toLowerCase().endsWith(".pdf");
  const fileAvailable = !!source && source.status !== "deleted" && !!pdf;
  useEffect(() => {
    const node = viewer.current;
    if (!node) return;
    const observer = new ResizeObserver((entries) =>
      setWidth(Math.max(220, Math.floor(entries[0].contentRect.width - 28))),
    );
    observer.observe(node);
    return () => observer.disconnect();
  }, [sourceId]);
  useEffect(() => {
    setError(null);
    setTextSpans([]);
    setBlobUrl(null);
    if (!fileAvailable) return;
    const controller = new AbortController();
    fetch(`/api/sources/${sourceId}/file`, {
      credentials: "include",
      signal: controller.signal,
    })
      .then((response) => {
        if (!response.ok) throw new Error(response.statusText);
        return response.blob();
      })
      .then((blob) => {
        if (!controller.signal.aborted) setBlobUrl(URL.createObjectURL(blob));
      })
      .catch((cause) => {
        if (!controller.signal.aborted)
          setError(
            cause instanceof Error ? cause.message : tr("sourceUnavailable"),
          );
      });
    return () => controller.abort();
  }, [sourceId, fileAvailable]);
  useEffect(() => {
    if (!source || source.status !== "ready" || pdf) return;
    const controller = new AbortController();
    setTextLoading(true);
    setTextSpans([]);
    setError(null);
    const position = textCursor ? `cursor=${encodeURIComponent(textCursor)}` : textAnchor ? `anchor=${encodeURIComponent(textAnchor)}` : "";
    api<{ items: Span[]; next_cursor: string | null }>(`/sources/${sourceId}/spans?limit=100&${position}`, { signal: controller.signal })
      .then((result) => {
        if (controller.signal.aborted) return;
        setTextSpans(result.items);
        setTextNextCursor(result.next_cursor);
      })
      .catch((cause) => {
        if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : tr("sourceUnavailable"));
      })
      .finally(() => {
        if (!controller.signal.aborted) setTextLoading(false);
      });
    return () => controller.abort();
  }, [sourceId, source?.status, textCursor, textAnchor]);
  useEffect(() => {
    if (span?.page) setPage(span.page);
  }, [span?.id]);
  useEffect(() => {
    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl);
    };
  }, [blobUrl]);
  const bboxValid =
    !!span?.bbox &&
    !!span.page_width &&
    !!span.page_height &&
    span.bbox[0] >= 0 &&
    span.bbox[2] <= span.page_width &&
    span.bbox[1] >= 0 &&
    span.bbox[3] <= span.page_height &&
    span.bbox[2] > span.bbox[0] &&
    span.bbox[3] > span.bbox[1];
  const lineMatches = span
    ? textSpans.some((item) => item.id === span.id)
    : false;
  return (
    <div className="source-viewer">
      <div className="source-toolbar">
        <div>
          <span className="eyebrow">{tr("sourceFile")}</span>
          <strong title={source?.filename}>
            {source?.filename || shortId(sourceId)}
          </strong>
        </div>
        {source && <StatusTag status={source.status} tr={tr} />}
      </div>
      {source && (
        <div className="source-provenance">
          <span>{source.document_class}</span>
          <span>{localeDate(source.created_at)}</span>
          <span className="mono">{shortId(source.sha256)}</span>
          {source.status !== "deleted" && (
            <a
              className="text-button"
              href={`/api/sources/${sourceId}/file`}
              download={source.filename}
              title="Download original file"
            >
              Download
            </a>
          )}
        </div>
      )}
      {span && (
        <div className="selected-excerpt">
          <div className="section-label">
            {tr(
              source?.status === "deleted"
                ? "historicalExcerpt"
                : "sourceExcerpt",
            )}{" "}
            <span>
              {span.page
                ? `${tr("page")} ${span.page}`
                : span.line_start
                  ? `${tr("lines")} ${span.line_start}–${span.line_end || span.line_start}`
                  : shortId(span.id)}
            </span>
          </div>
          <blockquote>{span.text}</blockquote>
        </div>
      )}
      {source?.status === "deleted" || !source ? (
        <div className="source-failure">
          <CircleAlert size={22} />
          <p>{tr("sourceUnavailable")}</p>
        </div>
      ) : error ? (
        <div className="source-failure">
          <CircleAlert size={22} />
          <p>{error}</p>
        </div>
      ) : (
        <div className="document-window" ref={viewer}>
          {pdf ? (
            blobUrl ? (
              <>
                <div className="pdf-controls">
                  <button
                    disabled={page <= 1}
                    onClick={() => setPage((value) => value - 1)}
                    aria-label="Previous page"
                  >
                    <ChevronLeft size={17} />
                  </button>
                  <span>
                    {tr("page")} <strong>{page}</strong> / {pageCount || "—"}
                  </span>
                  <button
                    disabled={page >= pageCount}
                    onClick={() => setPage((value) => value + 1)}
                    aria-label="Next page"
                  >
                    <ChevronRight size={17} />
                  </button>
                </div>
                <Suspense
                  fallback={
                    <div className="document-loading">{tr("opening")}…</div>
                  }
                >
                  <PdfDocument
                    blobUrl={blobUrl}
                    page={page}
                    width={width}
                    span={span}
                    bboxValid={bboxValid}
                    sourceExcerpt={tr("sourceExcerpt")}
                    opening={tr("opening")}
                    onPageCount={(count) => {
                      setPageCount(count);
                      setPage((current) => Math.min(Math.max(1, current), count));
                    }}
                    onError={setError}
                  />
                </Suspense>
                {span && !bboxValid && (
                  <p className="highlight-note">{tr("highlightUnavailable")}</p>
                )}
              </>
            ) : (
              <div className="document-loading">{tr("opening")}…</div>
            )
          ) : (
            <div className="txt-document">
              {textAnchor && (
                <button className="text-button" disabled={textLoading} onClick={() => setTextPosition({ spanId: textSpanId, anchor: null, cursors: [""] })}>Start of document</button>
              )}
              {(textNextCursor || textCursors.length > 1) && (
                <div className="source-pagination" aria-label="Text pages">
                  <button className="text-button" disabled={textLoading || textCursors.length === 1} onClick={() => setTextPosition({ spanId: textSpanId, anchor: textAnchor, cursors: textCursors.slice(0, -1) })}>Previous</button>
                  <span className="mono">{textCursors.length}</span>
                  <button className="text-button" disabled={textLoading || !textNextCursor} onClick={() => textNextCursor && setTextPosition({ spanId: textSpanId, anchor: textAnchor, cursors: [...textCursors, textNextCursor] })}>Next</button>
                </div>
              )}
              {source.status !== "ready" ? (
                <p className="muted-message">
                  {source.status === "failed"
                    ? "Source text could not be indexed."
                    : "Text preview is available when processing finishes."}
                </p>
              ) : textLoading ? (
                <div className="document-loading">{tr("opening")}…</div>
              ) : textSpans.length ? (
                textSpans.map((item) => (
                  <div
                    className={`txt-span ${item.id === span?.id && lineMatches ? "txt-highlight" : ""}`}
                    key={item.id}
                  >
                    <span className="line-number">
                      {item.line_start || "·"}
                    </span>
                    <pre>{item.text}</pre>
                  </div>
                ))
              ) : (
                <p className="muted-message">No extracted text is available.</p>
              )}
              {span && !lineMatches && (
                <p className="highlight-note">{tr("highlightUnavailable")}</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RunsPage({
  tr,
  health,
  setNotice,
}: {
  tr: I18n;
  health: Health | null;
  setNotice: (value: string | null) => void;
}) {
  const [page, setPage] = useState<{
    key: string;
    items: Run[];
    nextCursor: string | null;
  } | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [filter, setFilter] = useState<RunStatus | "all">("all");
  const [pageCursors, setPageCursors] = useState<(string | null)[]>([null]);
  const [pageIndex, setPageIndex] = useState(0);
  const [reload, setReload] = useState(0);
  const [loading, setLoading] = useState(false);
  const refreshGeneration = useRef(0);
  const cursor = pageCursors[pageIndex] || null;
  const pageKey = `${filter}:${cursor || ""}`;
  const currentPage = page?.key === pageKey ? page : null;
  const visible = currentPage?.items || [];
  const refreshRuns = useCallback(async (signal: AbortSignal) => {
    const generation = ++refreshGeneration.current;
    const params = new URLSearchParams({ limit: "100", status: filter });
    if (cursor) params.set("cursor", cursor);
    const [list, summary] = await Promise.all([
      api<RunPage>(`/runs?${params.toString()}`, { signal }),
      api<Stats>("/stats", { signal }),
    ]);
    if (signal.aborted || generation !== refreshGeneration.current) return;
    setPage({ key: pageKey, items: list.items, nextCursor: list.next_cursor });
    setStats(summary);
  }, [cursor, filter, pageKey]);
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    refreshRuns(controller.signal)
      .catch((error) =>
        !controller.signal.aborted &&
        setNotice(error instanceof Error ? error.message : tr("error")),
      )
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [refreshRuns, reload, setNotice, tr]);
  usePendingTraceRefresh(
    visible.some((run) => isActive(run.status) || tracePending(run)),
    refreshRuns,
  );
  return (
    <main className="report-page">
      <div className="report-hero">
        <span className="eyebrow">OBSERVATION / 02</span>
        <h1>{tr("allRuns")}</h1>
        <p>{tr("timeline")}</p>
      </div>
      <div className="stat-strip">
        <StatCell label={tr("allHistoryTotal")} value={stats?.total_runs} />
        <StatCell label={tr("complete")} value={stats?.runtime_status_counts.succeeded} />
        <StatCell label={tr("failed")} value={stats?.runtime_status_counts.failed} />
        <StatCell
          label={tr("latency")}
          value={stats?.latency_p50_ms == null
            ? null
            : `${Math.round(stats.latency_p50_ms)} ms`}
          detail={!stats
            ? undefined
            : stats.latency_n === 0
              ? tr("noTimedRuns")
              : `${stats.latency_n} ${tr(stats.latency_n === 1 ? "timedRun" : "timedRuns")}`}
        />
        <StatCell label={tr("tracePending")} value={stats?.terminal_trace_status_counts.pending} />
        <StatCell label={tr("traceIncomplete")} value={stats?.terminal_trace_status_counts.incomplete} />
        <StatCell label={tr("traceUnrecoverable")} value={stats?.terminal_trace_status_counts.unrecoverable} />
      </div>
      <div className="report-list-head">
        <h2>{tr("allRuns")}</h2>
        <button className="secondary-button" onClick={() => {
          setPageCursors([null]);
          setPageIndex(0);
          setReload((value) => value + 1);
        }} disabled={loading}>
          <RefreshCw size={14} /> {tr("refresh")}
        </button>
        <SelectField
          compact
          label={tr("filter")}
          value={filter}
          onChange={(value) => {
            setFilter(value as RunStatus | "all");
            setPageCursors([null]);
            setPageIndex(0);
          }}
          options={[
            "all",
            "queued",
            "running",
            "succeeded",
            "failed",
            "cancelled",
            "interrupted",
          ].map((value) => ({
            value,
            label: value === "all" ? tr("allStatuses") : value,
          }))}
        />
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>{tr("question")}</th>
              <th>{tr("answerModel")}</th>
              <th>{tr("variant")}</th>
              <th>{tr("executionStatus")}</th>
              <th>{tr("answerStatus")}</th>
              <th>{tr("duration")}</th>
              <th>{tr("trace")}</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((run) => (
              <tr key={run.id}>
                <td>
                  <Link
                    to={`/workspace/${run.conversation_id}?run=${run.id}`}
                    className="table-primary"
                  >
                    {run.question}
                    <span className="mono">
                      {shortId(run.id)} · {localeDate(run.created_at)}
                    </span>
                  </Link>
                </td>
                <td>{run.model === "gpt-6-sol" ? "Sol" : "Luna"}</td>
                <td className="mono">{run.variant}</td>
                <td>
                  <StatusTag status={run.status} tr={tr} />
                </td>
                <td>{run.answer ? <StatusTag status={run.answer.status} tr={tr} /> : "—"}</td>
                <td className="mono">
                  {run.finished_at
                    ? `${Math.round((new Date(run.finished_at).valueOf() - new Date(run.created_at).valueOf()) / 1000)}s`
                    : "—"}
                </td>
                <td>
                  {safeTrace(run.trace_url, health?.langfuse_url || null) ? (
                    <>
                      <a
                        target="_blank"
                        rel="noopener noreferrer"
                        href={safeTrace(run.trace_url, health?.langfuse_url || null)!}
                        aria-label={tr("trace")}
                      >
                        <ExternalLink size={16} />
                      </a>
                      {["pending", "incomplete", "unrecoverable"].includes(run.trace_status || "") && (
                        <span className="muted">{traceStatusLabel(run, tr)}</span>
                      )}
                    </>
                  ) : (
                    <span className="muted">{terminal.has(run.status) ? traceStatusLabel(run, tr) : "—"}</span>
                  )}
                </td>
              </tr>
            ))}
            {visible.length === 0 && (
              <tr>
                <td colSpan={7} className="table-empty">
                  {loading ? tr("loadingRuns") : currentPage ? tr("noRuns") : tr("runsUnavailable")}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="runs-pagination" aria-label={tr("runsPagination")}>
        <span className="mono" role="status">
          {tr("page")} {pageIndex + 1} · {visible.length} {tr("rows")}
        </span>
        <button
          className="secondary-button"
          disabled={loading || pageIndex === 0}
          onClick={() => setPageIndex((value) => value - 1)}
        >
          <ChevronLeft size={14} /> {tr("previousPage")}
        </button>
        <button
          className="secondary-button"
          disabled={loading || !currentPage?.nextCursor}
          onClick={() => {
            if (!currentPage?.nextCursor) return;
            setPageCursors((previous) => [
              ...previous.slice(0, pageIndex + 1),
              currentPage.nextCursor,
            ]);
            setPageIndex((value) => value + 1);
          }}
        >
          {tr("nextPage")} <ChevronRight size={14} />
        </button>
      </div>
    </main>
  );
}

function StatCell({
  label,
  value,
  detail,
}: {
  label: string;
  value: string | number | null | undefined;
  detail?: string;
}) {
  return (
    <div className="stat-cell">
      <span>{label}</span>
      <strong>{value ?? "—"}</strong>
      {detail && <small className="stat-detail">{detail}</small>}
    </div>
  );
}

function ExperimentsPage({
  tr,
  setNotice,
}: {
  tr: I18n;
  setNotice: (value: string | null) => void;
}) {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [campaignId, setCampaignId] = useState("");
  const [reload, setReload] = useState(0);
  const [cursors, setCursors] = useState<string[]>([""]);
  const [nextCursor, setNextCursor] = useState<string | null>(null);
  const [loadedCursor, setLoadedCursor] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const cursor = cursors.at(-1) || "";
  const visibleExperiments = loadedCursor === cursor ? experiments : [];
  const campaigns = [
    ...new Map(visibleExperiments.map((item) => [item.campaign_id, item])).values(),
  ];
  const selectedCampaign = campaigns.some((item) => item.campaign_id === campaignId) ? campaignId : campaigns[0]?.campaign_id;
  const selected = visibleExperiments.filter(
    (item) => item.campaign_id === selectedCampaign,
  );
  useEffect(() => {
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const result = await api<{ items: Experiment[]; next_cursor: string | null }>(`/experiments?limit=25&cursor=${encodeURIComponent(cursor)}`);
        if (!active) return;
        setExperiments(result.items);
        setNextCursor(result.next_cursor);
        setLoadedCursor(cursor);
        setLoadError(null);
        if (result.items.some((item) => ["running", "frozen", "importing", "incomplete"].includes(item.status)))
          timer = setTimeout(load, result.items.some((item) => item.status === "incomplete") ? 15000 : 5000);
      } catch (error) {
        if (active) {
          setLoadError(error instanceof Error ? error.message : tr("error"));
          setNotice(error instanceof Error ? error.message : tr("error"));
          timer = setTimeout(load, 10000);
        }
      }
    };
    load();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [cursor, reload]);
  return (
    <main className="report-page">
      <div className="report-hero">
        <span className="eyebrow">EVALUATION / 03</span>
        <h1>{tr("experiments")}</h1>
        <p>{tr("judgedBy")}</p>
      </div>
      <div className="experiment-intro">
        <FlaskConical size={25} />
        <div>
          <strong>{tr("outcomes")}</strong>
          <p>{tr("modelRule")}</p>
        </div>
      </div>
      <div className="report-list-head">
        <h2>{tr("experiments")}</h2>
        <button
          className="secondary-button"
          onClick={() => setReload((value) => value + 1)}
        >
          <RefreshCw size={14} />
          Refresh
        </button>
        <span className="mono">
          {String(selected.length).padStart(2, "0")} {tr("rows")}
        </span>
      </div>
      {loadError && <p className="error-text" role="alert">{loadError} · Retrying…</p>}
      {loadedCursor !== cursor ? (
        <div className="report-empty">{tr("dataPending")}</div>
      ) : visibleExperiments.length === 0 ? (
        <div className="report-empty">
          <FlaskConical size={30} />
          <h3>{tr("noExperiments")}</h3>
          <p>{tr("dataPending")}</p>
        </div>
      ) : (
        <>
          <label className="campaign-select">
            <span>Dataset / campaign</span>
            <select
              value={selectedCampaign}
              onChange={(event) => setCampaignId(event.target.value)}
            >
              {campaigns.map((item) => (
                <option key={item.campaign_id} value={item.campaign_id}>
                  {item.name.split(" · ")[0]} ·{" "}
                  {localeDate(item.created_at)} · {item.status}
                </option>
              ))}
            </select>
          </label>
          <ExperimentComparison items={selected} />
        </>
      )}
      {(nextCursor || cursors.length > 1) && (
        <div className="source-pagination" aria-label="Experiment pages">
          <button className="text-button" disabled={loadedCursor !== cursor || cursors.length === 1} onClick={() => setCursors((items) => items.slice(0, -1))}>Previous</button>
          <span className="mono">{cursors.length}</span>
          <button className="text-button" disabled={loadedCursor !== cursor || !nextCursor} onClick={() => nextCursor && setCursors((items) => [...items, nextCursor])}>Next</button>
        </div>
      )}
    </main>
  );
}

function ExperimentRunLinks({ runIds }: { runIds: string[] }) {
  if (!runIds.length) return <>—</>;
  return (
    <>
      {runIds.map((runId, index) => (
        <div key={runId}>
          <Link to={`/runs/${runId}`}>Run {index + 1} · {shortId(runId)}</Link>
        </div>
      ))}
    </>
  );
}

function ExperimentPage({
  tr,
  setNotice,
}: {
  tr: I18n;
  setNotice: (value: string | null) => void;
}) {
  const { experimentId } = useParams();
  const [reload, setReload] = useState(0);
  const [experiment, setExperiment] = useState<ExperimentDetail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  useEffect(() => {
    if (!experimentId) return;
    setLoadError(null);
    let active = true;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const result = await api<ExperimentDetail>(
          `/experiments/${encodeURIComponent(experimentId)}`,
        );
        if (!active) return;
        setExperiment(result);
        setLoadError(null);
        if (["running", "frozen", "importing", "incomplete"].includes(result.status))
          timer = setTimeout(load, result.status === "incomplete" ? 15000 : 5000);
      } catch (error) {
        if (active) {
          setLoadError(error instanceof Error ? error.message : tr("error"));
          setNotice(error instanceof Error ? error.message : tr("error"));
          timer = setTimeout(load, 10000);
        }
      }
    };
    load();
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [experimentId, reload]);
  if (!experiment || experiment.id !== experimentId)
    return (
      <div className="report-page">
        <div className="report-empty" role={loadError ? "alert" : "status"}>
          {loadError ? `${loadError} · Retrying…` : tr("dataPending")}
        </div>
      </div>
    );
  return (
    <main className="report-page">
      {loadError && <p className="error-text" role="alert">{loadError} · Retrying…</p>}
      <Link className="back-link" to="/experiments">
        <ArrowLeft size={15} />
        {tr("back")}
      </Link>
      <div className="report-hero">
        <span className="eyebrow">
          EVALUATION / {localeDate(experiment.created_at)}
        </span>
        <h1>{experiment.name}</h1>
        <p>
          {experiment.model} / {experiment.variant} · {tr("judgedBy")}
        </p>
      </div>
      <div className="stat-strip">
        <StatCell label={tr("planned")} value={experiment.planned} />
        <StatCell label="Primary completed" value={experiment.completed} />
        <StatCell label={tr("status")} value={experiment.status} />
      </div>
      <div className="report-list-head">
        <h2>{tr("metrics")}</h2>
        <button
          className="secondary-button"
          onClick={() => setReload((value) => value + 1)}
        >
          <RefreshCw size={14} />
          Refresh
        </button>
      </div>
      {experiment.metrics && Object.keys(experiment.metrics).length ? (
        <ExperimentSummary metrics={experiment.metrics} />
      ) : (
        <p>{tr("noMetrics")}</p>
      )}
      {experiment.breakdown && (
        <ExperimentBreakdown value={experiment.breakdown} />
      )}
      <div className="report-list-head">
        <h2>{tr("cases")}</h2>
        <span className="mono">{experiment.cases?.length || 0}</span>
      </div>
      {experiment.cases?.length ? (
        <div className="table-scroll">
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Family</th>
                <th>Primary result</th>
                <th>Primary runs</th>
                <th>Latest correction</th>
              </tr>
            </thead>
            <tbody>
              {experiment.cases.map((item, index) => {
                const corrected = Boolean(
                  item.primary.attempt_id &&
                    item.latest.attempt_id &&
                    (item.primary.attempt_id !== item.latest.attempt_id ||
                      item.primary.judge_request_id !== item.latest.judge_request_id),
                );
                return (
                  <tr key={item.case_id || index}>
                    <td className="mono">{item.case_id || index + 1}</td>
                    <td className="mono">{item.family_id || "—"}</td>
                    <td>
                      {item.primary.judge_verdict ||
                        (item.primary.attempt_id
                          ? item.status === "completed"
                            ? "Not assessed"
                            : item.status
                          : "Pending")}
                      {item.primary.attempt_id && item.primary.judge_verdict && (
                        <div className="muted">{item.status}</div>
                      )}
                      {item.primary.attempt_id && (
                        <div className="muted mono" title={item.primary.attempt_id}>
                          Attempt {shortId(item.primary.attempt_id)}
                        </div>
                      )}
                    </td>
                    <td>
                      <ExperimentRunLinks runIds={item.primary.run_ids} />
                    </td>
                    <td>
                      {corrected ? (
                        <>
                          <div>{item.latest.judge_verdict || item.latest.judge_status || item.latest.state}</div>
                          <div className="muted">{item.latest.state}{item.latest.retry_reason ? ` · ${item.latest.retry_reason}` : ""}</div>
                          {item.latest.attempt_id && (
                            <div className="muted mono" title={item.latest.attempt_id}>
                              Attempt {shortId(item.latest.attempt_id)}
                            </div>
                          )}
                          {item.latest.judge_request_id && (
                            <div className="muted mono" title={item.latest.judge_request_id}>
                              Judge {shortId(item.latest.judge_request_id)}
                            </div>
                          )}
                          <ExperimentRunLinks runIds={item.latest.run_ids} />
                        </>
                      ) : (
                        "—"
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="report-empty">{tr("noCases")}</div>
      )}
    </main>
  );
}

export default App;
