import { useEffect, useMemo, useRef, useState } from "react";
import sampleRoomJson from "../../data/rooms/sample_living_room.json";
import {
  createSession,
  getHealth,
  getPlan,
  parseRoomText,
  streamChat,
  type DesignPlan,
  type HealthStatus,
  type RankedProduct,
  type Room,
  type UsageMetrics,
} from "./api";
import { Chat, type TranscriptMessage } from "./components/Chat";
import { MetricsBar } from "./components/MetricsBar";
import { PlanView } from "./components/PlanView";
import { ProductList } from "./components/ProductList";
import { RoomPanel } from "./components/RoomPanel";

const sampleText = JSON.stringify(sampleRoomJson, null, 2);

function readRoom(text: string): { room: Room | null; error: string | null } {
  try {
    return { room: parseRoomText(text), error: null };
  } catch (error) {
    return { room: null, error: error instanceof Error ? error.message : "Invalid room JSON." };
  }
}

export function App() {
  const [jsonText, setJsonText] = useState(sampleText);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionRoomText, setSessionRoomText] = useState<string | null>(null);
  const [messages, setMessages] = useState<TranscriptMessage[]>([]);
  const [plan, setPlan] = useState<DesignPlan | null>(null);
  const [products, setProducts] = useState<RankedProduct[]>([]);
  const [metrics, setMetrics] = useState<UsageMetrics | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const sendingRef = useRef(false);

  const parsed = useMemo(() => readRoom(jsonText), [jsonText]);

  useEffect(() => {
    let cancelled = false;
    getHealth()
      .then((status) => {
        if (!cancelled) {
          setHealth(status);
          setHealthError(null);
        }
      })
      .catch((reason: unknown) => {
        if (!cancelled) {
          setHealth(null);
          setHealthError(reason instanceof Error ? reason.message : "API unreachable.");
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  function onBudgetChange(budget: number) {
    if (!parsed.room || Number.isNaN(budget)) return;
    setJsonText(JSON.stringify({ ...parsed.room, budget_usd: budget }, null, 2));
  }

  async function startSession() {
    if (!parsed.room || opening || sendingRef.current) return;
    setOpening(true);
    setError(null);
    try {
      const nextId = await createSession(parsed.room);
      setSessionId(nextId);
      setSessionRoomText(JSON.stringify(parsed.room));
      setMessages([]);
      setPlan(null);
      setProducts([]);
      setMetrics(null);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not open a session.");
    } finally {
      setOpening(false);
    }
  }

  async function sendMessage(message: string) {
    if (!sessionId || sendingRef.current) return;
    sendingRef.current = true;
    setStreaming(true);
    setError(null);
    setMessages((current) => [
      ...current,
      { role: "user", content: message },
      { role: "assistant", content: "" },
    ]);
    try {
      await streamChat(sessionId, message, {
        onToken(text) {
          setMessages((current) => {
            const next = current.slice();
            const last = next[next.length - 1];
            if (!last || last.role !== "assistant") return current;
            next[next.length - 1] = { role: "assistant", content: last.content + text };
            return next;
          });
        },
        onPlan(payload) {
          setPlan(payload.plan);
          setProducts(payload.products);
          setMetrics(payload.metrics);
        },
        onDone() {
          /* Metrics and the plan are already on screen; the saved session is read next. */
        },
      });
      const saved = await getPlan(sessionId);
      setPlan(saved.plan);
      setProducts(saved.products);
      setMetrics(saved.metrics);
    } catch (reason) {
      const detail = reason instanceof Error ? reason.message : "The planning turn failed.";
      setError(detail);
      setMessages((current) => {
        const next = current.slice();
        const last = next[next.length - 1];
        if (last && last.role === "assistant" && last.content.length === 0) {
          next[next.length - 1] = { role: "assistant", content: detail };
        }
        return next;
      });
    } finally {
      sendingRef.current = false;
      setStreaming(false);
    }
  }

  const healthLabel = healthError
    ? "API offline"
    : health?.status === "ok"
      ? "Index ready"
      : health
        ? "Index degraded"
        : "Checking API";
  const healthTitle = health
    ? `chroma ${health.chroma_ready ? "ready" : "not ready"}, embedder ${health.embedder_ready ? "ready" : "not ready"}`
    : (healthError ?? "Checking http://localhost:8000/api/health");

  return (
    <div className="app">
      <header className="mast">
        <div>
          <p className="kicker">Nestingale</p>
          <h1>Compose the living room.</h1>
          <p className="lede">A room scan, a budget, and a catalog-grounded plan — written as you talk.</p>
        </div>
        <div className="mast-meta">
          <p className={`health health-${healthError ? "offline" : health?.status === "ok" ? "ok" : "wait"}`} title={healthTitle}>
            <span className="health-dot" aria-hidden="true" />
            {healthLabel}
          </p>
          <p className="session">{sessionId ? `Session ${sessionId.slice(0, 8)}` : "No session yet"}</p>
        </div>
      </header>

      <MetricsBar metrics={metrics} />
      {error ? (
        <p className="banner" role="alert">
          {error}
        </p>
      ) : null}

      <main className="spread">
        <RoomPanel
          jsonText={jsonText}
          room={parsed.room}
          roomError={parsed.error}
          sessionId={sessionId}
          roomDirty={sessionRoomText !== null && JSON.stringify(parsed.room) !== sessionRoomText}
          busy={opening || streaming}
          opening={opening}
          onJsonChange={setJsonText}
          onBudgetChange={onBudgetChange}
          onReloadSample={() => setJsonText(sampleText)}
          onStartSession={() => {
            void startSession();
          }}
        />
        <Chat messages={messages} streaming={streaming} canSend={sessionId !== null && !opening} onSend={(message) => void sendMessage(message)} />
        <div className="rail">
          <PlanView plan={plan} />
          <ProductList products={products} />
        </div>
      </main>
    </div>
  );
}
