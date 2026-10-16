export const API_BASE = "http://localhost:8000";

export type Wall = "north" | "south" | "east" | "west";

export interface Dimensions {
  length_m: number;
  width_m: number;
  height_m: number;
}

export interface Opening {
  type: string;
  wall: Wall;
  width_m: number;
}

export interface Room {
  room_type: "living_room";
  dimensions: Dimensions;
  openings: Opening[];
  existing_furniture: Record<string, unknown>[];
  style_preference: string;
  budget_usd: number;
  user_notes: string;
}

export interface DesignPlanItem {
  product_id: string;
  role: string;
  placement_note: string;
  why_it_fits: string;
}

export interface DesignPlan {
  summary: string;
  style_direction: string;
  items: DesignPlanItem[];
  total_usd: number;
  budget_remaining_usd: number;
  constraint_notes: string[];
  alternatives: DesignPlanItem[];
}

export interface RankedProduct {
  id: string;
  name?: string;
  category?: string;
  style_tags?: string[];
  price_usd: number;
  width_cm?: number;
  depth_cm?: number;
  height_cm?: number;
  material?: string;
  role?: string;
  placement_note?: string;
  why_it_fits?: string;
}

export interface UsageMetrics {
  latency_ms: number;
  prompt_tokens: number;
  completion_tokens: number;
  approx_cost_usd: number;
}

export interface PlanPayload {
  plan: DesignPlan | null;
  products: RankedProduct[];
  metrics: UsageMetrics | null;
}

export interface HealthStatus {
  status: string;
  chroma_ready: boolean;
  embedder_ready: boolean;
}

export interface ChatHandlers {
  onToken: (text: string) => void;
  onPlan: (payload: PlanPayload) => void;
  onDone: () => void;
}

const WALLS = new Set<Wall>(["north", "south", "east", "west"]);

export function parseRoom(input: unknown): Room {
  if (!input || typeof input !== "object") {
    throw new Error("Room JSON must be an object.");
  }
  const room = input as Record<string, unknown>;
  if (room.room_type !== "living_room") {
    throw new Error('room_type must be "living_room".');
  }
  if (!room.dimensions || typeof room.dimensions !== "object") {
    throw new Error("dimensions is required.");
  }
  const dims = room.dimensions as Record<string, unknown>;
  for (const key of ["length_m", "width_m", "height_m"] as const) {
    if (typeof dims[key] !== "number" || !(dims[key] > 0)) {
      throw new Error(`${key} must be a number greater than 0.`);
    }
  }
  if (!Array.isArray(room.openings)) {
    throw new Error("openings must be a list.");
  }
  const openings = room.openings.map((opening, index) => {
    if (!opening || typeof opening !== "object") {
      throw new Error(`openings[${index}] must be an object.`);
    }
    const item = opening as Record<string, unknown>;
    if (typeof item.type !== "string" || item.type.length === 0) {
      throw new Error(`openings[${index}].type is required.`);
    }
    if (typeof item.wall !== "string" || !WALLS.has(item.wall as Wall)) {
      throw new Error(`openings[${index}].wall must be north, south, east, or west.`);
    }
    if (typeof item.width_m !== "number" || !(item.width_m > 0)) {
      throw new Error(`openings[${index}].width_m must be greater than 0.`);
    }
    return { type: item.type, wall: item.wall as Wall, width_m: item.width_m };
  });
  if (!Array.isArray(room.existing_furniture)) {
    throw new Error("existing_furniture must be a list.");
  }
  if (typeof room.style_preference !== "string" || room.style_preference.trim().length === 0) {
    throw new Error("style_preference is required.");
  }
  if (typeof room.budget_usd !== "number" || Number.isNaN(room.budget_usd) || room.budget_usd < 0) {
    throw new Error("budget_usd must be a number greater than or equal to 0.");
  }
  if (room.user_notes !== undefined && typeof room.user_notes !== "string") {
    throw new Error("user_notes must be a string.");
  }
  return {
    room_type: "living_room",
    dimensions: {
      length_m: dims.length_m as number,
      width_m: dims.width_m as number,
      height_m: dims.height_m as number,
    },
    openings,
    existing_furniture: room.existing_furniture as Record<string, unknown>[],
    style_preference: room.style_preference,
    budget_usd: room.budget_usd,
    user_notes: typeof room.user_notes === "string" ? room.user_notes : "",
  };
}

export function parseRoomText(text: string): Room {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    throw new Error("Room JSON is not valid.");
  }
  return parseRoom(value);
}

async function errorMessage(response: Response): Promise<string> {
  const text = await response.text();
  try {
    const body = JSON.parse(text) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            return String((item as { msg: unknown }).msg);
          }
          return JSON.stringify(item);
        })
        .join("; ");
    }
  } catch {
    /* response was not JSON */
  }
  return text.trim() || `Request failed (${response.status}).`;
}

export async function getHealth(): Promise<HealthStatus> {
  const response = await fetch(`${API_BASE}/api/health`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()) as HealthStatus;
}

export async function createSession(room: Room): Promise<string> {
  const response = await fetch(`${API_BASE}/api/sessions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(room),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  const body = (await response.json()) as { session_id?: string };
  if (!body.session_id) throw new Error("Session response did not include session_id.");
  return body.session_id;
}

export async function getPlan(sessionId: string): Promise<PlanPayload> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/plan`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return (await response.json()) as PlanPayload;
}

interface SseEvent {
  event: string;
  data: string;
}

export function parseSseBlock(block: string): SseEvent | null {
  let event = "message";
  const data: string[] = [];
  for (const line of block.split(/\r?\n/)) {
    if (!line || line.startsWith(":")) continue;
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice(5).replace(/^ /, ""));
    }
  }
  if (data.length === 0) return null;
  return { event, data: data.join("\n") };
}

function dispatchSse(block: string, handlers: ChatHandlers): void {
  const parsed = parseSseBlock(block);
  if (!parsed) return;
  const payload = JSON.parse(parsed.data) as Record<string, unknown>;
  if (parsed.event === "token") {
    handlers.onToken(typeof payload.text === "string" ? payload.text : "");
    return;
  }
  if (parsed.event === "plan") {
    handlers.onPlan({
      plan: (payload.plan as DesignPlan | null) ?? null,
      products: Array.isArray(payload.products) ? (payload.products as RankedProduct[]) : [],
      metrics: (payload.metrics as UsageMetrics | null) ?? null,
    });
    return;
  }
  if (parsed.event === "done") {
    handlers.onDone();
  }
}

export async function streamChat(
  sessionId: string,
  message: string,
  handlers: ChatHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API_BASE}/api/sessions/${sessionId}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Accept: "text/event-stream",
    },
    body: JSON.stringify({ message }),
    signal,
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  if (!response.body) throw new Error("Chat response had no stream.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split(/\r?\n\r?\n/);
    buffer = blocks.pop() ?? "";
    for (const block of blocks) dispatchSse(block, handlers);
  }
  buffer += decoder.decode();
  if (buffer.trim()) dispatchSse(buffer, handlers);
}

export function formatUsd(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

export function formatCost(value: number): string {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 4,
    maximumFractionDigits: 6,
  }).format(value);
}

export function formatLatency(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

export function formatCount(value: number): string {
  return new Intl.NumberFormat("en-US").format(value);
}
