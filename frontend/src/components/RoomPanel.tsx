import type { Room } from "../api";
import { formatUsd } from "../api";

interface RoomPanelProps {
  jsonText: string;
  room: Room | null;
  roomError: string | null;
  sessionId: string | null;
  roomDirty: boolean;
  busy: boolean;
  opening: boolean;
  onJsonChange: (text: string) => void;
  onBudgetChange: (budget: number) => void;
  onReloadSample: () => void;
  onStartSession: () => void;
}

export function RoomPanel({
  jsonText,
  room,
  roomError,
  sessionId,
  roomDirty,
  busy,
  opening,
  onJsonChange,
  onBudgetChange,
  onReloadSample,
  onStartSession,
}: RoomPanelProps) {
  return (
    <section className="panel room-panel" aria-labelledby="room-heading">
      <header className="panel-head">
        <p className="kicker">Room</p>
        <h2 id="room-heading">The scan</h2>
      </header>

      {room ? (
        <dl className="spec">
          <div>
            <dt>Size</dt>
            <dd>
              {room.dimensions.length_m} × {room.dimensions.width_m} × {room.dimensions.height_m} m
            </dd>
          </div>
          <div>
            <dt>Style</dt>
            <dd>{room.style_preference}</dd>
          </div>
          <div>
            <dt>Budget</dt>
            <dd>{formatUsd(room.budget_usd)}</dd>
          </div>
          <div>
            <dt>Openings</dt>
            <dd>
              {room.openings.length === 0
                ? "None"
                : room.openings.map((opening) => `${opening.type} · ${opening.wall} · ${opening.width_m} m`).join("; ")}
            </dd>
          </div>
          <div>
            <dt>Already there</dt>
            <dd>
              {room.existing_furniture.length === 0
                ? "Empty room"
                : `${room.existing_furniture.length} piece${room.existing_furniture.length === 1 ? "" : "s"}`}
            </dd>
          </div>
          {room.user_notes ? (
            <div className="spec-wide">
              <dt>Notes</dt>
              <dd>{room.user_notes}</dd>
            </div>
          ) : null}
        </dl>
      ) : (
        <p className="muted">Fix the room JSON to see the summary.</p>
      )}

      <label className="field" htmlFor="budget">
        <span>Budget (USD)</span>
        <input
          id="budget"
          type="number"
          min={0}
          step={50}
          inputMode="decimal"
          value={room ? room.budget_usd : ""}
          disabled={!room || busy}
          onChange={(event) => onBudgetChange(event.target.valueAsNumber)}
        />
      </label>

      <label className="field" htmlFor="room-json">
        <span>Room JSON</span>
        <textarea
          id="room-json"
          className="json"
          spellCheck={false}
          value={jsonText}
          disabled={busy}
          onChange={(event) => onJsonChange(event.target.value)}
          rows={14}
        />
      </label>
      {roomError ? (
        <p className="field-error" role="status">
          {roomError}
        </p>
      ) : null}
      {sessionId && roomDirty ? (
        <p className="hint">This room differs from the open session. Start again to apply it.</p>
      ) : null}

      <div className="actions">
        <button type="button" className="primary" onClick={onStartSession} disabled={busy || !room}>
          {opening ? "Opening…" : sessionId ? "Start a new session" : "Open this room"}
        </button>
        <button type="button" className="secondary" onClick={onReloadSample} disabled={busy}>
          Reload sample
        </button>
      </div>
    </section>
  );
}
