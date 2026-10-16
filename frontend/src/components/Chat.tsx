import { useEffect, useRef, useState } from "react";

export interface TranscriptMessage {
  role: "user" | "assistant";
  content: string;
}

interface ChatProps {
  messages: TranscriptMessage[];
  streaming: boolean;
  canSend: boolean;
  onSend: (message: string) => void;
}

const PROMPTS = [
  "Need a sofa that faces the window and a coffee table.",
  "Add a rug and a floor lamp that stay inside the budget.",
];

export function Chat({ messages, streaming, canSend, onSend }: ChatProps) {
  const [draft, setDraft] = useState(PROMPTS[0]);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const log = logRef.current;
    if (log) log.scrollTop = log.scrollHeight;
  }, [messages, streaming]);

  function submit() {
    const message = draft.trim();
    if (!message || !canSend || streaming) return;
    onSend(message);
    setDraft("");
  }

  return (
    <section className="panel chat-panel" aria-labelledby="chat-heading">
      <header className="panel-head">
        <p className="kicker">Conversation</p>
        <h2 id="chat-heading">Ask for the room</h2>
      </header>

      <div className="chat-log" ref={logRef} aria-live="polite">
        {messages.length === 0 ? (
          <p className="muted empty-chat">
            Open the room, then ask for pieces. Replies stream in, and the plan lands when the turn finishes.
          </p>
        ) : (
          messages.map((message, index) => {
            const live = streaming && index === messages.length - 1 && message.role === "assistant";
            return (
              <article key={`${message.role}-${index}`} className={`turn turn-${message.role}${live ? " is-streaming" : ""}`}>
                <p className="turn-role">{message.role === "user" ? "You" : "Planner"}</p>
                <p className="bubble">{message.content || (live ? "" : "…")}</p>
              </article>
            );
          })
        )}
      </div>

      <div className="prompts">
        {PROMPTS.map((prompt) => (
          <button key={prompt} type="button" className="chip" onClick={() => setDraft(prompt)} disabled={streaming}>
            {prompt}
          </button>
        ))}
      </div>

      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          submit();
        }}
      >
        <label className="sr-only" htmlFor="message">
          Message
        </label>
        <textarea
          id="message"
          rows={3}
          value={draft}
          placeholder={canSend ? "Describe what the room still needs" : "Open a room session first"}
          disabled={!canSend || streaming}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              submit();
            }
          }}
        />
        <button type="submit" className="primary" disabled={!canSend || streaming || draft.trim().length === 0}>
          {streaming ? "Planning…" : "Send"}
        </button>
      </form>
    </section>
  );
}
