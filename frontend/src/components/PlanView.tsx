import type { DesignPlan } from "../api";
import { formatUsd } from "../api";

interface PlanViewProps {
  plan: DesignPlan | null;
}

export function PlanView({ plan }: PlanViewProps) {
  return (
    <section className="panel plan-panel" aria-labelledby="plan-heading">
      <header className="panel-head">
        <p className="kicker">Plan</p>
        <h2 id="plan-heading">{plan ? plan.style_direction : "Waiting on a turn"}</h2>
      </header>

      {!plan ? (
        <p className="muted">Plan cards appear after the first reply: style, placements, and what the budget still allows.</p>
      ) : (
        <>
          <p className="summary">{plan.summary}</p>
          {plan.items.length === 0 ? (
            <p className="muted">No pieces stayed inside the budget and the catalog.</p>
          ) : (
            <ol className="plan-cards">
              {plan.items.map((item, index) => (
                <li key={`${item.product_id}-${index}`} className="plan-card">
                  <p className="rank">{String(index + 1).padStart(2, "0")}</p>
                  <div>
                    <h3>{item.role}</h3>
                    <p className="sku">{item.product_id}</p>
                    <p>{item.placement_note}</p>
                  </div>
                </li>
              ))}
            </ol>
          )}
          <dl className="totals">
            <div>
              <dt>Plan total</dt>
              <dd>{formatUsd(plan.total_usd)}</dd>
            </div>
            <div>
              <dt>Still available</dt>
              <dd>{formatUsd(plan.budget_remaining_usd)}</dd>
            </div>
          </dl>
          {plan.constraint_notes.length > 0 ? (
            <div className="notes">
              <h3>Constraint notes</h3>
              <ul>
                {plan.constraint_notes.map((note, index) => (
                  <li key={`${note}-${index}`}>{note}</li>
                ))}
              </ul>
            </div>
          ) : null}
          {plan.alternatives.length > 0 ? (
            <div className="notes">
              <h3>Alternatives</h3>
              <ul>
                {plan.alternatives.map((item, index) => (
                  <li key={`${item.product_id}-${index}`}>
                    <strong>{item.role}</strong> · {item.product_id}. {item.why_it_fits}
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}
