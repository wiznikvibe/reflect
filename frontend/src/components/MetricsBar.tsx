import type { UsageMetrics } from "../api";
import { formatCost, formatCount, formatLatency } from "../api";

interface MetricsBarProps {
  metrics: UsageMetrics | null;
}

export function MetricsBar({ metrics }: MetricsBarProps) {
  const cells = [
    { label: "Latency", value: metrics ? formatLatency(metrics.latency_ms) : "—" },
    { label: "Prompt tokens", value: metrics ? formatCount(metrics.prompt_tokens) : "—" },
    { label: "Completion tokens", value: metrics ? formatCount(metrics.completion_tokens) : "—" },
    { label: "Approx. cost", value: metrics ? formatCost(metrics.approx_cost_usd) : "—" },
  ];

  return (
    <section className="metrics" aria-label="Last turn metrics">
      <p className="metrics-label">Last turn</p>
      <dl>
        {cells.map((cell) => (
          <div key={cell.label}>
            <dt>{cell.label}</dt>
            <dd>{cell.value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
