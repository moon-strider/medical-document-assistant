import { Link } from "react-router-dom";
import { useState } from "react";
import type { Experiment } from "./api";
import { documentClassLabels, questionClassLabels } from "./taxonomy";

type Metric = Record<string, unknown>;

export type Breakdown = Record<string, Record<string, Metric>>;

function object(value: unknown): Metric {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Metric)
    : {};
}

function number(value: unknown): number | null {
  if (value === null || value === undefined || value === "") return null;
  const result = Number(value);
  return Number.isFinite(result) ? result : null;
}

function fraction(value: unknown) {
  const metric = object(value);
  const denominator = number(metric.denominator);
  const numerator = number(metric.numerator);
  return denominator && numerator !== null
    ? `${numerator} / ${denominator}`
    : "—";
}

function percent(value: unknown) {
  const metric = object(value);
  return number(metric.denominator)
    ? Math.max(0, Math.min(100, (number(metric.value) || 0) * 100))
    : null;
}

function milliseconds(value: unknown) {
  const amount = number(value);
  return amount === null ? "—" : `${(amount / 1000).toFixed(2)} s`;
}

function money(value: unknown) {
  const group = object(value);
  const total = number(group.estimated_total_api_usd);
  const known = number(group.known_subtotal_api_usd);
  const missing = number(group.unestimated_call_count) || 0;
  const format = (amount: number) =>
    new Intl.NumberFormat("en", {
      style: "currency",
      currency: "USD",
      minimumFractionDigits: 2,
      maximumFractionDigits: 6,
    }).format(amount);
  if (total !== null) return format(total);
  if (missing && known !== null)
    return `${format(known)} + ${missing} unknown`;
  return "—";
}

function callCount(value: unknown): string {
  const count = number(object(value).call_count);
  return `${count === null ? "unknown" : count} calls`;
}

function CostNote() {
  return (
    <p className="metric-note">
      Estimated API cost from reported tokens and a versioned price snapshot, including subscription calls. This does not infer the actual bill. Answer generation and Sol judging are shown separately.
    </p>
  );
}

export function ExperimentBreakdown({
  value,
}: {
  value: Breakdown;
}) {
  const [selected, setSelected] = useState("anchor_document_class");
  const axes: Record<string, string> = {
    anchor_document_class: "Primary document class",
    question_class: "Question class",
    required_document_class: "All required document classes",
    anchor_format: "Primary source format",
    difficulty: "Difficulty",
    gold_status: "Gold answer status",
    turn_count: "Conversation turns",
  };
  const labels: Record<string, string> = {
    ...documentClassLabels,
    ...questionClassLabels,
  };
  const available = Object.keys(axes).filter((axis) => value[axis]);
  const axis = available.includes(selected) ? selected : available[0];
  if (!axis) return null;
  return (
    <section>
      <div className="report-list-head">
        <h2>Results by group</h2>
      </div>
      <label className="campaign-select">
        <span>Group by</span>
        <select
          value={axis}
          onChange={(event) => setSelected(event.target.value)}
        >
          {available.map((key) => (
            <option key={key} value={key}>
              {axes[key]}
            </option>
          ))}
        </select>
      </label>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <th>Group</th>
              <th>n</th>
              <th>Primary confirmed</th>
              <th>Assessable</th>
              <th>Evidence delivered</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(value[axis]).map(([key, metric]) => (
              <tr key={key}>
                <td>{labels[key] || key}</td>
                <td className="mono">{String(metric.planned ?? "—")}</td>
                <td className="mono">
                  {fraction(metric.confirmed_grounded_success)}
                </td>
                <td className="mono">{fraction(metric.judge_assessable)}</td>
                <td className="mono">
                  {fraction(metric.evidence_case_availability)}
                  <small>Unknown final input: {String(metric.evidence_unknown_positive_cases ?? "—")}</small>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="metric-note">
        Each case has one primary document class. Required classes can overlap, so their counts can exceed the case total. This constructed synthetic sample does not estimate clinical population frequencies.
      </p>
    </section>
  );
}

export function ExperimentComparison({
  items,
}: {
  items: Experiment[];
}) {
  return (
    <>
      <div className="table-scroll">
        <table className="data-table comparison-table">
          <thead>
            <tr>
              <th>Variant / model</th>
              <th>Primary confirmed / planned</th>
              <th>Latest projection confirmed / planned</th>
              <th>Assessable / planned</th>
              <th>Primary completed</th>
              <th>Answer latency p50</th>
              <th>API estimate · answers</th>
              <th>API estimate · judge</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => {
              const metric = object(item.metrics);
              const rate = percent(metric.confirmed_grounded_success);
              const costs = object(metric.api_equivalent_cost);
              return (
                <tr key={item.id}>
                  <td>
                    <Link
                      to={`/experiments/${encodeURIComponent(item.id)}`}
                      className="comparison-name"
                    >
                      {item.variant} /{" "}
                      {item.model === "gpt-6-sol" ? "Sol" : "Luna"}
                    </Link>
                  </td>
                  <td>
                    <strong>
                      {fraction(metric.confirmed_grounded_success)}
                    </strong>
                    <div className="comparison-bar" aria-hidden="true">
                      <span style={{ width: `${rate || 0}%` }} />
                    </div>
                  </td>
                  <td className="mono">
                    {fraction(item.latest_correction.confirmed_grounded_success)}
                    <small>Corrected rows: {item.latest_correction.attempted_rows}</small>
                  </td>
                  <td className="mono">{fraction(metric.judge_assessable)}</td>
                  <td className="mono">
                    {item.completed} / {item.planned}
                  </td>
                  <td className="mono">
                    {milliseconds(
                      object(metric.latency_target_completed).p50_ms,
                    )}
                  </td>
                  <td className="mono">{money(costs.generation)}</td>
                  <td className="mono">{money(costs.judge)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="metric-note">
        Primary confirmed cases use all planned cases as the denominator and never change after a retry. The latest projection also covers all planned cases; corrected rows counts only cases with a later answer or judge attempt. Pending or unassessable cases are not labelled semantic failures. Latency uses completed primary answers with the sample size shown.
      </p>
      <CostNote />
    </>
  );
}

export function ExperimentSummary({
  metrics,
}: {
  metrics: Metric;
}) {
  const entries = [
    [
      "Primary confirmed / planned",
      "confirmed_grounded_success",
    ],
    [
      "Completed / planned",
      "technical_completion",
    ],
    ["Assessable / planned", "judge_assessable"],
    [
      "Correct answer status",
      "answer_status_exact",
    ],
    [
      "Required claims present",
      "required_claim_recall_conditional",
    ],
    [
      "Supported answer claims",
      "own_claim_support",
    ],
    [
      "Useful citation links",
      "citation_edge_precision",
    ],
    [
      "Evidence sets in final model input",
      "final_packet",
    ],
    [
      "Evidence sets available in index",
      "indexed_evidence_case_availability",
    ],
    [
      "Original traces delivered",
      "trace_original_completed",
    ],
  ];
  const target = object(metrics.latency_target_completed);
  const total = object(metrics.latency_total_completed);
  const costs = object(metrics.api_equivalent_cost);
  const latestCorrection = object(metrics.latest_correction);
  const finalPacket = object(object(metrics.diagnostic_evidence).final_packet);
  return (
    <>
      <div className="evaluation-metrics">
        {entries.map(([label, key]) => {
          const metric = key === "final_packet" ? finalPacket.cases : metrics[key];
          return (
            <div className="stat-cell" key={key}>
              <span>{label}</span>
              <strong>{fraction(metric)}</strong>
              <small>
                {percent(metric) === null ? "N/A" : `${percent(metric)!.toFixed(1)}%`}
                {key === "final_packet" && ` · Unknown final input: ${String(finalPacket.unknown_positive_cases ?? 0)}`}
              </small>
            </div>
          );
        })}
        <div className="stat-cell">
          <span>Latest projection confirmed / planned</span>
          <strong>{fraction(latestCorrection.confirmed_grounded_success)}</strong>
          <small>Corrected rows: {String(latestCorrection.attempted_rows ?? 0)}</small>
        </div>
      </div>
      <div className="evaluation-metrics compact-metrics">
        <div className="stat-cell">
          <span>Answer latency p50 / p90</span>
          <strong>
            {milliseconds(target.p50_ms)} / {milliseconds(target.p90_ms)}
          </strong>
          <small>n = {String(target.n ?? 0)}</small>
        </div>
        <div className="stat-cell">
          <span>Whole case latency p50 / p90</span>
          <strong>
            {milliseconds(total.p50_ms)} / {milliseconds(total.p90_ms)}
          </strong>
          <small>n = {String(total.n ?? 0)}</small>
        </div>
        <div className="stat-cell">
          <span>API estimate · answers</span>
          <strong>{money(costs.generation)}</strong>
          <small>
            {callCount(costs.generation)}
          </small>
        </div>
        <div className="stat-cell">
          <span>API estimate · Sol judge</span>
          <strong>{money(costs.judge)}</strong>
          <small>
            {callCount(costs.judge)}
          </small>
        </div>
      </div>
      <CostNote />
    </>
  );
}
