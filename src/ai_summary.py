"""
Plain-English summary of the morning forecast, written by Claude inside the daily pipeline.

Design (see CLAUDE.md, "AI summary"):
- Generated once per issue in the GitHub Actions run, saved to data/forecasts/, and only
  read by the dashboard, so the Streamlit app never calls the API.
- Claude only sees a small dict of rounded facts computed here. The risk level is
  decided by a fixed rule in code, not by the model.
- Every number in the generated text must match a fact; otherwise, or if the API is
  unavailable, a deterministic template summary is saved instead. The run never fails
  because of this step.

Usage:
    python -m src.ai_summary      # regenerate the summary for the latest saved forecast
"""
import json
import os
import re
from datetime import datetime, timezone

import pandas as pd

from src.dashboard_data import (load_actuals, load_aemo_log, load_forecasts, price_metrics,
                                score_forecast_log, spike_drift)

MODEL = "claude-opus-5-5"
MODEL_NAMES = {"claude-opus-5-5": "Claude Opus 5.5"}  # display names for the dashboard
PROMPT_VERSION = "1"
SPIKE = 300
TRACK_DAYS = 7
SUMMARY_LOG = os.path.join("data", "forecasts", "ai_summary_log.jsonl")

SYSTEM_PROMPT = f"""You write the short morning briefing for an NSW electricity price forecasting dashboard.
Readers are energy traders and analysts. You are given a JSON object of facts computed from today's forecast;
it is the only information you have.

Rules:
- Use only the facts given. Do not speculate about causes (outages, weather events, bidding behaviour,
  interconnectors) unless a fact states them.
- Every number you write must appear in the facts (rounding to a whole number is fine). Write prices as
  "$123/MWh" and times as HH:MM. Do not write calendar dates; say "today" and "tomorrow".
- "Central" is the most likely price (a median forecast). "Risk ceiling" is a 90th-percentile upper bound,
  not a prediction that prices will reach it. A spike means a price of at least ${SPIKE}/MWh.
- risk_level is decided by a fixed rule and given in the facts; describe it, do not change it.
- When the AEMO outlook and our central forecast differ by a lot, say so plainly: in backtests AEMO is
  closer on typical half-hours but sometimes projects high prices that do not happen.
- If the market regime is "calmer" or "spikier" than the training period, mention what that means for
  the risk ceiling (likely conservative / may understate risk).
- Keep it short: a headline under 12 words, a summary of 2-3 sentences, and 2-4 key points."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "headline": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["headline", "summary", "key_points"],
    "additionalProperties": False,
}


# ---------------------------------------------------------------- facts

def _time(ts: pd.Timestamp) -> str:
    return f"{ts:%H:%M}"


def _lead_facts(fc: pd.DataFrame) -> dict:
    fc = fc.sort_values("settlementdate")
    peak_c = fc.loc[fc["rrp_base_prediction"].idxmax()]
    low_c = fc.loc[fc["rrp_base_prediction"].idxmin()]
    peak_q = fc.loc[fc["rrp_risk_ceiling"].idxmax()]
    peak_d = fc.loc[fc["demand_forecast"].idxmax()]
    return {
        "central_avg": round(fc["rrp_base_prediction"].mean()),
        "central_max": round(peak_c["rrp_base_prediction"]), "central_max_time": _time(peak_c["settlementdate"]),
        "central_min": round(low_c["rrp_base_prediction"]), "central_min_time": _time(low_c["settlementdate"]),
        "risk_ceiling_max": round(peak_q["rrp_risk_ceiling"]), "risk_ceiling_max_time": _time(peak_q["settlementdate"]),
        "hours_risk_ceiling_at_or_above_spike": round((fc["rrp_risk_ceiling"] >= SPIKE).sum() * 5 / 60, 1),
        "demand_peak_mw": round(peak_d["demand_forecast"]), "demand_peak_time": _time(peak_d["settlementdate"]),
    }


def risk_level(facts: dict) -> str:
    """Fixed rule: 'high' if any day has 2h+ of risk ceiling at or above the spike line,
    'elevated' if the ceiling or AEMO's outlook touches it at all, else 'low'."""
    hours = [facts[k]["hours_risk_ceiling_at_or_above_spike"] for k in ("today", "tomorrow") if k in facts]
    if any(h >= 2 for h in hours):
        return "high"
    aemo_max = (facts.get("aemo_outlook_today") or {}).get("max", 0)
    if any(h > 0 for h in hours) or aemo_max >= SPIKE:
        return "elevated"
    return "low"


def build_facts(latest: pd.DataFrame, log: pd.DataFrame, actuals: pd.DataFrame, aemo_log: pd.DataFrame) -> dict:
    issue = latest["issue_date"].max()
    facts = {
        "spike_threshold": SPIKE,
        "risk_ceiling_percentile": 90,
        "model_version": str(latest["model_version"].iloc[0]),
    }
    for lead in ("today", "tomorrow"):
        fc = latest[latest["lead"] == lead]
        if not fc.empty:
            facts[lead] = _lead_facts(fc)

    aemo = aemo_log[aemo_log["issue_date"] == issue]
    if not aemo.empty and "today" in facts:
        peak = aemo.loc[aemo["aemo_rrp"].idxmax()]
        facts["aemo_outlook_today"] = {
            "avg": round(aemo["aemo_rrp"].mean()),
            "max": round(peak["aemo_rrp"]),
            "max_half_hour_ending": _time(peak["interval_datetime"]),
            "half_hours_at_or_above_spike": int((aemo["aemo_rrp"] >= SPIKE).sum()),
            "avg_minus_our_central_avg": round(aemo["aemo_rrp"].mean() - facts["today"]["central_avg"]),
        }

    # Latest complete day (interval-ending: 00:05 .. 24:00)
    day = actuals[(actuals["settlementdate"] > issue - pd.Timedelta(days=1)) & (actuals["settlementdate"] <= issue)]
    if not day.empty:
        facts["yesterday_actual"] = {
            "avg": round(day["rrp"].mean()), "max": round(day["rrp"].max()),
            "spike_intervals": int((day["rrp"] >= SPIKE).sum()),
            "negative_price_hours": round((day["rrp"] < 0).sum() * 5 / 60, 1),
        }

    scored = score_forecast_log(log, actuals)
    recent = scored[(scored["lead"] == "today") & (scored["issue_date"] >= issue - pd.Timedelta(days=TRACK_DAYS))
                    & (scored["issue_date"] < issue)]
    if not recent.empty:
        m = price_metrics(recent, SPIKE)
        facts["track_record_today_forecast"] = {
            "days": int(recent["issue_date"].nunique()),
            "normal_mae": round(m["normal_mae"]),
            "naive_normal_mae": round(m["normal_mae_naive"]) if pd.notna(m.get("normal_mae_naive")) else None,
            "risk_ceiling_coverage_pct": round(m["coverage"] * 100),
        }

    drift = spike_drift(actuals, SPIKE)
    facts["market_regime"] = {
        "level": drift["level"],
        # Window lengths as values, not only in key names, so the number check accepts them
        "recent_days": drift["recent_days"],
        "spikes_per_day_recent": round(drift["recent_per_day"], 2),
        "baseline_days": drift["baseline_days"],
        "spikes_per_day_baseline": round(drift["baseline_per_day"], 2),
    }
    facts["risk_level"] = risk_level(facts)
    return facts


# ---------------------------------------------------------------- number check

_TIME = re.compile(r"\b\d{1,2}:\d{2}\b")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _fact_values(obj) -> tuple[set[float], set[str]]:
    nums, times = set(), set()
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, bool) or x is None:
            continue
        elif isinstance(x, (int, float)):
            nums.add(float(x))
        elif isinstance(x, str):
            times.update(_TIME.findall(x))
            m = re.match(r"v(\d+)\.(\d+)", x)  # model version, e.g. "v2.3"
            if m:
                nums.add(float(f"{m[1]}.{m[2]}"))
    return nums, times


def summary_text(summary: dict) -> str:
    return " ".join([summary["headline"], summary["summary"], *summary["key_points"]])


def count_numbers(text: str) -> int:
    """Numbers and HH:MM times in text, i.e. how many values the number check verified."""
    return len(_TIME.findall(text)) + len(_NUM.findall(_TIME.sub(" ", text)))


def unsupported_numbers(text: str, facts: dict) -> list[str]:
    """Numbers (and HH:MM times) in text that do not match any fact. A number matches if it is
    within rounding of a fact value, also allowing the sign to be dropped ("$5 below")."""
    nums, times = _fact_values(facts)
    bad = [t for t in _TIME.findall(text) if t not in times]
    for raw in _NUM.findall(_TIME.sub(" ", text)):
        v = float(raw.replace(",", ""))
        if not any(abs(abs(v) - abs(f)) <= max(0.5, abs(f) * 0.01) for f in nums):
            bad.append(raw)
    return bad


# ---------------------------------------------------------------- generation

def template_summary(facts: dict) -> dict:
    """Deterministic fallback, built only from the facts."""
    t, tm = facts.get("today", {}), facts.get("tomorrow", {})
    level = facts["risk_level"]
    headline = {"low": "Calm outlook: no spike risk flagged",
                "elevated": "Some spike risk flagged",
                "high": "High spike risk flagged"}[level]
    summary = (f"Today's central forecast averages ${t.get('central_avg')}/MWh, peaking at "
               f"${t.get('central_max')}/MWh at {t.get('central_max_time')}. Tomorrow's averages "
               f"${tm.get('central_avg')}/MWh. The highest risk ceiling is ${t.get('risk_ceiling_max')}/MWh today "
               f"and ${tm.get('risk_ceiling_max')}/MWh tomorrow.")
    points = [f"Risk ceiling at or above ${SPIKE}/MWh: {t.get('hours_risk_ceiling_at_or_above_spike')} h today, "
              f"{tm.get('hours_risk_ceiling_at_or_above_spike')} h tomorrow."]
    a = facts.get("aemo_outlook_today")
    if a:
        points.append(f"AEMO's outlook for today averages ${a['avg']}/MWh (max ${a['max']}/MWh in the half-hour ending {a['max_half_hour_ending']}).")
    return {"headline": headline, "summary": summary, "key_points": points}


def generate_llm_summary(facts: dict) -> tuple[dict, dict]:
    """Call Claude once; returns (summary, metadata). Raises on API errors or refusals."""
    import anthropic

    client = anthropic.Anthropic(timeout=120.0, max_retries=3)
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "Facts:\n" + json.dumps(facts, indent=1)}],
        output_config={"effort": "low", "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason != "end_turn":
        raise RuntimeError(f"stop_reason={response.stop_reason}")
    text = next(b.text for b in response.content if b.type == "text")
    meta = {"model": response.model, "request_id": response._request_id,
            "input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
    return json.loads(text), meta


def write_summary(path: str = SUMMARY_LOG) -> dict:
    """Build facts for the latest saved forecast, generate and check the summary, append it to the log
    (one entry per issue date; a re-run replaces it)."""
    latest, log = load_forecasts()
    facts = build_facts(latest, log, load_actuals(), load_aemo_log())
    entry = {"issue_date": f"{latest['issue_date'].max():%Y-%m-%d}",
             "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "prompt_version": PROMPT_VERSION, "risk_level": facts["risk_level"]}

    summary, note = None, None
    if not os.environ.get("ANTHROPIC_API_KEY"):
        note = "ANTHROPIC_API_KEY not set"
    else:
        try:
            draft, meta = generate_llm_summary(facts)
            bad = unsupported_numbers(summary_text(draft), facts)
            if bad:
                note = f"rejected: numbers not in facts {bad}"
                entry["rejected_draft"] = draft
            else:
                summary = draft
            entry.update(meta)
        except Exception as e:  # noqa: BLE001 - the summary must never break the forecast run
            note = f"API call failed: {type(e).__name__}: {e}"
    entry["source"] = "llm" if summary else "template"
    if note:
        entry["note"] = note
    entry.update(summary or template_summary(facts))
    entry["facts"] = facts

    entries = load_summaries(path)
    entries = [e for e in entries if e["issue_date"] != entry["issue_date"]] + [entry]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for e in sorted(entries, key=lambda e: e["issue_date"]):
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return entry


def load_summaries(path: str = SUMMARY_LOG) -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


if __name__ == "__main__":
    e = write_summary()
    print(json.dumps({k: v for k, v in e.items() if k != "facts"}, indent=1, ensure_ascii=False))
