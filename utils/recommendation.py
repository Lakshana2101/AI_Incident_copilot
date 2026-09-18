"""
recommendation.py
------------------
Generates an "AI Incident Analysis" (probable issue, supporting evidence,
recommended investigation steps, suggested resolution) using the Groq LLM
API. If no GROQ_API_KEY is set, or the API call fails for any reason, this
module falls back to a simple rule-based recommendation so the app never
crashes and always shows something useful.
"""

import os


def _build_prompt(incident, predicted_category, anomaly_status, anomaly_score,
                   similar_incident, similar_resolution, similarity_percent):
    return f"""
You are an experienced IT Operations assistant helping an engineer triage an incident.
Be careful: you are NOT certain of the root cause. Use cautious language such as
"probable issue", "possible cause", and "recommended investigation" -- never claim
a guaranteed root cause.

CURRENT INCIDENT
- Incident ID: {incident.get('incident_id')}
- Error message: {incident.get('error_message')}
- CPU usage: {incident.get('cpu_usage')}%
- Memory usage: {incident.get('memory_usage')}%
- Response time: {incident.get('response_time')} ms
- Error count: {incident.get('error_count')}
- Network latency: {incident.get('network_latency')} ms
- Predicted category (ML model): {predicted_category}
- Anomaly status (ML model): {anomaly_status} (anomaly score: {anomaly_score:.3f})

MOST SIMILAR HISTORICAL INCIDENT (similarity: {similarity_percent}%)
- Error message: {similar_incident}
- How it was resolved previously: {similar_resolution}

Please respond in this exact format, with short, clear bullet points:

Probable Issue:
<1-2 sentences>

Supporting Evidence:
- <bullet>
- <bullet>

Recommended Investigation Steps:
- <bullet>
- <bullet>
- <bullet>

Suggested Resolution:
- <bullet>
- <bullet>
"""


def _rule_based_fallback(incident, predicted_category, anomaly_status,
                          similar_resolution, similarity_percent):
    """A simple, deterministic fallback that needs no external API."""

    high_cpu = incident.get("cpu_usage", 0) > 80
    high_mem = incident.get("memory_usage", 0) > 80
    high_latency = incident.get("network_latency", 0) > 250
    high_errors = incident.get("error_count", 0) > 10

    evidence = []
    if high_cpu:
        evidence.append(f"CPU usage is elevated at {incident.get('cpu_usage')}%.")
    if high_mem:
        evidence.append(f"Memory usage is elevated at {incident.get('memory_usage')}%.")
    if high_latency:
        evidence.append(f"Network latency is high at {incident.get('network_latency')} ms.")
    if high_errors:
        evidence.append(f"Error count is elevated at {incident.get('error_count')}.")
    if not evidence:
        evidence.append("Metrics are only mildly elevated compared to normal baselines.")
    evidence.append(
        f"A historical incident with {similarity_percent}% similarity in error "
        f"message was found in the {predicted_category} category."
    )

    probable_issue = (
        f"Probable issue relates to the {predicted_category} component, "
        f"consistent with a previously seen '{similar_resolution[:60]}...' style incident "
        if len(similar_resolution) > 60 else
        f"Probable issue relates to the {predicted_category} component, "
        f"consistent with a previously seen incident resolved by: {similar_resolution}"
    )

    investigation_steps = [
        f"Review recent logs and metrics for the {predicted_category.lower()} component.",
        "Check for recent deployments, config changes, or scaling events around the incident time.",
        "Compare current metrics against the similar historical incident to confirm the pattern.",
    ]

    suggested_resolution = [
        f"Consider a similar remediation to the historical incident: {similar_resolution}",
        "Monitor the system closely after applying any fix to confirm metrics return to normal.",
    ]

    return {
        "probable_issue": probable_issue,
        "supporting_evidence": evidence,
        "investigation_steps": investigation_steps,
        "suggested_resolution": suggested_resolution,
        "source": "fallback",
    }


def generate_recommendation(incident, predicted_category, anomaly_status, anomaly_score,
                             similar_incident, similar_resolution, similarity_percent):
    """
    Main entry point. Tries Groq LLM first; falls back to rule-based logic
    if the API key is missing or the call fails for any reason.
    """
    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not api_key:
        return _rule_based_fallback(
            incident, predicted_category, anomaly_status,
            similar_resolution, similarity_percent,
        )

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        prompt = _build_prompt(
            incident, predicted_category, anomaly_status, anomaly_score,
            similar_incident, similar_resolution, similarity_percent,
        )

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are a careful, cautious IT operations assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0.3,
        )

        text = response.choices[0].message.content
        return {
            "raw_text": text,
            "source": "groq",
        }

    except Exception as e:
        # Never crash the app -- fall back to rule-based recommendation.
        fallback = _rule_based_fallback(
            incident, predicted_category, anomaly_status,
            similar_resolution, similarity_percent,
        )
        fallback["error_note"] = f"Groq API call failed, used fallback. ({str(e)[:120]})"
        return fallback
