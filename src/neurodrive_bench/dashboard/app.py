from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from neurodrive_bench.dashboard.data import (
    available_models,
    build_metric_rows,
    filter_artifacts,
    load_artifacts,
    load_summary,
)

try:
    import streamlit as st
except ImportError:  # pragma: no cover
    st = None


DEFAULT_OUTPUT_DIR = "outputs/baseline_robustness_suite"


def dashboard_stub_text() -> str:
    return "\n".join(
        [
            "NeuroDrive Debug Console",
            "- Benchmark Summary",
            "- Graceful Degradation View",
            "- Metrics Table",
            "- Trace Explorer",
            "- Event Log Viewer",
        ]
    )


def launch_dashboard() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install the dashboard extras with: pip install -e .[dashboard]"
        )

    st.set_page_config(
        page_title="NeuroDrive Debug Console",
        page_icon="ND",
        layout="wide",
    )

    _inject_styles()
    st.title("NeuroDrive Debug Console")
    st.caption("Robustness-first artifact explorer for NeuroDrive Bench")

    with st.sidebar:
        st.header("Data Source")
        output_dir = st.text_input("Artifact directory", value=_default_output_dir())
        refresh = st.button("Reload Artifacts", use_container_width=True)

    if refresh:
        st.cache_data.clear()

    artifacts = _load_artifacts_cached(output_dir)
    summary = _load_summary_cached(output_dir)
    if not artifacts:
        st.warning(f"No benchmark artifacts were found in `{output_dir}`.")
        st.code("$env:PYTHONPATH='src'; python -m neurodrive_bench.cli run --config configs/benchmark.example.yaml")
        return

    models = available_models(artifacts)

    with st.sidebar:
        st.header("Filters")
        selected_model = st.selectbox("Model", options=["All"] + models)
        selected_artifacts = artifacts if selected_model == "All" else filter_artifacts(artifacts, selected_model)
        artifact_labels = [
            f"{item['model_name']} | stress={float(item['stress_level']):.2f}" for item in selected_artifacts
        ]
        selected_label = st.selectbox("Episode", options=artifact_labels)

    selected_artifact = selected_artifacts[artifact_labels.index(selected_label)]

    _render_summary(selected_artifacts, summary)
    _render_degradation_view(selected_artifacts)
    _render_episode_details(selected_artifact)
    _render_trace_explorer(selected_artifact)


def _load_artifacts_cached(output_dir: str) -> list[dict[str, Any]]:
    if st is None:
        return load_artifacts(output_dir)
    loader = st.cache_data(show_spinner=False)(load_artifacts)
    return loader(output_dir)


def _load_summary_cached(output_dir: str) -> dict[str, Any] | None:
    if st is None:
        return load_summary(output_dir)
    loader = st.cache_data(show_spinner=False)(load_summary)
    return loader(output_dir)


def _render_summary(artifacts: list[dict[str, Any]], summary: dict[str, Any] | None) -> None:
    rows = build_metric_rows(artifacts)
    best_gdi = max(float(row["graceful_degradation_index"]) for row in rows)
    worst_collision = max(float(row["collision_rate"]) for row in rows)
    avg_latency = sum(float(row.get("adaptation_latency", 0.0)) for row in rows) / len(rows)
    avg_uncertainty = sum(float(row.get("mean_uncertainty", 0.0)) for row in rows) / len(rows)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Best GDI", f"{best_gdi:.3f}")
    col2.metric("Worst Collision Rate", f"{worst_collision:.3f}")
    col3.metric("Avg Adapt Latency", f"{avg_latency:.3f}")
    col4.metric("Avg Uncertainty", f"{avg_uncertainty:.3f}")

    if summary and summary.get("leaderboard"):
        with st.expander("Benchmark Leaderboard", expanded=True):
            st.dataframe(summary["leaderboard"], use_container_width=True)


def _render_degradation_view(artifacts: list[dict[str, Any]]) -> None:
    st.subheader("Graceful Degradation View")
    rows = build_metric_rows(artifacts)
    
    col_chart, col_metric = st.columns([3, 1])
    with col_metric:
        y_axis = st.selectbox(
            "Y-Axis Metric", 
            options=["graceful_degradation_index", "collision_rate", "mean_adaptation", "mean_uncertainty", "adaptation_latency"],
            index=0
        )
    
    plot_rows = [
        {
            "model": row["model_name"],
            "stress_level": row["stress_level"],
            "graceful_degradation_index": row.get("graceful_degradation_index", 0.0),
            "collision_rate": row.get("collision_rate", 0.0),
            "mean_adaptation": row.get("mean_adaptation", 0.0),
            "mean_uncertainty": row.get("mean_uncertainty", 0.0),
            "adaptation_latency": row.get("adaptation_latency", 0.0),
        }
        for row in rows
    ]
    with col_chart:
        st.line_chart(plot_rows, x="stress_level", y=y_axis, color="model")

    with st.expander("Benchmark Metrics Table", expanded=False):
        st.dataframe(rows, use_container_width=True)


def _render_episode_details(artifact: dict[str, Any]) -> None:
    st.subheader("Episode Details")
    metrics = artifact.get("metrics", {})
    stress = artifact.get("stress_profile", {})

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Metrics**")
        st.json(metrics)
    with right:
        st.markdown("**Stress Profile**")
        st.json(stress)


def _render_trace_explorer(artifact: dict[str, Any]) -> None:
    st.subheader("Trace Explorer")
    episode = artifact.get("episode", {})
    samples = list(episode.get("samples", []))
    events = list(episode.get("events", []))

    if not samples:
        st.info("This artifact does not include trace samples yet.")
        return

    trace_rows = [
        {
            "step": sample.get("step"),
            "speed": sample.get("telemetry", {}).get("speed"),
            "lane_offset": sample.get("telemetry", {}).get("lane_offset"),
            "heading_error": sample.get("telemetry", {}).get("heading_error"),
            "obstacle_distance": sample.get("telemetry", {}).get("obstacle_distance"),
            "steering": sample.get("control", {}).get("steering"),
            "throttle": sample.get("control", {}).get("throttle"),
            "uncertainty_score": sample.get("uncertainty_score"),
            "adaptation_level": sample.get("adaptation_level"),
        }
        for sample in samples
    ]

    st.line_chart(trace_rows, x="step", y=["speed", "obstacle_distance"])
    st.line_chart(trace_rows, x="step", y=["lane_offset", "heading_error"])
    st.line_chart(trace_rows, x="step", y=["steering", "adaptation_level", "uncertainty_score"])

    trace_col, event_col = st.columns([2, 1])
    with trace_col:
        st.markdown("**Trace Samples**")
        st.dataframe(trace_rows, use_container_width=True)
    with event_col:
        st.markdown("**Event Log**")
        if events:
            # Highlight event steps for easier cross-referencing
            for event in events:
                event_type = event.get("event_type", "unknown")
                color = "red" if "collision" in event_type else "orange"
                st.markdown(f"- **Step {event.get('step')}**: :{color}[{event_type}]")
            st.dataframe(events, use_container_width=True)
        else:
            st.caption("No events were recorded for this episode.")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(56, 189, 248, 0.15), transparent 35%),
                linear-gradient(180deg, #0a0f18 0%, #1e293b 100%);
        }
        .stApp * {
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #172554 100%);
        }
        [data-testid="stSidebar"] * {
            color: #eef4f8;
        }
        h1, h2, h3 {
            letter-spacing: 0.02em;
        }
        /* Fix table and dropdown backgrounds for dark mode */
        div[data-baseweb="select"] > div {
            background-color: #1e293b !important;
            color: white !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _default_output_dir() -> str:
    if "--output-dir" in sys.argv:
        index = sys.argv.index("--output-dir")
        if index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return DEFAULT_OUTPUT_DIR


if __name__ == "__main__":
    launch_dashboard()
