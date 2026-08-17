from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any


def generate_index_page(output_dir: str | Path) -> Path:
    """Generate a self-contained HTML index page for the demo bundle."""
    output_path = Path(output_dir)
    if not output_path.exists():
        raise FileNotFoundError(f"Benchmark output directory not found: {output_path}")

    summary = _load_summary(output_path)
    figures = _discover_figures(output_path)
    csvs = _discover_csvs(output_path)
    episodes = _discover_episodes(output_path)
    report_exists = (output_path / "research_report.md").exists()
    manifest_exists = (output_path / "demo_bundle_manifest.json").exists()

    html = _build_html(
        bundle_name=output_path.name,
        summary=summary,
        figures=figures,
        csvs=csvs,
        episodes=episodes,
        report_exists=report_exists,
        manifest_exists=manifest_exists,
    )

    index_path = output_path / "index.html"
    index_path.write_text(html, encoding="utf-8")
    return index_path


def _load_summary(output_path: Path) -> dict[str, Any] | None:
    summary_path = output_path / "benchmark_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _discover_figures(output_path: Path) -> list[str]:
    figures_dir = output_path / "figures"
    if not figures_dir.exists():
        return []
    return sorted(
        str(p.relative_to(output_path)).replace("\\", "/")
        for p in figures_dir.glob("*.svg")
    )


def _discover_csvs(output_path: Path) -> list[str]:
    exports_dir = output_path / "exports"
    if not exports_dir.exists():
        return []
    return sorted(
        str(p.relative_to(output_path)).replace("\\", "/")
        for p in exports_dir.glob("*.csv")
    )


def _discover_episodes(output_path: Path) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for path in sorted(output_path.glob("*.json")):
        if path.name in ("benchmark_summary.json", "demo_bundle_manifest.json"):
            continue
        if path.name.endswith("_aggregate.json"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            episodes.append(
                {
                    "filename": path.name,
                    "model_name": data.get("model_name", "unknown"),
                    "stress_level": float(data.get("stress_level", 0)),
                    "gdi": float(data.get("metrics", {}).get("graceful_degradation_index", 0)),
                    "collision_rate": float(data.get("metrics", {}).get("collision_rate", 0)),
                }
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return episodes


def _build_html(
    *,
    bundle_name: str,
    summary: dict[str, Any] | None,
    figures: list[str],
    csvs: list[str],
    episodes: list[dict[str, Any]],
    report_exists: bool,
    manifest_exists: bool,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Build leaderboard rows
    leaderboard_html = ""
    if summary and summary.get("leaderboard"):
        rows = []
        for rank, row in enumerate(summary["leaderboard"], start=1):
            medal = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, str(rank))
            gdi = float(row.get("mean_gdi", 0))
            slope = float(row.get("degradation_slope", 0))
            collision = float(row.get("mean_collision_rate", 0))
            
            # Extract neural metrics (they might need to be added to leaderboard logic in summary.py, but for now we fetch if available)
            # Actually summary.py doesn't put mean_adaptation in the leaderboard root. We'll leave leaderboard as is, or we can fetch it from the model curve.
            # Let's get it from the model's aggregate data if we can, wait we don't have model aggregates in the leaderboard row.
            # I will just update the visual style of the row.

            # Color-code GDI
            if gdi >= 0.8:
                gdi_class = "metric-good"
            elif gdi >= 0.6:
                gdi_class = "metric-ok"
            else:
                gdi_class = "metric-warn"

            rows.append(
                f'<tr class="animated-row" style="animation-delay: {rank * 50}ms"><td class="rank">{medal}</td>'
                f'<td class="model-name">{escape(row.get("model_name", ""))}</td>'
                f'<td class="{gdi_class}">{gdi:.4f}</td>'
                f"<td>{slope:+.4f}</td>"
                f"<td>{collision:.4f}</td></tr>"
            )
        leaderboard_html = "\n".join(rows)

    # Build figure cards
    figure_cards = ""
    for fig in figures:
        title = Path(fig).stem.replace("_", " ").title()
        figure_cards += f"""
        <div class="figure-card">
            <h4>{escape(title)}</h4>
            <a href="{escape(fig)}" target="_blank">
                <img src="{escape(fig)}" alt="{escape(title)}" />
            </a>
            <p class="figure-link"><a href="{escape(fig)}" target="_blank">Open full size ↗</a></p>
        </div>"""

    # Build CSV list
    csv_items = ""
    for csv_path in csvs:
        name = Path(csv_path).stem.replace("_", " ").title()
        csv_items += f'<li><a href="{escape(csv_path)}" download>{escape(name)}</a> <span class="badge">CSV</span></li>\n'

    # Build episode table
    episode_rows = ""
    for ep in episodes:
        gdi = ep["gdi"]
        if gdi >= 0.8:
            gdi_class = "metric-good"
        elif gdi >= 0.6:
            gdi_class = "metric-ok"
        else:
            gdi_class = "metric-warn"

        episode_rows += (
            f'<tr><td><a href="{escape(ep["filename"])}" target="_blank">'
            f'{escape(ep["model_name"])}</a></td>'
            f'<td>{ep["stress_level"]:.2f}</td>'
            f'<td class="{gdi_class}">{gdi:.4f}</td>'
            f'<td>{ep["collision_rate"]:.4f}</td></tr>\n'
        )

    # Build quick-links
    quick_links = []
    if report_exists:
        quick_links.append('<a href="research_report.md" class="quick-link">📄 Research Report</a>')
    quick_links.append('<a href="benchmark_summary.json" class="quick-link">📊 Summary JSON</a>')
    if manifest_exists:
        quick_links.append('<a href="demo_bundle_manifest.json" class="quick-link">📦 Bundle Manifest</a>')
    quick_links.append('<a href="DEMO_BUNDLE_README.md" class="quick-link">📖 Bundle README</a>')
    quick_links_html = "\n".join(quick_links)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>NeuroDrive Bench — {escape(bundle_name)}</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Outfit:wght@500;700;800&display=swap" rel="stylesheet">
<style>
:root {{
    --bg-primary: #0a0f18;
    --bg-card: rgba(22, 37, 54, 0.45);
    --bg-card-hover: rgba(28, 48, 72, 0.65);
    --accent-teal: #2dd4bf;
    --accent-blue: #38bdf8;
    --accent-violet: #a78bfa;
    --accent-amber: #fbbf24;
    --accent-rose: #fb7185;
    --text-primary: #f8fafc;
    --text-secondary: #cbd5e1;
    --text-muted: #64748b;
    --border: rgba(30, 58, 95, 0.5);
    --shadow: 0 8px 32px rgba(0,0,0,0.4);
    --radius: 16px;
    --blur: blur(12px);
}}

* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    background: var(--bg-primary);
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(45, 212, 191, 0.05), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(56, 189, 248, 0.05), transparent 25%);
    color: var(--text-primary);
    line-height: 1.6;
    min-height: 100vh;
}}

h1, h2, h3, h4, .model-name, .rank {{
    font-family: 'Outfit', sans-serif;
}}

@keyframes gradientPan {{
    0% {{ background-position: 0% 50%; }}
    50% {{ background-position: 100% 50%; }}
    100% {{ background-position: 0% 50%; }}
}}

.hero {{
    background: linear-gradient(-45deg, #0f172a, #1e293b, #0f172a, #172554);
    background-size: 400% 400%;
    animation: gradientPan 15s ease infinite;
    padding: 5rem 2rem 3.5rem;
    border-bottom: 1px solid var(--border);
    text-align: center;
    position: relative;
    overflow: hidden;
}}

.hero::after {{
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 100px;
    background: linear-gradient(to top, var(--bg-primary), transparent);
}}

.hero h1 {{
    font-size: 3.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #67e8f9, #38bdf8, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.6rem;
    position: relative;
    z-index: 2;
}}

.hero .subtitle {{
    color: var(--text-secondary);
    font-size: 1.15rem;
    font-weight: 400;
    position: relative;
    z-index: 2;
}}

.hero .timestamp {{
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 0.8rem;
    position: relative;
    z-index: 2;
}}

.container {{
    max-width: 1100px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
}}

.quick-links {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
    justify-content: center;
    margin: 2rem 0 2.5rem;
}}

.quick-link {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.4rem;
    background: rgba(56, 189, 248, 0.05);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 100px;
    color: var(--accent-blue);
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

.quick-link:hover {{
    background: rgba(56, 189, 248, 0.15);
    border-color: rgba(56, 189, 248, 0.4);
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 8px 20px rgba(56, 189, 248, 0.2);
    color: #fff;
}}

.section {{
    margin-bottom: 2.5rem;
}}

.section h2 {{
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

.section h2 .icon {{
    font-size: 1.4rem;
}}

/* Leaderboard table */
.leaderboard-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: var(--bg-card);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border-radius: var(--radius);
    border: 1px solid var(--border);
    overflow: hidden;
    box-shadow: var(--shadow);
}}

.leaderboard-table thead {{
    background: rgba(0,0,0,0.2);
}}

.leaderboard-table th {{
    padding: 0.85rem 1rem;
    text-align: left;
    font-size: 0.78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
}}

.leaderboard-table td {{
    padding: 0.75rem 1rem;
    border-top: 1px solid var(--border);
    font-size: 0.92rem;
    font-variant-numeric: tabular-nums;
}}

.leaderboard-table tbody tr {{
    transition: all 0.2s ease;
}}

.leaderboard-table tbody tr:hover {{
    background: var(--bg-card-hover);
    transform: scale(1.005);
}}

@keyframes fadeInUp {{
    from {{ opacity: 0; transform: translateY(10px); }}
    to {{ opacity: 1; transform: translateY(0); }}
}}

.animated-row {{
    animation: fadeInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1) both;
}}

.rank {{ font-size: 1.3rem; text-align: center; }}

.model-name {{
    font-weight: 600;
    color: var(--accent-teal);
}}

.metric-good {{ color: var(--accent-teal); font-weight: 600; }}
.metric-ok {{ color: var(--accent-amber); font-weight: 600; }}
.metric-warn {{ color: var(--accent-rose); font-weight: 600; }}

/* Figures */
.figures-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 1.2rem;
}}

.figure-card {{
    background: var(--bg-card);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

.figure-card:hover {{
    border-color: var(--accent-blue);
    box-shadow: 0 12px 32px rgba(56, 189, 248, 0.15);
    transform: translateY(-4px);
}}

.figure-card h4 {{
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.8rem;
}}

.figure-card img {{
    width: 100%;
    border-radius: 8px;
    background: #f8f8f4;
    display: block;
}}

.figure-link {{
    margin-top: 0.6rem;
    text-align: right;
}}

.figure-link a {{
    color: var(--accent-blue);
    text-decoration: none;
    font-size: 0.82rem;
    font-weight: 500;
}}

.figure-link a:hover {{ text-decoration: underline; }}

/* Data exports */
.exports-list {{
    list-style: none;
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
}}

.exports-list li {{
    background: var(--bg-card);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.5rem 1rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}}

.exports-list li:hover {{
    border-color: var(--accent-teal);
    background: var(--bg-card-hover);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(45, 212, 191, 0.15);
}}

.exports-list a {{
    color: var(--accent-teal);
    text-decoration: none;
    font-weight: 500;
    font-size: 0.9rem;
}}

.exports-list a:hover {{ text-decoration: underline; }}

.badge {{
    display: inline-block;
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 0.15rem 0.45rem;
    border-radius: 4px;
    background: rgba(45, 212, 191, 0.15);
    color: var(--accent-teal);
    vertical-align: middle;
    margin-left: 0.3rem;
}}

/* Episode table */
.episode-table {{
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    background: var(--bg-card);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
    font-size: 0.88rem;
}}

.episode-table thead {{
    background: rgba(0,0,0,0.2);
}}

.episode-table th {{
    padding: 0.7rem 0.9rem;
    text-align: left;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-secondary);
}}

.episode-table td {{
    padding: 0.55rem 0.9rem;
    border-top: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
}}

.episode-table a {{
    color: var(--accent-teal);
    text-decoration: none;
    font-weight: 500;
}}

.episode-table a:hover {{ text-decoration: underline; }}

.episode-table tbody tr:hover {{
    background: var(--bg-card-hover);
}}

/* Footer */
.footer {{
    text-align: center;
    color: var(--text-muted);
    font-size: 0.78rem;
    padding: 2rem 1rem 1rem;
    border-top: 1px solid var(--border);
    margin-top: 2rem;
}}

@media (max-width: 640px) {{
    .hero h1 {{ font-size: 1.8rem; }}
    .figures-grid {{ grid-template-columns: 1fr; }}
    .container {{ padding: 1rem 1rem 3rem; }}
}}
</style>
</head>
<body>
<div class="hero">
    <h1>NeuroDrive Bench</h1>
    <p class="subtitle">Benchmark Output — <strong>{escape(bundle_name)}</strong></p>
    <p class="timestamp">Generated {escape(timestamp)}</p>
</div>

<div class="container">
    <div class="quick-links">
        {quick_links_html}
    </div>

    {"" if not leaderboard_html else f'''
    <div class="section">
        <h2><span class="icon">🏆</span> Leaderboard</h2>
        <table class="leaderboard-table">
            <thead>
                <tr>
                    <th style="text-align:center">Rank</th>
                    <th>Model</th>
                    <th>Mean GDI</th>
                    <th>Degradation Slope</th>
                    <th>Mean Collision Rate</th>
                </tr>
            </thead>
            <tbody>
                {leaderboard_html}
            </tbody>
        </table>
    </div>
    '''}

    {"" if not figure_cards else f'''
    <div class="section">
        <h2><span class="icon">📈</span> Figures</h2>
        <div class="figures-grid">
            {figure_cards}
        </div>
    </div>
    '''}

    {"" if not csv_items else f'''
    <div class="section">
        <h2><span class="icon">📁</span> Data Exports</h2>
        <ul class="exports-list">
            {csv_items}
        </ul>
    </div>
    '''}

    {"" if not episode_rows else f'''
    <div class="section">
        <h2><span class="icon">🧪</span> Episode Artifacts ({len(episodes)})</h2>
        <table class="episode-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Stress Level</th>
                    <th>GDI</th>
                    <th>Collision Rate</th>
                </tr>
            </thead>
            <tbody>
                {episode_rows}
            </tbody>
        </table>
    </div>
    '''}

    <div class="footer">
        NeuroDrive Bench · Robustness-first autonomous driving benchmarking
    </div>
</div>
</body>
</html>"""
