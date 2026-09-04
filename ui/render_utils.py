import re

import pandas as pd
import plotly.express as px
import streamlit as st

# Brand colors, used for the surrounding UI chrome (KPI cards, alerts,
# section accents) - not for chart data itself.
BRAND_RED = "#E50914"
BRAND_RED_LIGHT = "#FF4757"
BRAND_RED_DEEP = "#B71C1C"

# Chart data uses a general-purpose palette rather than the brand colors: a
# chart is read by its category colors, and locking every bar/slice to
# red-black-white made multi-category charts (pie slices especially) hard to
# tell apart. Two vivid qualitative sets are chained so charts with more
# categories than one palette holds still get distinct colors instead of
# wrapping around and repeating. The dark background/font still match the page.
GENERAL_SEQUENCE = list(px.colors.qualitative.Plotly) + [
    c for c in px.colors.qualitative.Bold if c not in set(px.colors.qualitative.Plotly)
]

_NAMED_COLOURS = {
    "red": "#EF553B",
    "blue": "#636EFA",
    "green": "#00CC96",
    "yellow": "#FECB52",
    "orange": "#FFA15A",
    "purple": "#AB63FA",
    "pink": "#FF6692",
    "brown": "#8C564B",
    "gray": "#7F7F7F",
    "grey": "#7F7F7F",
    "black": "#111111",
    "white": "#F5F5F5",
    "teal": "#19D3F3",
    "cyan": "#19D3F3",
    "magenta": "#FF00FF",
    "gold": "#FFD700",
    "navy": "#001F3F",
    "maroon": "#800000",
    "olive": "#808000",
    "lime": "#32CD32",
    "indigo": "#4B0082",
    "violet": "#8F00FF",
    "turquoise": "#40E0D0",
    "coral": "#FF7F50",
}

CHART_THEME = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#CCCCCC", "family": "Inter, sans-serif"},
    "margin": {"t": 50, "b": 40, "l": 40, "r": 20},
    "colorway": GENERAL_SEQUENCE,
}


def _extract_question(reasoning):
    if not reasoning:
        return ""
    match = re.search(r'\*\*Interpreted question:\*\*\s+"([^"]+)"', reasoning)
    return match.group(1) if match else ""


def _palette_from_question(question):
    if not question:
        return GENERAL_SEQUENCE

    lowered = question.lower()
    if any(
        phrase in lowered
        for phrase in ("multicolor", "multi color", "multi-color", "colorful",
                       "colourful", "general colors", "general colours",
                       "different colors", "different colours")
    ):
        return GENERAL_SEQUENCE

    picked = []
    for hex_code in re.findall(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b", question):
        if hex_code not in picked:
            picked.append(hex_code)

    for name, hex_code in _NAMED_COLOURS.items():
        if re.search(rf"\b{name}\b", lowered) and hex_code not in picked:
            picked.append(hex_code)

    if not picked:
        return GENERAL_SEQUENCE

    return picked + [c for c in GENERAL_SEQUENCE if c not in set(picked)]


def _apply_single_trace_colour(fig, palette):
    first = palette[0] if palette else GENERAL_SEQUENCE[0]
    fig.update_traces(marker_color=first, line_color=first)


def _colours_for_rows(row_count, palette):
    """Return an explicit colour for each category, independent of any theme."""
    active_palette = palette or GENERAL_SEQUENCE
    return [active_palette[index % len(active_palette)] for index in range(row_count)]


def _categorical_columns(df):
    """Columns that are not numeric.

    Deliberately not `select_dtypes(include=['object'])`: pandas 3 gives text
    columns the `str` dtype, which that filter misses entirely.
    """
    numeric = set(df.select_dtypes("number").columns)
    return [c for c in df.columns if c not in numeric]


def _kpi_card(label, value, footnote, accent):
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {accent}1a 0%, {accent}0d 100%);
                padding: 15px; border-radius: 12px; border: 1px solid {accent}4d;
                border-left: 5px solid {accent}; text-align: center;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 100%;'>
        <h4 style='color: #CCCCCC; font-size: 0.85rem; margin: 0; text-transform: uppercase;
                   letter-spacing: 1px;'>{label}</h4>
        <p style='color: #FFFFFF; font-size: 1.8rem; font-weight: 800; margin: 5px 0;'>{value}</p>
        <p style='color: {accent}; font-size: 0.78rem; font-weight: 600; margin: 0;'>{footnote}</p>
    </div>
    """, unsafe_allow_html=True)


def render_kpis(kpis):
    """Three cards, every figure derived from the returned rows."""
    c1, c2, c3 = st.columns(3)

    label = kpis.get("primary_label", "Result")
    value = kpis.get("primary_val", 0)
    value_str = f"${value:,.2f}" if kpis.get("is_currency") else f"{value:,.2f}".rstrip("0").rstrip(".")

    with c1:
        mean = kpis.get("mean_val")
        note = f"avg {mean:,.2f} per row" if mean is not None else "computed from result"
        _kpi_card(label, value_str, note, "#FFFFFF")

    with c2:
        _kpi_card("Rows Returned", f"{kpis.get('row_count', 0):,}",
                  f"{kpis.get('column_count', 0)} columns", "#E50914")

    with c3:
        # Shown only when the result actually has an ordered period column.
        if "trend_pct" in kpis:
            pct = kpis["trend_pct"]
            arrow = "▲" if pct >= 0 else "▼"
            colour = "#FFFFFF" if pct >= 0 else "#E50914"
            _kpi_card("Change Over Period", f"{pct:+.1f}%",
                      f"{arrow} {kpis.get('trend_label', '')}", colour)
        else:
            _kpi_card("Change Over Period", "N/A",
                      "no time column in result", "#999999")


def render_chart(df, hint, metric, turn_index=0, question_text=""):
    if df.empty:
        st.info("The query returned no rows.")
        return

    if hint == "table":
        st.dataframe(df, width='stretch')
        return

    categorical = _categorical_columns(df)
    numeric = df.select_dtypes("number").columns.tolist()
    palette = _palette_from_question(question_text)

    try:
        if hint == "single_value" and metric:
            val = df[metric].iloc[0]
            pretty = metric.replace("_", " ").title()
            shown = f"{val:,.2f}" if isinstance(val, (int, float)) else str(val)
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, rgba(229, 9, 20,0.15) 0%, rgba(229, 9, 20,0.05) 100%);
                        padding: 45px; border-radius: 15px; border: 1px solid rgba(229, 9, 20,0.2);
                        text-align: center; box-shadow: 0 10px 30px rgba(0,0,0,0.2);'>
                <h2 style='color: #E50914; font-size: 1.1rem; font-weight: 600; margin: 0;
                           text-transform: uppercase; letter-spacing: 2px;'>{pretty}</h2>
                <p style='color: #FFFFFF; font-size: 3.4rem; font-weight: 800; margin: 15px 0;'>{shown}</p>
            </div>
            """, unsafe_allow_html=True)
            return

        if hint == "line" and metric and categorical:
            fig = px.line(df, x=categorical[0], y=metric, markers=True,
                          color_discrete_sequence=palette)
            _apply_single_trace_colour(fig, palette)
            fig.update_layout(title=f"{metric.replace('_', ' ').title()} over {categorical[0].replace('_', ' ').title()}")

        elif hint == "area" and metric and categorical:
            fig = px.area(df, x=categorical[0], y=metric, color_discrete_sequence=palette)
            _apply_single_trace_colour(fig, palette)
            fig.update_layout(title=f"{metric.replace('_', ' ').title()} over {categorical[0].replace('_', ' ').title()}")

        elif hint in ("pie", "donut") and metric and categorical:
            # A donut is only drawn when it was actually asked for; "pie chart"
            # renders as a real pie.
            fig = px.pie(df, values=metric, names=categorical[0],
                         hole=0.5 if hint == "donut" else 0,
                         color_discrete_sequence=palette)
            fig.update_traces(textposition="inside", textinfo="percent+label",
                              marker=dict(
                                  colors=_colours_for_rows(len(df), palette),
                                  line=dict(color="#141414", width=2),
                              ))
            fig.update_layout(title=f"{metric.replace('_', ' ').title()} by {categorical[0].replace('_', ' ').title()}")

        elif hint == "bar" and metric and categorical:
            plot_df = df.nlargest(30, metric) if len(df) > 30 else df
            # Coloured by category, so each bar gets its own colour from the
            # qualitative palette. Colouring by the metric instead would give a
            # single-hue gradient, which reads as one flat colour.
            fig = px.bar(plot_df, x=categorical[0], y=metric, color=categorical[0],
                         color_discrete_sequence=palette)
            title = f"{metric.replace('_', ' ').title()} by {categorical[0].replace('_', ' ').title()}"
            if len(df) > 30:
                title += " (top 30)"
            # The x-axis already names every bar; the legend would just repeat it.
            fig.update_layout(title=title, showlegend=False)

        elif hint == "scatter" and len(numeric) >= 2:
            fig = px.scatter(df, x=numeric[0], y=numeric[1],
                             color_discrete_sequence=palette)
            _apply_single_trace_colour(fig, palette)
            fig.update_layout(title=f"{numeric[1].replace('_', ' ').title()} vs {numeric[0].replace('_', ' ').title()}")

        elif hint == "histogram" and metric:
            fig = px.histogram(df, x=metric, nbins=30,
                               color_discrete_sequence=palette)
            _apply_single_trace_colour(fig, palette)
            fig.update_layout(title=f"Distribution of {metric.replace('_', ' ').title()}")

        elif hint == "frequency" and categorical:
            counts = df[categorical[0]].value_counts().head(30).reset_index()
            counts.columns = [categorical[0], "count"]
            fig = px.bar(counts, x="count", y=categorical[0], orientation="h",
                         color=categorical[0], color_discrete_sequence=palette,
                         text_auto=True)
            fig.update_layout(title=f"Frequency: {categorical[0].replace('_', ' ').title()}",
                              showlegend=False)

        elif metric and categorical:
            plot_df = df.head(30)
            fig = px.bar(plot_df, x=categorical[0], y=metric, color=categorical[0],
                         color_discrete_sequence=palette)
            fig.update_layout(showlegend=False)

        elif len(numeric) >= 2:
            fig = px.scatter(df, x=numeric[0], y=numeric[1],
                             color_discrete_sequence=palette)
            _apply_single_trace_colour(fig, palette)
            fig.update_layout(title=f"{numeric[1].title()} vs {numeric[0].title()}")

        else:
            st.dataframe(df, width='stretch')
            return

        fig.update_layout(**CHART_THEME)
        # theme=None: Streamlit's default chart theme re-colours figures from
        # .streamlit/config.toml (primaryColor), which would repaint every
        # chart in the brand red. The figure already carries its own dark
        # styling via CHART_THEME, so opting out keeps the palette set here.
        st.plotly_chart(fig, width='stretch', theme=None,
                        key=f"chart_{hint}_{turn_index}")

    except Exception as exc:
        st.warning(f"Could not render a chart for this result: {exc}")
        st.dataframe(df, width='stretch')


_SEVERITY_COLOURS = {"high": "#E50914", "medium": "#FF4757", "low": "#999999"}


def _fmt(value, currency=False):
    if currency:
        return f"${value:,.2f}"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def render_overview(overview, on_question=None):
    """The automatic briefing shown as soon as a file is uploaded."""
    sheets = overview.get("sheets") or []
    if not sheets:
        return

    primary = next(
        (s for s in sheets if s["name"] == overview.get("primary_sheet")), sheets[0]
    )

    st.markdown("""
    <h4 style='color: #E50914; margin: 25px 0 5px 0; font-weight: 700; font-size: 1.3rem;'>
    📊 What's in your data
    </h4>
    <hr style='border: 1px solid #E50914; background-color: #E50914; opacity: 1; margin: 0 0 15px 0;'>
    """, unsafe_allow_html=True)

    # --- narrative -------------------------------------------------------
    if overview.get("summary"):
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, rgba(229, 9, 20, 0.12) 0%, rgba(229, 9, 20, 0.04) 100%);
                    padding: 20px 24px; border-radius: 12px; border-left: 5px solid #E50914;
                    margin-bottom: 20px;'>
            <p style='color: #E5E5E5; margin: 0; line-height: 1.7; font-size: 1rem;'>{overview['summary']}</p>
        </div>
        """, unsafe_allow_html=True)

    # --- shape + headline metrics ---------------------------------------
    cards = [("📄 Rows", f"{overview.get('total_rows', 0):,}",
              f"{overview.get('sheet_count', 1)} sheet(s)", "#E50914")]

    coverage = primary.get("time_coverage")
    if coverage:
        cards.append(("📅 Time Span", f"{coverage['distinct_months']} months",
                      f"{coverage['from']} → {coverage['to']}", "#FF4757"))

    for metric in primary.get("headline_metrics", [])[:2]:
        cards.append((
            f"💰 Total {metric['label']}" if metric["is_currency"] else f"Σ {metric['label']}",
            _fmt(metric["total"], metric["is_currency"]),
            f"avg {_fmt(metric['mean'], metric['is_currency'])}",
            "#FFFFFF",
        ))

    for row_start in range(0, len(cards), 4):
        columns = st.columns(min(4, len(cards) - row_start))
        for col, (label, value, note, accent) in zip(columns, cards[row_start:row_start + 4]):
            with col:
                _kpi_card(label, value, note, accent)

    st.write("")

    # --- breakdowns + quality -------------------------------------------
    left, right = st.columns(2)

    with left:
        breakdowns = primary.get("breakdowns") or []
        if breakdowns:
            breakdown = breakdowns[0]
            st.markdown(f"###### 🏆 Top {breakdown['category'].replace('_', ' ')} "
                        f"by {breakdown['metric'].replace('_', ' ')}")
            frame = pd.DataFrame([
                {breakdown["category"]: r["name"],
                 breakdown["metric"]: r["value"],
                 "share": f"{r['share']}%"}
                for r in breakdown["rows"]
            ])
            fig = px.bar(frame, x=breakdown["category"], y=breakdown["metric"],
                         color=breakdown["category"],
                         color_discrete_sequence=GENERAL_SEQUENCE, text="share")
            fig.update_layout(**CHART_THEME, showlegend=False, height=300)
            st.plotly_chart(fig, width='stretch', theme=None, key="overview_breakdown")
            st.caption(f"{breakdown['distinct']} distinct value(s) in "
                       f"`{breakdown['category']}`.")

        for pair in (primary.get("correlations") or [])[:3]:
            st.markdown(
                f"🔗 `{pair['a']}` and `{pair['b']}` move **{pair['direction']}** "
                f"(r = {pair['r']}) — an association, not a cause."
            )

    with right:
        quality = primary.get("quality") or []
        st.markdown("###### 🧹 Data quality")
        if not quality:
            st.success("No structural problems found in this sheet.")
        else:
            for finding in quality[:6]:
                colour = _SEVERITY_COLOURS.get(finding["severity"], "#999999")
                st.markdown(
                    f"<div style='border-left: 3px solid {colour}; padding: 4px 10px; "
                    f"margin-bottom: 7px; background: rgba(255, 255, 255,0.03);'>"
                    f"<span style='color:{colour};font-weight:700;font-size:0.72rem;"
                    f"text-transform:uppercase;'>{finding['severity']}</span><br>"
                    f"<span style='color:#CCCCCC;font-size:0.88rem;'>{finding['issue']}</span></div>",
                    unsafe_allow_html=True,
                )
            st.caption("Worth checking before you trust any aggregate.")

        # Say so when the checks above read a sample rather than every row, so
        # an estimated share is never mistaken for an exact count.
        if primary.get("sampled"):
            st.caption(
                f"Quality checks and column statistics were profiled from a random "
                f"{primary['profiled_rows']:,}-row sample of {overview.get('total_rows', 0):,} rows. "
                f"Totals, breakdowns and the date range above are computed from every row."
            )

    # --- column roles ----------------------------------------------------
    with st.expander("🧬 How each column was read", expanded=False):
        for sheet in sheets:
            st.markdown(f"**`{sheet['name']}`** — {sheet['rows']:,} rows")
            roles = sheet["roles"]
            role_labels = {
                "numeric": "📊 Measures", "categorical": "🏷️ Categories",
                "temporal": "📅 Dates", "identifier": "🔑 Keys / free text",
            }
            for role, label in role_labels.items():
                if roles.get(role):
                    st.markdown(f"- {label}: " + ", ".join(f"`{c}`" for c in roles[role]))
            st.write("")
        st.caption(
            "Measures are what gets aggregated; categories are what it gets grouped by. "
            "Keys are excluded from metrics so an ID is never summed."
        )

    # --- questions grounded in the real columns --------------------------
    questions = primary.get("suggested_questions") or []
    if questions and on_question:
        st.markdown("###### 💬 Ask about this data")
        for row_start in range(0, min(len(questions), 6), 3):
            columns = st.columns(3)
            for col, question in zip(columns, questions[row_start:row_start + 3]):
                col.button(question, width='stretch',
                           key=f"ov_q_{question}", on_click=on_question, args=(question,))


def render_insight(text):
    if not text:
        return
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, rgba(255, 71, 87, 0.12) 0%, rgba(255, 71, 87, 0.04) 100%);
                padding: 18px 22px; border-radius: 12px; border-left: 5px solid #FF4757;
                margin-bottom: 18px;'>
        <p style='color: #FF4757; font-weight: 700; margin: 0 0 8px 0; font-size: 0.8rem;
                  text-transform: uppercase; letter-spacing: 1px;'>🧠 Analyst Insight</p>
        <p style='color: #E5E5E5; margin: 0; line-height: 1.65; font-size: 0.97rem;'>{text}</p>
    </div>
    """, unsafe_allow_html=True)


def render_provenance(data):
    """What the agents retrieved and how they got here."""
    definitions = data.get("definitions_used") or []
    examples = data.get("examples_used") or []
    plan = data.get("plan") or []

    if not (definitions or examples or plan):
        return

    with st.expander("🔗 Retrieved context & plan", expanded=False):
        if plan:
            st.markdown("**Plan followed**")
            for i, step in enumerate(plan, 1):
                st.markdown(f"{i}. {step}")
            for step in data.get("step_results") or []:
                with st.expander(f"code for: {step['step']}", expanded=False):
                    st.code(step["code"], language="python")
            st.divider()

        if definitions:
            st.markdown("**Certified definitions applied** (from your glossary)")
            for item in definitions:
                how = item.get("matched", "semantic")
                st.markdown(f"- **{item['term']}** — {item['definition']}  \n"
                            f"  <span style='color:#999999;font-size:0.8rem;'>matched: {how}</span>",
                            unsafe_allow_html=True)
            st.divider()

        if examples:
            st.markdown("**Similar past analyses used as examples**")
            for item in examples:
                origin = "this workbook" if item["same_file"] else "another workbook"
                st.markdown(f"- _{item['question']}_ "
                            f"<span style='color:#999999;font-size:0.8rem;'>"
                            f"(similarity {item['score']}, {origin})</span>",
                            unsafe_allow_html=True)


def render_data_results(data, turn_index=0):
    """Render one answer: KPIs, chart, table, and the code that produced it."""
    if data.get("status") == "needs_clarification":
        # The question was ambiguous; the agents asked instead of guessing.
        st.markdown(data.get("reasoning", ""))
        return

    if data.get("status") == "failed":
        st.error("The agents could not complete this analysis.")
        st.markdown(data.get("reasoning", ""))
        if data.get("code"):
            with st.expander("🔍 See the code that failed"):
                st.code(data["code"], language="python")
        return

    if data.get("warning"):
        st.warning(f"**Worth a second look:** {data['warning']}")

    render_insight(data.get("insight", ""))
    render_kpis(data.get("kpis", {}))

    st.markdown("<hr style='border: 1px solid #E50914; opacity: 1; margin: 25px 0;'>",
                unsafe_allow_html=True)

    df = pd.DataFrame(data.get("data", []))

    left, right = st.columns([2, 1])
    with left:
        st.markdown("###### 📊 Visualization")
        if data.get("chart_note"):
            st.caption(f"ℹ️ {data['chart_note'].capitalize()}.")
        elif data.get("requested_chart"):
            st.caption(f"✅ Shown as a {data['requested_chart']} chart, as you asked.")
        render_chart(
            df,
            data.get("chart_hint", "table"),
            data.get("metric_column"),
            turn_index,
            _extract_question(data.get("reasoning", "")),
        )
    with right:
        st.markdown("###### 📋 Result Table")
        st.dataframe(df, width='stretch', height=420)

    render_provenance(data)

    with st.expander("🔍 See Generated Code & Reasoning"):
        st.markdown(data.get("reasoning", ""))
        st.markdown("---")
        st.markdown("**Pandas code generated by the Analysis Agent:**")
        st.code(data.get("code", "# no code"), language="python")

        if not df.empty:
            st.divider()
            d1, d2 = st.columns(2)
            stamp = pd.Timestamp.now().strftime("%H%M%S")
            with d1:
                st.download_button(
                    "📥 Download Result (CSV)",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name=f"agentic_bi_result_{stamp}.csv",
                    mime="text/csv", width='stretch',
                    key=f"dl_csv_{turn_index}",
                )
            with d2:
                st.download_button(
                    "🐍 Download Code (.py)",
                    data.get("code", ""),
                    file_name=f"analysis_{stamp}.py",
                    mime="text/plain", width='stretch',
                    key=f"dl_code_{turn_index}",
                )
