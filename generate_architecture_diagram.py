"""Generates agentic_bi_architecture.jpeg - a clean, compact blueprint of the
real 10-agent LangGraph pipeline (Profile -> Clarify -> RAG -> Planner ->
Analysis -> Impact -> Execute -> Critic -> Insight -> BI).

Laid out as a 4x3 snake (boustrophedon) grid rather than one wide row, so the
image stays close to square and text stays legible once Streamlit scales it
down to fit a column width.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BG = "#0A0A0A"
BOX_FILL = "#FFFFFF"
BOX_EDGE = "#E50914"
BOX_TEXT = "#0A0A0A"
END_FILL = "#E50914"
END_TEXT = "#FFFFFF"
FLOW_COLOR = "#CCCCCC"
LOOP_COLOR = "#FF4757"
TITLE_COLOR = "#FFFFFF"
SUB_COLOR = "#999999"

# --- Grid geometry -----------------------------------------------------
COLS = 3
W, H = 3.7, 1.55          # box size
GAP_X, GAP_Y = 0.85, 1.25  # gaps between boxes
LEFT_MARGIN, RIGHT_MARGIN = 1.5, 1.5
TOP_MARGIN, BOTTOM_MARGIN = 2.1, 4.3

col_x = [LEFT_MARGIN + c * (W + GAP_X) for c in range(COLS)]
grid_width = LEFT_MARGIN + COLS * W + (COLS - 1) * GAP_X + RIGHT_MARGIN
n_rows = 4
row_y = [TOP_MARGIN + r * (H + GAP_Y) for r in range(n_rows)]  # row 0 = top-most, but stored top->bottom below
grid_height = TOP_MARGIN + n_rows * H + (n_rows - 1) * GAP_Y + BOTTOM_MARGIN

fig, ax = plt.subplots(figsize=(grid_width * 0.92, grid_height * 0.92), dpi=155)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.set_xlim(0, grid_width)
ax.set_ylim(0, grid_height)
ax.axis("off")

top = grid_height - TOP_MARGIN
# y position (bottom-left corner) for each row, row index 0 = top row
rows_y = [top - H - r * (H + GAP_Y) for r in range(n_rows)]


def box(col, row, label, sub, fill=BOX_FILL, edge=BOX_EDGE, text=BOX_TEXT):
    x, y = col_x[col], rows_y[row]
    patch = FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.07,rounding_size=0.14",
                            linewidth=2.0, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(patch)
    ax.text(x + W / 2, y + H * 0.64, label, ha="center", va="center",
             fontsize=15.5, fontweight="bold", color=text, zorder=4)
    ax.text(x + W / 2, y + H * 0.27, sub, ha="center", va="center",
             fontsize=11, color=text, zorder=4, style="italic")
    return (x, y, W, H)


def edge_pt(b, side):
    x, y, w, h = b
    return {"right": (x + w, y + h / 2), "left": (x, y + h / 2),
            "top": (x + w / 2, y + h), "bottom": (x + w / 2, y)}[side]


def straight(b1, b2, side1, side2, color=FLOW_COLOR, lw=2.4):
    p1, p2 = edge_pt(b1, side1), edge_pt(b2, side2)
    ax.annotate("", xy=p2, xytext=p1,
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=2, shrinkB=2))


def polyline(points, color, lw=2.2, ls="-", label=None, label_pos=None, fontsize=10.5):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, solid_capstyle="round", zorder=2)
    ax.annotate("", xy=points[-1], xytext=points[-2],
                 arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0))
    if label:
        lx, ly = label_pos if label_pos else points[len(points) // 2]
        ax.text(lx, ly, label, ha="center", va="center", fontsize=fontsize,
                 color=color, style="italic", zorder=5,
                 bbox=dict(boxstyle="round,pad=0.25", fc=BG, ec="none"))


# --- Title ---------------------------------------------------------------
ax.text(grid_width / 2, grid_height - 1.0, "MegAI — Agentic Data Analysis Pipeline",
        ha="center", fontsize=25, fontweight="bold", color=TITLE_COLOR, family="sans-serif")
ax.text(grid_width / 2, grid_height - 1.65,
        "Upload a spreadsheet, ask a question in plain English, get a verified, RAG-grounded answer",
        ha="center", fontsize=12.5, color=SUB_COLOR, family="sans-serif")

# --- Row 0 (L -> R): Upload, Profile, Clarify ---
b_upload = box(0, 0, "Upload", "Excel / CSV / ZIP")
b_profile = box(1, 0, "Profile Agent", "Profiles every sheet")
b_clarify = box(2, 0, "Clarify Agent", "Asks back if ambiguous")
straight(b_upload, b_profile, "right", "left")
straight(b_profile, b_clarify, "right", "left")

# --- Row 1 (R -> L): RAG, Planner, Analysis ---
b_rag = box(2, 1, "RAG Agent", "Glossary + Query Memory")
b_planner = box(1, 1, "Planner Agent", "Splits into steps")
b_analysis = box(0, 1, "Analysis Agent", "Writes pandas code")
straight(b_clarify, b_rag, "bottom", "top")
straight(b_rag, b_planner, "left", "right")
straight(b_planner, b_analysis, "left", "right")

# --- Row 2 (L -> R): Impact, Execute, Critic ---
b_impact = box(0, 2, "Impact Agent", "AST safety audit")
b_execute = box(1, 2, "Execute Agent", "Sandboxed run")
b_critic = box(2, 2, "Critic Agent", "Verifies the answer")
straight(b_analysis, b_impact, "bottom", "top")
straight(b_impact, b_execute, "right", "left")
straight(b_execute, b_critic, "right", "left")

# --- Row 3 (R -> L): Insight, BI, Answer ---
b_insight = box(2, 3, "Insight Agent", "Writes narrative")
b_bi = box(1, 3, "BI Agent", "Assembles response")
b_answer = box(0, 3, "Answer", "KPIs, Chart & Table", fill=END_FILL, edge=END_FILL, text=END_TEXT)
straight(b_critic, b_insight, "bottom", "top")
straight(b_insight, b_bi, "left", "right")
straight(b_bi, b_answer, "left", "right")

# --- Bottom band layout: explicit y-levels, evenly spaced, nothing overlaps ---
Y_LEGEND = 3.75
Y_RETRY_LABEL = 2.95
Y_RETRY_LINE = 2.45
Y_WRITEBACK_LABEL = 1.55
Y_WRITEBACK_LINE = 1.05
Y_FOOTER = 0.35

# --- Feedback loop: Critic (also Impact / Execute) retries Analysis ---
# Routed down through the empty footer band and back up the left margin,
# entering Analysis on its free left edge.
critic_bottom = edge_pt(b_critic, "bottom")
analysis_left = edge_pt(b_analysis, "left")
loop_x = LEFT_MARGIN * 0.35
polyline(
    [critic_bottom,
     (critic_bottom[0], Y_RETRY_LINE),
     (loop_x, Y_RETRY_LINE),
     (loop_x, analysis_left[1]),
     analysis_left],
    color=LOOP_COLOR, lw=2.0, ls=(0, (6, 4)),
)
ax.text(grid_width / 2, Y_RETRY_LABEL,
        "Impact / Execute / Critic can each retry Analysis with the exact error attached (bounded by MAX_ANALYSIS_ATTEMPTS)",
        ha="center", va="center", fontsize=11, color=LOOP_COLOR, style="italic", zorder=5)

# --- Query-memory write-back: BI -> RAG Agent ---
# Dips just below the box row, jogs out to the right margin, then drops
# straight down - this keeps it clear of the retry-loop label text instead
# of cutting through it on the way down.
bi_bottom = edge_pt(b_bi, "bottom")
rag_right = edge_pt(b_rag, "right")
loop_x2 = grid_width - RIGHT_MARGIN * 0.35
jog_y = BOTTOM_MARGIN - 0.25
polyline(
    [bi_bottom,
     (bi_bottom[0], jog_y),
     (loop_x2, jog_y),
     (loop_x2, rag_right[1]),
     rag_right],
    color=LOOP_COLOR, lw=2.0, ls=(0, (6, 4)),
)
ax.text(grid_width / 2, Y_WRITEBACK_LABEL,
        "critic-verified analyses are embedded back into query memory as worked examples for the next similar question",
        ha="center", va="center", fontsize=10.5, color=LOOP_COLOR, style="italic", zorder=5)

# --- Legend ---
ax.plot([LEFT_MARGIN, LEFT_MARGIN + 0.9], [Y_LEGEND, Y_LEGEND], color=FLOW_COLOR, lw=2.4)
ax.text(LEFT_MARGIN + 1.15, Y_LEGEND, "pipeline flow", ha="left", va="center", fontsize=10.5, color=SUB_COLOR)
ax.plot([LEFT_MARGIN + 3.2, LEFT_MARGIN + 4.1], [Y_LEGEND, Y_LEGEND], color=LOOP_COLOR, lw=2.2, linestyle=(0, (6, 4)))
ax.text(LEFT_MARGIN + 4.35, Y_LEGEND, "conditional / retry", ha="left", va="center", fontsize=10.5, color=SUB_COLOR)

# --- Footer credit ---
ax.text(grid_width / 2, Y_FOOTER,
        "Orchestrated by LangGraph  ·  Groq LLM  ·  FastAPI + Streamlit  ·  MegAI by Megha Patel",
        ha="center", fontsize=10.5, color=SUB_COLOR)

plt.tight_layout()
out_path = "agentic_bi_architecture.jpeg"
plt.savefig(out_path, facecolor=BG, bbox_inches="tight", pad_inches=0.3)
print(f"Saved {out_path}")
