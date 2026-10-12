"""Render the figures from a $OUT result directory.

    benchmark/eval/bench.sh plot                 # every figure into $OUT/figures
    benchmark/eval/plot.py $OUT --only models    # one figure
    benchmark/eval/plot.py $OUT --format pdf,png --texture
    benchmark/eval/plot.py $OUT --compare $OTHER   # two machines side by side

Each figure ships a `.tex` tabular of the same numbers next to it, so a value
that is hard to read off the plot is still available to the reader (and to the
text) without going back to the tsv.

Figure forms are picked from what the data has to show, not from habit:

    micro_speedup   ratios around a 1.0 baseline   -> diverging bars from 1.0
    fusion          the same variant before/after  -> dumbbell
    models          three systems per model        -> grouped bars, linear us
    dynamic         a measure over a length axis   -> lines
    batch           per-sequence cost over a batch axis -> lines, log-log
    precision       two variants, three dtypes     -> small multiples
    ablation        one knob at a time             -> grouped bars

--compare adds a second (or third) result directory and switches to the
cross-machine figures, which compare ratios rather than microseconds: absolute
times are a property of the accelerator, ratios are a property of the system
under test, and the claim those figures have to support is that the ordering
between fused, torch.compile and eager survives a change of GPU.

    compare_speedup speedup per benchmark, one series per machine
    compare_models  end-to-end ratio per model, one series per machine
    compare_dynamic ratio against torch.compile over sequence length

Absolute times span four orders of magnitude across the micro grid, so those
are shown as ratios or as dumbbells on a log axis; a bar whose baseline is not
zero misencodes magnitude, so no bar chart here uses a log length.
"""

import argparse
import collections
import csv
import json
import os
import math
import random
import re
import statistics
import sys

from rowdata import local

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

# --- palette ---------------------------------------------------------------
# Slots 1-4 of the reference categorical palette, used in fixed order, never
# cycled. Checked on white (OKLab dE x100, Machado 2009 CVD simulation at
# severity 1.0). Worst pair on the active pairlist:
#
#                                   CVD dE          normal dE
#   2 machines (compare figures)    9.2 (deutan)    27.6      PASS
#   3 series, adjacent + all-pairs  9.2 (deutan)    24.0      PASS
#   4 series, adjacent              9.1 (protan)    22.9      PASS
#   diverging blue<->red            21.6 (protan)   32.3      PASS
#
# Aqua (2.82:1) and yellow (2.17:1) are below 3:1 against white ("relief
# required"); the .tex table beside each figure is that relief.
SERIES = ["#1a4fa0", "#eb6834", "#1baf7a", "#eda100", "#8a5cd6", "#6b6b6b",
          "#c04a8a", "#4fa8e0", "#a0522d"]
POS, NEG = "#1a4fa0", "#e34948"      # diverging poles, neutral midpoint is the rule line
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
# Opt-in only: texture is for print and full CVD, never decoration.
HATCH = ["", "///", "...", "xxx", "\\\\", "++", "oo", "--", "||"]

SINGLE_COL, DOUBLE_COL = 3.25, 6.75   # single- and double-column figure widths, inches
# Figures not used at column width; they are drawn at the
# full text width.  --column overrides this.
WIDE = {"micro_speedup", "models", "ablation", "micro_baselines",
        "compile_overhead", "integration", "lazy", "suites_speedup",
        "suites_first", "suites_accuracy"}
# Width each figure is drawn at, so 7pt in the pdf is 7pt in print.
# \columnwidth is 3.25in; entries are the \includegraphics fractions in sections/*.tex.
PAGE_WIDTH = {
    "lazy_col": 0.98 * SINGLE_COL,
    "batch": 0.82 * SINGLE_COL,
    "models_col": 0.85 * SINGLE_COL,
    "micro_speedup_col": 0.90 * SINGLE_COL,
}

SYSTEM_LABEL = {
    "ours": "MetaTensor",
    "torch-compile": "torch.compile",
    "torch-compile-ro": "torch.compile (reduce-overhead)",
    "torch-compile-nobreak": "torch.compile, torch.where (no break)",
    "jax-host": "JAX, host branch (graph break)",
    "jax-static": "JAX, static argument (retrace)",
    "jax-cond": "JAX, lax.cond (no break)",
    "torch-compile-mat": "torch.compile (max-autotune)",
    "torch-eager": "PyTorch eager",
    "torch-compile-dynamic": "torch.compile (dynamic)",
    "torch-compile-static": "torch.compile (static)",
    "torch-inductor": "TorchInductor",
    "torch-inductor-cg": "TorchInductor (cudagraphs)",
    "jax": "JAX/XLA",
    "jax-xla": "JAX/XLA",
    "iree": "IREE",
    "torch-tensorrt": "Torch-TensorRT",
    "triton": "Triton (handwritten)",
    # where the DAG lives (bench.sh lazy)
    "virtual": "MetaTensor (DAG in the JIT)",
    "deferred": "deferred library (DAG at run time)",
    "eager": "eager (no DAG)",
    # early exit (bench.sh control)
    "jax-perlayer": "JAX/XLA (per-layer jit, host sync)",
    "jax-while": "JAX/XLA (lax.while_loop)",
}
# Names come from benchmarks.toml, the file the grid is read from. A run whose
# variants predate an entry still plots, with the bare variant number.
try:
    import grid as _grid
    VARIANT_LABEL = _grid.labels(_grid.load())
except Exception:  # no benchmarks.toml next to this checkout
    VARIANT_LABEL = {}


def style(font):
    plt.rcParams.update({
        "font.family": font,
        "font.serif": ["STIX Two Text", "DejaVu Serif", "Times New Roman"],
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        # Hairline, solid, recessive chrome, never dashed.
        "axes.edgecolor": GRID, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "legend.frameon": False, "legend.handlelength": 1.6,
        "figure.dpi": 200, "savefig.dpi": 200,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        # Embed TrueType rather than Type 3, which most venues reject.
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def despine(ax, keep=("left", "bottom")):
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


# --- data ------------------------------------------------------------------

def read_tsv(path):
    if not os.path.exists(local(path)):
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_jsonl(path):
    if not os.path.exists(local(path)):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def med(xs):
    xs = [float(x) for x in xs if x not in (None, "")]
    return statistics.median(xs) if xs else None


def micro_values(rows):
    """Read the dtype as summarize.py does (our rows and torch's use different tsv columns)."""
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        dtype = r.get("breaks") or r.get("graphs") or "float64"
        key = (dtype, int(r["variant"]), int(r["k"]), int(r["n"]))
        g[key][r["mode"]].append(float(r["steady_us"]))
    return g


def micro_groups(rows):
    return {k: {m: med(v) for m, v in d.items()}
            for k, d in micro_values(rows).items()}


def band(xs):
    """(median, min, max) over the rounds. With three rounds a standard
    deviation is noise; the observed range is what there is to report."""
    xs = [float(x) for x in xs if x not in (None, "")]
    if not xs:
        return None, None, None
    return statistics.median(xs), min(xs), max(xs)


def ratio_band(num, den):
    """Envelope of num/den when the rounds are not paired across the two
    systems, so the extremes are taken from opposite ends."""
    nm, nlo, nhi = band(num)
    dm, dlo, dhi = band(den)
    if not nm or not dm:
        return None, None, None
    return nm / dm, nlo / dhi, nhi / dlo


def err(values, lows, highs):
    """Matplotlib wants distances from the point, not absolute bounds."""
    return [[v - lo for v, lo in zip(values, lows)],
            [hi - v for v, hi in zip(values, highs)]]


ERRBAR = dict(ecolor=INK_2, elinewidth=0.7, capsize=2.2, capthick=0.7)


def spread_str(lo, hi, fmt="%.1f"):
    return (fmt + "-" + fmt) % (lo, hi) if lo is not None else "n/a"


# --- system identity -------------------------------------------------------
# One colour, one hatch and one marker per system, shared by every figure.
# MetaTensor is the darkest hue and the first slot of a bar group. Hatches and
# markers carry identity in greyscale and under full CVD.
SYSTEM_IDENTITY = collections.OrderedDict([
    #  system                  colour     hatch  marker
    ("ours",                  ("#1a4fa0", "",    "o")),
    ("torch-compile",         ("#eb6834", "///", "s")),
    ("torch-compile-static",  ("#eb6834", "///", "s")),
    ("torch-compile-ro",      ("#1baf7a", "...", "D")),
    ("torch-compile-dynamic", ("#1baf7a", "...", "D")),
    ("torch-compile-mat",     ("#eda100", "xxx", "^")),
    ("torch-eager",           ("#8a5cd6", "\\\\", "v")),
    ("torch-inductor",        ("#eb6834", "///", "s")),
    ("torch-inductor-cg",     ("#1baf7a", "...", "D")),
    ("jax",                   ("#6b6b6b", "++",  "P")),
    ("jax-xla",               ("#6b6b6b", "++",  "P")),
    ("torch-compile-nobreak", ("#eda100", "xxx", "^")),
    ("jax-host",              ("#6b6b6b", "++",  "P")),
    ("jax-static",            ("#c04a8a", "oo",  "*")),
    ("jax-cond",              ("#4fa8e0", "--",  "X")),
    ("iree",                  ("#c04a8a", "oo",  "*")),
    ("torch-tensorrt",        ("#4fa8e0", "--",  "X")),
    ("triton",                ("#a0522d", "||",  "h")),
    # not a baseline: the same code with the JIT off, the "before" end of the
    # fusion dumbbell.
    ("nojit",                 ("#9a9994", "",    "o")),
    # The three arms of the DAG-placement experiment. virtual is our system and
    # keeps its colour; the others are the same runtime with the DAG elsewhere.
    ("virtual",               ("#1a4fa0", "",    "o")),
    ("deferred",              ("#d94f70", "\\\\", "s")),
    ("eager",                 ("#9a9994", "",    "^")),
    ("nofuse",                ("#9a9994", "",    "^")),
    ("nojit-port",            ("#3f3f3f", "..",  "v")),
    ("jax-perlayer",          ("#6b6b6b", "++",  "P")),
    ("jax-while",             ("#3f3f3f", "..",  "X")),
])
MARKERS = "osD^vP*Xh"
# Machines are not systems; the compare_* figures colour by machine.
MACHINE_COLORS = ["#1a4fa0", "#eb6834", "#1baf7a"]
MACHINE_HATCH = ["", "///", "..."]


def style_of(system):
    return SYSTEM_IDENTITY.get(system, ("#6b6b6b", "", "o"))


def legend_below(fig, ax=None, ncol=2, handles=None, labels=None, fontsize=7):
    """One legend per figure, below the axes (inside, it ends up over a bar)."""
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return
    # Fewer columns rather than a legend wider than the page; fit is measured.
    for n in range(max(1, ncol), 0, -1):
        leg = fig.legend(handles, labels, loc="outside lower center", ncol=n,
                         frameon=False, columnspacing=1.1, handlelength=1.4,
                         handletextpad=0.5, fontsize=fontsize,
                         borderaxespad=0.2)
        if n == 1:
            return leg
        fig.canvas.draw()
        if leg.get_window_extent().width / fig.dpi <= fig.get_size_inches()[0]:
            return leg
        leg.remove()


def drawn(systems, args):
    """IREE runs 1-2 orders of magnitude behind XLA on this GPU and flattens shared
    axes; the figures show it only with --iree (_iree suffix), the tables always."""
    if getattr(args, "iree", False):
        return list(systems)
    return [s for s in systems if s != "iree"]


def micro_label(variant, k, n):
    name = VARIANT_LABEL.get(variant, "v%d" % variant)
    if variant <= 5:
        return "%s k=%d n=%s" % (name, k, si(n))
    return "%s n=%s" % (name, si(n))


def si(n):
    for div, suf in ((1000000, "M"), (1000, "k")):
        if n >= div:
            v = n / float(div)
            return "%s%s" % (("%.1f" % v).rstrip("0").rstrip("."), suf)
    return str(n)


# --- table companion -------------------------------------------------------

def write_table(path, caption, header, rows, align=None):
    align = align or ("l" + "r" * (len(header) - 1))
    esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
    out = [r"\begin{tabular}{%s}" % align, r"\toprule",
           " & ".join(esc(h) for h in header) + r" \\", r"\midrule"]
    out += [" & ".join(esc(c) for c in row) + r" \\" for row in rows]
    out += [r"\bottomrule", r"\end{tabular}", "%% %s" % caption]
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


# --- figures ---------------------------------------------------------------

def fig_micro_speedup(out, args):
    """Ratios sit either side of a 1.0 baseline, so the form is a diverging bar."""
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    pts = []
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused") or not m.get("torch-compile"):
            continue
        r, lo, hi = ratio_band(m["torch-compile"], m["fused"])
        pts.append((r, micro_label(v, k, n), lo, hi, m))
    if not pts:
        return None
    pts.sort()
    vals = [p[0] for p in pts]
    labels = [p[1] for p in pts]
    lows = [p[2] for p in pts]
    highs = [p[3] for p in pts]
    y = range(len(pts))
    # app rows only exist when the point was also run with --applevel; a
    # point without them just gets no marker.
    app_y, app_x = [], []
    for i, (_, _, _, _, m) in enumerate(pts):
        if m.get("app"):
            am, _, _ = ratio_band(m["torch-compile"], m["app"])
            if am is not None:
                app_y.append(i)
                app_x.append(am)

    fig, ax = plt.subplots(figsize=(args.width, 0.155 * len(pts) + 1.15))
    colors = [POS if v >= 1 else NEG for v in vals]
    # Bars grow from the 1.0 rule, which is the neutral midpoint of the scale.
    ax.barh(list(y), [v - 1 for v in vals], left=1, height=0.55,
            color=colors, linewidth=0,
            hatch=(HATCH[0] if not args.texture else None))
    ax.errorbar(vals, list(y), xerr=err(vals, lows, highs), fmt="none",
                zorder=4, **ERRBAR)
    if app_x:
        # No error bars, the row is busy enough (range is in the table). A white
        # face keeps the marker readable on a saturated bar.
        ax.scatter(app_x, app_y, s=18, facecolors="white", edgecolors=INK,
                   linewidth=0.9, zorder=6)
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([0.5, 1, 2, 4, 8, 16])
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "\u00d7"))
    ax.set_yticks(list(y), labels)
    ax.set_xlim(min(lows) / 1.5, max(highs) * 2.4)
    ax.set_xlabel("speedup over torch.compile\nlog2 scale; higher is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    # Direct-label the extremes only; the rest are in the table.
    for i in (0, len(pts) - 1):
        v = vals[i]
        # Always to the right of the row, past the whisker: a label placed to
        # the left of a sub-1x bar runs into the benchmark names.
        at = max(highs[i], 1.0)
        ax.annotate("%.2f\u00d7" % v, (at, i), xytext=(4, 0),
                    textcoords="offset points", va="center",
                    ha="left", fontsize=7, color=INK_2)
    legend_handles = [
        Line2D([], [], color=POS, lw=4, label="MetaTensor faster"),
        Line2D([], [], color=NEG, lw=4, label="torch.compile faster")]
    if app_x:
        legend_handles.append(Line2D(
            [], [], marker="o", linestyle="none", markerfacecolor="white",
            markeredgecolor=INK, markersize=5,
            label="app-level through PyPy"))
    legend_below(fig, handles=legend_handles,
                 labels=[h.get_label() for h in legend_handles], ncol=2)
    if args.titles:
        ax.set_title("Microbenchmark speedup, float64")
    rows = []
    header = ["benchmark", "MetaTensor", "range"]
    if app_x:
        header += ["app-level", "range"]
    header += ["torch.compile", "range", "speedup", "range"]
    if app_x:
        header += ["app speedup", "range"]
    for r, lab, lo, hi, m in pts:
        om, olo, ohi = band(m["fused"])
        cm, clo, chi = band(m["torch-compile"])
        row = [lab, "%.1f" % om, spread_str(olo, ohi)]
        if app_x:
            if m.get("app"):
                apm, aplo, aphi = band(m["app"])
                ar, arlo, arhi = ratio_band(m["torch-compile"], m["app"])
                row += ["%.1f" % apm, spread_str(aplo, aphi)]
            else:
                row += ["n/a", "n/a"]
        row += ["%.1f" % cm, spread_str(clo, chi), "%.2f" % r,
                spread_str(lo, hi, "%.2f")]
        if app_x:
            row += (["%.2f" % ar, spread_str(arlo, arhi, "%.2f")]
                    if m.get("app") else ["n/a", "n/a"])
        rows.append(row)
    write_table(os.path.join(args.outdir, "micro_speedup.tex"),
                "median us per iteration over the rounds, with the observed range",
                header, rows)
    return fig, "micro_speedup"


def fig_fusion(out, args):
    """The same benchmark before and after fusion: a dumbbell, one row each."""
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    pts = []
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused") or not m.get("nojit"):
            continue
        fm, flo, fhi = band(m["fused"])
        sm, slo, shi = band(m["nojit"])
        pts.append((sm / fm, sm, fm, micro_label(v, k, n),
                    (flo, fhi), (slo, shi)))
    if not pts:
        return None
    pts.sort()
    y = range(len(pts))

    fig, ax = plt.subplots(figsize=(args.width, 0.155 * len(pts) + 1.05))
    for i, (_, slow, fast, _lab, _f, _s) in enumerate(pts):
        ax.plot([fast, slow], [i, i], color=GRID, linewidth=1.2, zorder=1,
                solid_capstyle="round")
    slow = [p[1] for p in pts]
    fast = [p[2] for p in pts]
    ax.errorbar(slow, list(y), xerr=err(slow, [p[5][0] for p in pts],
                                       [p[5][1] for p in pts]),
                fmt="none", zorder=2, **ERRBAR)
    ax.errorbar(fast, list(y), xerr=err(fast, [p[4][0] for p in pts],
                                        [p[4][1] for p in pts]),
                fmt="none", zorder=2, **ERRBAR)
    ax.scatter(slow, list(y), s=14, color=style_of("nojit")[0], marker="o",
               zorder=3, linewidth=0, label="interpreted (no JIT)")
    ax.scatter(fast, list(y), s=14, color=style_of("ours")[0], marker="o",
               zorder=3, linewidth=0, label="MetaTensor (fused)")
    ax.set_xscale("log")
    ax.set_yticks(list(y), [p[3] for p in pts])
    ax.set_xlabel("steady-state time per iteration (\u00b5s, log scale)\n"
                  "lower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    best = pts[-1]
    # To the right of the slow end: to the left of the fast end it lands on
    # the benchmark name.
    ax.annotate("%.1f\u00d7" % best[0], (best[1], len(pts) - 1),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=7, color=INK_2)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("Effect of kernel fusion")
    write_table(os.path.join(args.outdir, "fusion.tex"),
                "median us per iteration over the rounds, with the observed range",
                ["benchmark", "fused", "range", "nojit", "range", "gain"],
                [[p[3], "%.1f" % p[2], spread_str(*p[4]),
                  "%.1f" % p[1], spread_str(*p[5]), "%.2f" % p[0]]
                 for p in reversed(pts)])
    return fig, "fusion"


def fig_fusion_stats(out, args):
    """Fusion region sizes and what ended them. Bars are force reasons per model
    (see fusion_stats.py and the README)."""
    rows = read_tsv(os.path.join(out, "fusion.tsv"))
    if not rows:
        return None
    reasons = [("forced_library", "library call"), ("forced_item", ".item()"),
               ("forced_loop", "loop/guard exit"),
               ("forced_leafcap", "leaf cap"), ("forced_assign", "assign"),
               ("forced_other", "other")]
    num = lambda r, k: float(r.get(k) or 0)
    fig, ax = plt.subplots(figsize=(args.width, 0.3 * len(rows) + 1.2))
    y = list(range(len(rows)))
    left = [0.0] * len(rows)
    for i, (key, label) in enumerate(reasons):
        vals = [num(r, key) for r in rows]
        if not any(vals):
            continue
        ax.barh(y, vals, left=left, height=0.7, linewidth=0,
                color=SERIES[i % len(SERIES)],
                hatch=HATCH[i % len(HATCH)] if args.texture else None,
                label=label)
        left = [a + b for a, b in zip(left, vals)]
    ax.set_yticks(y, [r["model"] for r in rows])
    ax.set_xlabel("fusion regions per forward,\nby what ended the region")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Why each fusion region ends")
    write_table(os.path.join(args.outdir, "fusion_stats.tex"),
                "per model forward: distinct fused kernels, kernel launches "
                "per iteration, DAG nodes per kernel, extra outputs, and the "
                "force reason of every region in the compiled traces",
                ["model", "kernels", "launches/iter", "nodes min", "median",
                 "max", "extra outs", "library", ".item()", "loop", "leaf cap",
                 "assign", "other"],
                [[r["model"], r["kernels"], r["launches_per_iter"],
                  r["nodes_min"], r["nodes_median"], r["nodes_max"],
                  r["extra_outputs_total"]] +
                 [r[k] for k, _ in reasons] for r in rows])
    return fig, "fusion_stats"


# Out of the population (and every figure, table and mean): two untrained
# fixtures and one re-uploaded mirror. No sweep measures them; the names stay
# so older result sets still draw only the population. select_models.py records
# the failed criterion.
PROBES = ("tiny-gpt2", "bert-tiny", "vit-tiny")

# Admitted below the usage floor, each as the only cover for something the
# population would lose: the small end of the encoder ladder and the only
# attention-free architecture. The headline mean is reported with and without them.
C3_EXEMPT = ("bert-mini", "mixer_b16")


def _model_medians(out):
    """{model: {system: median us}} from models.tsv."""
    rows = read_tsv(os.path.join(out, "models.tsv"))
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        try:
            g[r["model"]][r["system"]].append(float(r["steady_us"]))
        except (ValueError, KeyError):
            pass
    return dict((m, dict((s, med(v)) for s, v in d.items()))
                for m, d in g.items())


def ratio_ticks(lo, hi):
    """Log2 ticks for a ratio axis. Half-steps (1.5x) when the range is under one
    octave, plain powers of two when wider so labels do not collide."""
    span = math.log2(max(hi, 1.0) / min(lo, 1.0)) if lo > 0 else 8
    multipliers = (1, 1.5) if span <= 5 else (1,)
    cands = sorted(m * 2.0 ** k for k in range(-8, 8) for m in multipliers)
    ticks = [t for t in cands if lo / 1.15 <= t <= hi * 1.15]
    return ticks or [1]


def _geomean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else None


# Baselines the distribution is drawn against, in drawing order. "best-of-three"
# is the fastest torch.compile mode per model.
SPEEDUP_BASES = [("torch-eager", "PyTorch eager"),
                 ("best-of-three", "torch.compile, fastest of three modes"),
                 ("jax", "JAX/XLA"),
                 ("torch-tensorrt", "Torch-TensorRT")]
COMPILE_MODES = ("torch-compile", "torch-compile-ro", "torch-compile-mat")


def _base_us(d, base):
    if base == "best-of-three":
        cands = [d[s] for s in COMPILE_MODES if d.get(s)]
        return min(cands) if cands else None
    return d.get(base)


def fig_model_speedup(out, args):
    """Per-model speedup distribution: one sorted curve per baseline, geometric mean in the legend."""
    g = _model_medians(out)
    models = [m for m in g if m not in PROBES]
    if not models:
        return None
    fig, ax = plt.subplots(figsize=(args.width, 2.5))
    table_rows, curves = [], []
    for base, label in SPEEDUP_BASES:
        sp = []
        for m in models:
            ours, b = g[m].get("ours"), _base_us(g[m], base)
            if ours and b:
                sp.append(b / ours)
        if not sp:
            continue
        sp.sort()
        curves.append(sp)
        gm = _geomean(sp)
        color, _, marker = style_of("torch-compile" if base == "best-of-three" else base)
        ax.plot(range(1, len(sp) + 1), sp, marker=marker, markersize=3.5,
                linewidth=1.2, color=color,
                label="%s (gmean %.2f$\\times$)" % (label, gm))
        qual = [b / g[m]["ours"] for m in models
                if m not in C3_EXEMPT and g[m].get("ours")
                for b in [_base_us(g[m], base)] if b]
        table_rows.append([label, "%d" % len(sp), "%.2f" % gm,
                           "%.2f" % _geomean(qual) if qual else "n/a",
                           "%.2f" % sp[0], "%.2f" % sp[-1],
                           "%d" % sum(1 for x in sp if x > 1.0)])
    # Everything under the line is a loss; shade it.
    ax.axhspan(1e-3, 1.0, color="0.92", zorder=0)
    ax.axhline(1.0, color="0.35", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("model configurations, sorted by speedup")
    ax.set_ylabel("speedup over baseline\n(>1 favours MetaTensor)")
    ax.set_xlim(0.5, max(2, len(models)) + 0.5)
    # Speedup ticks read 0.8x and 2x, not the default log decades.
    lo = min(min(c) for c in curves) if curves else 0.5
    hi = max(max(c) for c in curves) if curves else 2.0
    ticks = [t for t in (0.25, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8)
             if lo / 1.15 <= t <= hi * 1.15]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["%g$\\times$" % t for t in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(lo / 1.15, hi * 1.15)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    legend_below(fig, ax, ncol=1 if args.width < DOUBLE_COL else 2)
    if args.titles:
        ax.set_title("End-to-end speedup distribution")
    write_table(os.path.join(args.outdir, "model_speedup.tex"),
                "speedup of MetaTensor over each baseline across the "
                "population, medians of three rounds",
                ["baseline", "models", "geomean", "geomean, C3 only",
                 "min", "max", "faster on"],
                table_rows)
    return fig, "model_speedup"


def fig_models(out, args):
    """Three systems per model, all in the same unit: grouped bars from zero."""
    rows = read_tsv(os.path.join(out, "models.tsv"))
    if not rows:
        return None
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[r["model"]][r["system"]].append(float(r["steady_us"]))
    systems = [s for s in ["ours", "torch-compile", "torch-compile-ro", "torch-compile-mat",
                           "torch-eager", "jax", "iree", "torch-tensorrt"]
               if any(s in d for d in g.values())]
    models = sorted((m for m in g if m not in PROBES),
                    key=lambda m: med(g[m].get("ours", [])) or 0)
    stats = {s: [band(g[m].get(s, [])) for m in models] for s in systems}
    vals = {s: [b[0] or 0 for b in stats[s]] for s in systems}

    shown = drawn(systems, args)
    # 3 systems keeps the original 0.26 half-height; more systems shrink to fit.
    h = 0.8 / len(shown)
    fig, ax = plt.subplots(figsize=(args.width, 0.42 * len(models) + 1.6))
    y = [i for i in range(len(models))]
    for si_, s in enumerate(shown):
        # MetaTensor first means topmost in every horizontal bar group.
        off = ((len(shown) - 1) / 2 - si_) * h
        color, hatch, _ = style_of(s)
        ax.barh([v + off for v in y], vals[s], height=h * 0.88,
                color=color, linewidth=0,
                hatch=hatch if args.texture else None,
                label=SYSTEM_LABEL[s])
        ax.errorbar(vals[s], [v + off for v in y],
                    xerr=err(vals[s], [b[1] or 0 for b in stats[s]],
                             [b[2] or 0 for b in stats[s]]),
                    fmt="none", zorder=4, **ERRBAR)
    ax.set_yticks(y, models)
    ax.set_xscale("log")
    ax.set_xlim(left=50)
    ax.set_xlabel("steady-state time per iteration (\u00b5s, log scale)\n"
                  "lower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3 if args.width >= DOUBLE_COL else 2)
    if args.titles:
        ax.set_title("End-to-end inference")

    ov, ov_st = vals.get("ours", [0] * len(models)), stats.get("ours", [(None, None, None)] * len(models))
    tc, tc_st = vals.get("torch-compile", [0] * len(models)), stats.get("torch-compile", [(None, None, None)] * len(models))
    tcro, tcro_st = vals.get("torch-compile-ro", [0] * len(models)), stats.get("torch-compile-ro", [(None, None, None)] * len(models))
    tcmat, tcmat_st = vals.get("torch-compile-mat", [0] * len(models)), stats.get("torch-compile-mat", [(None, None, None)] * len(models))
    te = vals.get("torch-eager", [0] * len(models))
    jx = vals.get("jax", [0] * len(models))
    ir = vals.get("iree", [0] * len(models))
    trt = vals.get("torch-tensorrt", [0] * len(models))
    header = ["model", "MetaTensor", "range", "torch.compile", "range",
              "compile-ro", "range", "compile-mat", "range", "eager"]
    if "jax" in systems:
        header.append("JAX/XLA")
    if "iree" in systems:
        header.append("IREE")
    header.append("TensorRT")
    header.append("ratio")
    if "jax" in systems:
        header.append("jax/compile")
    if "torch-tensorrt" in systems:
        header.append("trt/compile")
    table_rows = []
    for i, m in enumerate(models):
        row = [m, "%.0f" % ov[i], spread_str(ov_st[i][1], ov_st[i][2], "%.0f"),
               "%.0f" % tc[i], spread_str(tc_st[i][1], tc_st[i][2], "%.0f"),
               "%.0f" % tcro[i] if tcro[i] else "n/a",
               spread_str(tcro_st[i][1], tcro_st[i][2], "%.0f"),
               "%.0f" % tcmat[i] if tcmat[i] else "n/a",
               spread_str(tcmat_st[i][1], tcmat_st[i][2], "%.0f"),
               "%.0f" % te[i]]
        if "jax" in systems:
            row.append("%.0f" % jx[i] if jx[i] else "n/a")
        if "iree" in systems:
            row.append("%.0f" % ir[i] if ir[i] else "n/a")
        row.append("%.0f" % trt[i] if trt[i] else "n/a")
        row.append("%.2f" % (ov[i] / tc[i]) if tc[i] else "n/a")
        if "jax" in systems:
            row.append("%.2f" % (jx[i] / tc[i]) if tc[i] and jx[i] else "n/a")
        if "torch-tensorrt" in systems:
            row.append("%.2f" % (trt[i] / tc[i]) if tc[i] and trt[i] else "n/a")
        table_rows.append(row)
    write_table(os.path.join(args.outdir, "models.tex"),
                "median us per iteration over the rounds, with the observed range",
                header, table_rows)
    return fig, "models"


PHASE_ORDER = {"first": 0, "revisit": 1, "new": 2, "1": 0, "2": 1}
PHASE_LABEL = {"first": "first visit", "revisit": "revisit",
               "new": "new length distribution", "1": "first visit",
               "2": "revisit"}


def fig_dynamic(out, args):
    """One panel per phase (pooling would mix cold and warm visits). The line is
    the steady-state median per length; compile events are counters in the tables."""
    rows = read_tsv(os.path.join(out, "dynamic_summary.tsv"))
    if not rows:
        return None
    systems = [s for s in ["ours", "torch-compile-static",
                           "torch-compile-dynamic", "torch-eager"]
               if any(r["system"] == s for r in rows)]
    phases = sorted({r.get("pass", "1") for r in rows},
                    key=lambda p: PHASE_ORDER.get(p, 9))
    per = collections.defaultdict(list)
    for r in rows:
        per[(r["system"], r.get("pass", "1"), int(r["length"]))].append(r)
    lengths = sorted({int(r["length"]) for r in rows})
    per_phase_lengths = {p: sorted({int(r["length"]) for r in rows
                                    if r.get("pass", "1") == p})
                         for p in phases}

    # Stacked, since three panels side by side leave an inch each.
    if args.width >= DOUBLE_COL:
        fig, axes = plt.subplots(1, len(phases), squeeze=False, sharey=True,
                                 figsize=(args.width, args.width * 0.46))
        axes = list(axes[0])
    else:
        # Not sharex: each phase sweeps its own lengths.
        fig, axes = plt.subplots(len(phases), 1, squeeze=False, sharey=True,
                                 figsize=(args.width,
                                          1.25 * len(phases) + 1.0))
        axes = [a[0] for a in axes]
    top = 0.0
    for ax, phase in zip(axes, phases):
        xs = per_phase_lengths[phase]
        for s in systems:
            bands = [band([float(r["median_us"]) for r in per.get((s, phase, L), [])])
                     for L in xs]
            if not any(b[0] for b in bands):
                continue
            ys = [b[0] or 0.0 for b in bands]
            top = max(top, max(b[2] or 0.0 for b in bands))
            color, _, marker = style_of(s)
            ax.errorbar(xs, ys, yerr=err(ys, [b[1] or 0.0 for b in bands],
                                         [b[2] or 0.0 for b in bands]),
                        fmt="none", zorder=2, **ERRBAR)
            ax.plot(xs, ys, color=color, linewidth=1.3, marker=marker,
                    markersize=3.0, markeredgewidth=0,
                    label=SYSTEM_LABEL[s], zorder=3)
        ax.set_xticks(xs)
        ax.set_title(PHASE_LABEL.get(phase, phase), fontsize=7)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    # One x label per figure when stacked, one per panel otherwise.
    if args.width >= DOUBLE_COL:
        for ax in axes:
            ax.set_xlabel("sequence length (tokens)")
    else:
        axes[-1].set_xlabel("sequence length (tokens)")
    axes[0].set_ylabel("median time per step (\u00b5s)\nlower is better")
    axes[0].set_ylim(0, top * 1.08)
    legend_below(fig, axes[0], ncol=2)
    if args.titles:
        fig.suptitle("Changing sequence length")

    DELTAS = ["loops", "bridges", "kernels", "cache_hits", "new_graphs"]

    def delta(key, col):
        vs = [float(r.get(col) or 0) for r in per.get(key, [])]
        return med(vs) if vs else 0.0

    time_rows = []
    for s in systems:
        for phase in phases:
            row = [SYSTEM_LABEL[s], PHASE_LABEL.get(phase, phase)]
            for L in lengths:
                vs = [float(r["median_us"]) for r in per.get((s, phase, L), [])]
                row.append("%.0f [%s]" % (band(vs)[0],
                                          spread_str(*band(vs)[1:], fmt="%.0f"))
                           if vs else "-")
            row += ["%.0f" % sum(delta((s, phase, L), c) for L in lengths)
                    for c in DELTAS]
            time_rows.append(row)
    write_table(os.path.join(args.outdir, "dynamic.tex"),
                "median us per step over the rounds (range in brackets), and "
                "the compile counters summed over the phase. first visit is "
                "the first time each length is seen, revisit is the same "
                "lengths again, and the last phase is five lengths the run "
                "has not seen",
                ["system", "phase"] + [str(L) for L in lengths]
                + ["loops", "bridges", "kernels", "hits", "new graphs"],
                time_rows)
    write_table(os.path.join(args.outdir, "dynamic_deltas.tex"),
                "per-length compile counters: the delta inside that length's "
                "window, median over the rounds. new graphs counts graphs "
                "compiled in the window, which on a first visit are first "
                "compilations and on a revisit are recompilations",
                ["system", "phase", "length", "loops", "bridges", "kernels",
                 "cache hits", "new graphs"],
                [[SYSTEM_LABEL[s], PHASE_LABEL.get(p, p), str(L)]
                 + ["%.0f" % delta((s, p, L), c) for c in DELTAS]
                 for s in systems for p in phases
                 for L in per_phase_lengths[p]])
    return fig, "dynamic"


def fig_precision(out, args):
    """One axis of speedup instead of two panels with unrelated y-scales (a ratio is unitless)."""
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    dtypes = ["float64", "float32", "float16"]
    variants = sorted({k[1] for k in g if k[0] != "float64"})
    if not variants:
        return None
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.85))
    x = range(len(dtypes))
    w = 0.8 / max(len(variants), 1)
    table = []
    for vi, v in enumerate(variants):
        n = max(k[3] for k in g if k[1] == v)
        ratios, lows, highs, absolute = [], [], [], []
        for d in dtypes:
            m = g.get((d, v, 1, n)) or {}
            r, lo, hi = ratio_band(m.get("torch-compile", []), m.get("fused", []))
            ratios.append(r or 0)
            lows.append(lo or 0)
            highs.append(hi or 0)
            absolute.append((band(m.get("fused", [])), band(m.get("torch-compile", []))))
        off = (vi - (len(variants) - 1) / 2) * w
        ax.bar([i + off for i in x], ratios, w * 0.86, color=SERIES[vi],
               linewidth=0, hatch=HATCH[vi] if args.texture else None,
               label="%s (n=%s)" % (VARIANT_LABEL.get(v, "v%d" % v), si(n)))
        ax.errorbar([i + off for i in x], ratios,
                    yerr=err(ratios, lows, highs), fmt="none", zorder=4,
                    **ERRBAR)
        for d, r, lo, hi, (ob, cb) in zip(dtypes, ratios, lows, highs, absolute):
            table.append([VARIANT_LABEL.get(v, "v%d" % v), d,
                          "%.1f" % ob[0] if ob[0] else "n/a",
                          "%.1f" % cb[0] if cb[0] else "n/a",
                          "%.2f" % r if r else "n/a",
                          spread_str(lo, hi, "%.2f") if r else "n/a"])
    ax.axhline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xticks(list(x), ["float64", "float32", "float16"])
    ax.set_xlabel("element type")
    ax.set_ylabel("speedup over torch.compile (×)\nhigher is better")
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v) + "\u00d7"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Precision sweep")
    write_table(os.path.join(args.outdir, "precision.tex"),
                "median us per iteration over the rounds, with the speedup range",
                ["variant", "dtype", "MetaTensor", "torch.compile", "speedup",
                 "range"], table)
    return fig, "precision"


def fig_ablation(out, args):
    """One row per knob at two settings against the same model. One hue, with each
    setting written next to its bar, since settings differ per experiment."""
    rows = read_tsv(os.path.join(out, "ablation.tsv"))
    if not rows:
        return None
    g = collections.defaultdict(list)
    gl = collections.defaultdict(list)
    for r in rows:
        key = (r["experiment"], r["model"], r["variant"])
        if r.get("steady_us"):
            g[key].append(float(r["steady_us"]))
        if r.get("launches_per_iter"):
            gl[key].append(float(r["launches_per_iter"]))
    if not g:
        return None

    def setting_key(v):
        # "8" before "64": these are numbers wherever they look like numbers.
        try:
            return (0, float(v), "")
        except ValueError:
            return (1, 0.0, v)

    groups = collections.OrderedDict()
    for exp, model, var in sorted(g, key=lambda k: (k[0], k[1], setting_key(k[2]))):
        groups.setdefault((exp, model), []).append(
            (var,) + band(g[(exp, model, var)]))

    keys = list(groups)
    fig, ax = plt.subplots(figsize=(args.width, 0.52 * len(keys) + 0.8))
    height = 0.30
    ys, xs, lows, highs, texts, ticks = [], [], [], [], [], []
    for i, key in enumerate(keys):
        items = groups[key]
        for slot, (var, value, lo, hi) in enumerate(items):
            ys.append(i + ((len(items) - 1) / 2 - slot) * height)
            xs.append(value)
            lows.append(lo)
            highs.append(hi)
            texts.append(var)
        ticks.append("%s / %s" % (key[0].replace("_", " "), key[1]))
    ax.barh(ys, xs, height=height * 0.88, color=SERIES[0], linewidth=0,
            hatch=HATCH[0] if args.texture else None)
    ax.errorbar(xs, ys, xerr=err(xs, lows, highs), fmt="none", zorder=4,
                **ERRBAR)
    for xv, yv, t in zip(xs, ys, texts):
        ax.annotate(t, (xv, yv), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=6.5, color=INK_2)
    ax.set_yticks(range(len(keys)), ticks)
    ax.set_xlabel("steady-state time per iteration (\u00b5s)\nlower is better")
    ax.set_xlim(0, max(highs) * 1.22)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    if args.titles:
        ax.set_title("Ablations")
    write_table(os.path.join(args.outdir, "ablation.tex"),
                "median us per iteration over the rounds, with the observed "
                "range, and launches/iter where the experiment records it",
                ["experiment", "model", "setting", "us", "range", "launches/iter"],
                [[e.replace("_", " "), m, v, "%.0f" % band(g[(e, m, v)])[0],
                  spread_str(*band(g[(e, m, v)])[1:], fmt="%.0f"),
                  "%.1f" % statistics.median(gl[(e, m, v)]) if gl.get((e, m, v)) else ""]
                 for (e, m, v) in sorted(g, key=lambda k: (k[0], k[1], setting_key(k[2])))])
    return fig, "ablation"


def fig_explain(out, args):
    """Table only: Dynamo graph count/break count/op count per model, one
    row each, no rounds to spread over."""
    rows = read_tsv(os.path.join(out, "explain.tsv"))
    if not rows:
        return None
    write_table(os.path.join(args.outdir, "explain.tex"),
                "torch._dynamo.explain graph structure per model",
                ["model", "graphs", "breaks", "ops"],
                [[r["model"], r.get("graphs", ""), r.get("breaks", ""),
                  r.get("ops", "")] for r in rows])
    fig, ax = plt.subplots(figsize=(args.width, 0.3))
    ax.axis("off")
    return fig, "explain"


def append_notes(path, paragraphs):
    """Add a notes block under a table written by write_table."""
    with open(path, "a") as f:
        f.write("\n%% notes\n\\begin{minipage}{\\linewidth}\\footnotesize\n")
        for para in paragraphs:
            f.write("\\par\\smallskip\n" + para + "\n")
        f.write("\\end{minipage}\n")


def fig_op_inventory(out, args):
    """Table only: every operator a forward runs, both systems in one vocabulary,
    so differing cells and their reasons are visible."""
    rows = read_tsv(os.path.join(out, "op_inventory.tsv"))
    if not rows:
        return None
    words = [c for c in rows[0] if c not in ("model", "system", "notes")]
    # Drop a column no model uses, so the table fits the page.
    words = [w for w in words if any(int(r[w]) for r in rows)]
    short = {"layernorm": "LN", "rmsnorm": "RMS", "softmax": "smax",
             "residual_add": "res", "embedding_gather": "gather",
             "reshape_transpose": "shape", "reduce": "red", "matmul": "mm",
             "rotary": "rope"}
    table = [[r["model"], SYSTEM_LABEL.get(r["system"], r["system"])] +
             [r[w] for w in words] for r in rows]
    path = os.path.join(args.outdir, "op_inventory.tex")
    write_table(path,
                "operators per forward, both systems in one vocabulary",
                ["model", "system"] + [short.get(w, w) for w in words], table,
                align="ll" + "r" * len(words))
    append_notes(path, op_inventory_notes(out))
    fig, ax = plt.subplots(figsize=(args.width, 0.3))
    ax.axis("off")
    return fig, "op_inventory"


def op_inventory_notes(out):
    """The mapping table and the per-model reasons, from the tsv's companion
    notes file, turned into LaTeX paragraphs."""
    path = os.path.join(out, "op_inventory_notes.txt")
    if not os.path.exists(path):
        return []
    tex = lambda s: (str(s).replace("\\", "").replace("_", r"\_")
                     .replace("%", r"\%").replace("&", r"\&")
                     .replace("#", r"\#"))
    maps, alias, notes, diffs = [], [], [], []
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if parts[0] == "map":
            maps.append(r"\texttt{%s}$\rightarrow$%s" % (tex(parts[1]),
                                                         tex(parts[2])))
        elif parts[0] == "alias":
            alias.append(r"\texttt{%s}" % tex(parts[1]))
        elif parts[0] == "note":
            notes.append(tex(parts[2]))
        elif parts[0] == "diff":
            diffs.append((parts[2], parts[1]))
    paras = []
    if maps:
        paras.append(r"\textbf{Mapping.} torch's rows are one eager forward "
                     r"under \texttt{torch.profiler}, with each aten name "
                     r"mapped to a vocabulary word: " + ", ".join(maps) + ".")
    if alias:
        paras.append(r"\textbf{Not counted.} These are the same call seen at "
                     r"an outer or inner dispatch level as one already "
                     r"counted, so counting them too would double-count: " +
                     ", ".join(alias) + ".")
    paras += [r"\textbf{Note.} " + n for n in notes]
    if diffs:
        # Grouped by reason, not by model: one cause explains the same cell in several models.
        by = collections.OrderedDict()
        for why, cell in diffs:
            by.setdefault(why, []).append(cell)
        paras.append(r"\textbf{Cells that differ.} Every cell where the two "
                     r"rows disagree, grouped by the one reason it does.")
        for why, cells in by.items():
            paras.append(r"\emph{%s} %s." % (tex(why), "; ".join(
                tex(c) for c in cells)))
    return paras


def _role(model):
    """Where a row stands under the selection rule: population, in it
    under the usage exemption, or out of it."""
    if model in PROBES:
        return "excluded"
    if model in C3_EXEMPT:
        return "population, C3 exemption"
    return "population"


WEIGHT_LABEL = {"trained": "trained",
                "head-random": "encoder trained, head random",
                "random": "randomly initialised fixture"}


def fig_model_provenance(out, args):
    """Table only: which checkpoint each row is, its snapshot, parameter count and
    weight provenance."""
    rows = [r for r in read_tsv(os.path.join(out, "model_inventory.tsv"))
            if r.get("system") == "ours" and r.get("hf_id")]
    if not rows:
        return None
    table = []
    for r in rows:
        repo, _, rev = r["revision"].rpartition("@")
        table.append([r["model"], r["hf_id"], rev,
                      si(int(r["params_only"])),
                      WEIGHT_LABEL.get(r["weights"], r["weights"]),
                      _role(r["model"])])
    path = os.path.join(args.outdir, "model_provenance.tex")
    write_table(path,
                "checkpoint, snapshot and weight provenance per configuration",
                ["model", "HuggingFace id", "snapshot", "params", "weights",
                 "role"], table, align="lllrll")
    fig, ax = plt.subplots(figsize=(args.width, 0.3))
    ax.axis("off")
    return fig, "model_provenance"


def fig_model_inventory(out, args):
    """Table only: what each model computes, ours against torch, with remaining
    differences named in the notes column."""
    rows = read_tsv(os.path.join(out, "model_inventory.tsv"))
    if not rows:
        return None
    table = [[r["model"], SYSTEM_LABEL.get(r["system"], r["system"]),
              si(int(r["params_only"])),
              ("+" + si(int(r["params_plus_buffers"]) -
                        int(r["params_only"])))
              if int(r["params_plus_buffers"]) > int(r["params_only"])
              else "--",
              r["gflops"], r["gemm_calls"],
              r["bmm_calls"], r["conv_calls"], r["sdpa_calls"],
              r["notes"]] for r in rows]
    path = os.path.join(args.outdir, "model_inventory.tex")
    write_table(path,
                "per forward: parameters, GEMM/bmm/conv FLOPs and the "
                "cuBLAS/cuDNN calls that carry them, with the remaining "
                "differences named",
                ["model", "system", "params", "buffers", "GFLOP", "GEMM",
                 "bmm", "conv", "SDPA", "notes"], table,
                align="ll" + "r" * 7 + "p{0.30\\textwidth}")
    append_notes(path, [
        r"\emph{params} counts the learned weights; \emph{buffers} is what "
        r"the registered non-learned tensors that are still part of the "
        r"trained model add on top, which in these nine models is the "
        r"batch-norm running mean and variance of ResNet-18 (9600 values) "
        r"and nothing else. Both sides are counted the same way and agree "
        r"exactly on both columns for ResNet-18.",
        r"Excluded from both columns on both sides, as inputs or as tables "
        r"rebuilt from the config rather than anything training produced: "
        r"ours \texttt{image}, \texttt{rope.cos}, \texttt{rope.sin}, "
        r"\texttt{rope.p}; torch \texttt{position\_ids}, "
        r"\texttt{token\_type\_ids}, \texttt{inv\_freq}, "
        r"\texttt{original\_inv\_freq}, \texttt{num\_batches\_tracked} "
        r"(a counter, not data), and a module's causal-mask buffer where it "
        r"keeps one. The input token ids live in the config, not in the "
        r"checkpoint index, so they are outside our count by construction.",
    ])
    fig, ax = plt.subplots(figsize=(args.width, 0.3))
    ax.axis("off")
    return fig, "model_inventory"


DEOPT_PATTERNS = ["never", "alternate", "both-hot", "fresh", "probe-a",
                  "probe-e"]
DEOPT_LABEL = {
    "never": "guard holds",
    "alternate": "guard fails every iter",
    "both-hot": "both paths hot",
    "fresh": "fresh value every iter",
    "probe-a": "probe (a) shared branch",
    "probe-e": "probe (e) shape change",
}


TRANSITION_ARM = {"direct": ("#1a4fa0", "", "o"),
                  "drain": ("#8a5cd6", "\\\\", "v")}
TRANSITION_LABEL = {"direct": "descriptor kept",
                    "drain": "drained to canonical, re-recorded"}
# The three ways the interior consumer can be present: never, always, and
# arriving after a guard.
READ_ORDER = ["none", "every", "half"]
READ_LABEL = {"none": "no interior\nconsumer",
              "every": "consumer from\nthe first step",
              "half": "consumer after\na guard fails"}


def fig_transition(out, args):
    """What a fused value costs to carry across a guard.

    Top: launches per iteration (structure, identical across rounds and cache
    states). Bottom: cumulative time per cache state, where the extra kernel's
    compile shows up.
    """
    phases = read_tsv(os.path.join(out, "transition_phases.tsv"))
    rows = read_tsv(os.path.join(out, "transition.tsv"))
    if not phases and not rows:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(args.width, 4.1),
                            gridspec_kw={"height_ratios": [1.0, 0.85]})

    ax = axes[0]
    per = {(r["arm"], r["interior_reads"]): float(r["launches_per_iter"])
           for r in phases}
    reads = [k for k in READ_ORDER if any((a, k) in per
                                          for a in TRANSITION_ARM)]
    x = list(range(len(reads)))
    width = 0.38
    if per:
        for i, arm in enumerate(("direct", "drain")):
            vals = [per.get((arm, k), 0.0) for k in reads]
            colour, hatch, _ = TRANSITION_ARM[arm]
            ax.bar([v + (i - 0.5) * width for v in x], vals, width * 0.9,
                   color=colour, linewidth=0,
                   hatch=hatch if args.texture else None)
            for xi, v in zip(x, vals):
                if v:
                    ax.text(xi + (i - 0.5) * width, v + 0.12, "%.2f" % v,
                            ha="center", fontsize=6)
        ax.set_xticks(x, [READ_LABEL.get(k, k) for k in reads])
        ax.set_ylim(0, max(per.values()) * 1.22)
    else:
        ax.set_xticks([])
        ax.set_ylim(0, 1)
        ax.text(0.5, 0.5, "launch counts not recorded in this run",
               ha="center", va="center", transform=ax.transAxes,
               fontsize=7, color=INK_2)
    ax.set_ylabel("kernel launches\nper iteration (lower is better)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    ax = axes[1]
    g = collections.defaultdict(list)
    for r in rows:
        try:
            g[(r["cache"], r["arm"])].append(float(r["total_ms"]))
        except (KeyError, ValueError):
            pass
    caches = [c for c in ("cold", "warm") if any(k[0] == c for k in g)]
    x = list(range(len(caches)))
    for i, arm in enumerate(("direct", "drain")):
        vals = [med(g.get((c, arm), [])) or 0.0 for c in caches]
        colour, hatch, _ = TRANSITION_ARM[arm]
        ax.bar([v + (i - 0.5) * width for v in x], vals, width * 0.9,
               color=colour, linewidth=0,
               hatch=hatch if args.texture else None)
        lo = [min(g.get((c, arm), [0])) for c in caches]
        hi = [max(g.get((c, arm), [0])) for c in caches]
        ax.errorbar([v + (i - 0.5) * width for v in x], vals,
                    yerr=err(vals, lo, hi), fmt="none", zorder=4, **ERRBAR)
    ax.set_xticks(x, ["cold kernel cache", "warm kernel cache"][:len(caches)])
    ax.set_yscale("log")
    ax.set_ylabel("cumulative ms over the run\n(log scale, lower is better)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    legend_handles = [Line2D([], [], color=TRANSITION_ARM[arm][0], lw=4,
                             label=TRANSITION_LABEL[arm])
                      for arm in ("direct", "drain")]
    legend_below(fig, handles=legend_handles,
                labels=[h.get_label() for h in legend_handles],
                ncol=1 if args.width < DOUBLE_COL else 2)
    if args.titles:
        fig.suptitle("Carrying a fused value across a guard")

    table = []
    for k in reads:
        row = [READ_LABEL.get(k, k).replace("\n", " ")]
        for arm in ("direct", "drain"):
            row.append("%.2f" % per[(arm, k)] if (arm, k) in per else "n/a")
        if ("direct", k) in per and ("drain", k) in per:
            row.append("%.2f" % (per[("drain", k)] / per[("direct", k)]))
        else:
            row.append("n/a")
        table.append(row)
    write_table(os.path.join(args.outdir, "transition.tex"),
                "kernel launches per iteration, 400 iterations, identical "
                "across rounds and cache states",
                ["interior consumer", "descriptor kept", "drained",
                 "drained / kept"], table)
    return fig, "transition"


MOTION_ARMS = ["derived", "handwritten", "direct", "drain"]
MOTION_LABEL = {"derived": "derived (rule specialised in place)",
                "handwritten": "canonicalised first",
                "direct": "hand-written adapter, descriptor kept",
                "drain": "hand-written adapter, drained"}
MOTION_CASE = {"axis": "vector axis", "layout": "blocked layout",
               "axis+scoped": "vector axis,\nno dead locals",
               "layout+scoped": "blocked layout,\nno dead locals"}
MOTION_CASE_ORDER = ["layout", "layout+scoped", "axis", "axis+scoped"]


def _motion_rows(out):
    rows = read_tsv(os.path.join(out, "motion.tsv"))
    for r in rows:
        r["cache"] = r.get("cache") or "cold"
    return rows


def _sign_interval(xs):
    s = sorted(xs)
    n = len(s)
    if n < 6:
        return (s[0], s[-1]) if s else (0.0, 0.0)
    k = 2 if n >= 10 else 1
    return s[k - 1], s[n - k]


def fig_motion(out, args):
    """The MOTION demonstration, one change class per group. Top: kernel launches
    per step after the transition. Bottom: step latency, medians with sign-test
    intervals over fresh processes."""
    rows = [r for r in _motion_rows(out) if r["cache"] == "cold"]
    if not rows:
        return None
    for r in rows:
        if r.get("scoped") == "1":
            r["case"] = r["case"] + "+scoped"
    cases = [c for c in MOTION_CASE_ORDER if any(r["case"] == c for r in rows)]
    arms = [a for a in MOTION_ARMS if any(r["arm"] == a for r in rows)]
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["case"], r["arm"])].append(r)
    fig, axes = plt.subplots(2, 1, figsize=(args.width, 4.3), sharex=True)
    width = 0.8 / max(len(arms), 1)
    x = list(range(len(cases)))
    table = []
    for panel, key, ylabel in ((0, "launches_after",
                                "kernel launches per step\nafter the change"),
                               (1, "step_us", "step latency (us)")):
        ax = axes[panel]
        for i, arm in enumerate(arms):
            vals, lo, hi = [], [], []
            for c in cases:
                xs = [float(r[key]) for r in g.get((c, arm), [])]
                m = med(xs) if xs else 0.0
                a, b = _sign_interval(xs) if xs else (0.0, 0.0)
                vals.append(m)
                lo.append(a)
                hi.append(b)
            pos = [v + (i - (len(arms) - 1) / 2.0) * width for v in x]
            ax.bar(pos, vals, width * 0.9, color=SERIES[i], linewidth=0,
                   hatch=HATCH[i] if args.texture else None,
                   label=MOTION_LABEL[arm])
            if panel == 1:
                ax.errorbar(pos, vals, yerr=err(vals, lo, hi), fmt="none",
                            zorder=4, **ERRBAR)
            for p_, v in zip(pos, vals):
                ax.text(p_, v * 1.02 + 0.02, ("%.2f" if panel == 0 else
                                              "%.0f") % v,
                        ha="center", va="bottom", fontsize=5.5)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
        despine(ax)
    axes[1].set_xticks(x, [MOTION_CASE[c] for c in cases])
    legend_below(fig, axes[1], ncol=1 if args.width < DOUBLE_COL else 2)
    if args.titles:
        fig.suptitle("A change rule derived, against hand-written recovery")
    for c in cases:
        for arm in arms:
            rs = g.get((c, arm), [])
            if not rs:
                continue
            st = [float(r["step_us"]) for r in rs]
            at = [float(r["at_us"]) for r in rs]
            a, b = _sign_interval(st)
            table.append([MOTION_CASE[c].replace("\n", " "), MOTION_LABEL[arm],
                          "%.2f" % med([float(r["launches_after"]) for r in rs]),
                          "%.1f [%.1f, %.1f]" % (med(st), a, b),
                          "%.0f" % med(at),
                          "%.0f" % med([float(r["retained_bytes"]) for r in rs]),
                          str(len(rs))])
    write_table(os.path.join(args.outdir, "motion.tex"),
                "the MOTION demonstration: medians over fresh processes, "
                "sign-test intervals (97.9%% at n=10), cold kernel cache",
                ["change class", "arm", "launches/step", "step us",
                 "transition us", "retained bytes", "n"], table)
    return fig, "motion"


def fig_motion_retention(out, args):
    """Device bytes still held at the end of a run, against run length."""
    rows = [r for r in _motion_rows(out)
            if r["cache"] == "warm" and r.get("scoped", "0") != "1"]
    if not rows:
        return None
    cases = [c for c in MOTION_CASE_ORDER if any(r["case"] == c for r in rows)]
    fig, axes = plt.subplots(len(cases), 1,
                             figsize=(args.width, 2.1 * len(cases) + 0.5),
                             squeeze=False)
    for ci, c in enumerate(cases):
        ax = axes[ci][0]
        for i, arm in enumerate(MOTION_ARMS):
            pts = collections.defaultdict(list)
            for r in rows:
                if r["case"] == c and r["arm"] == arm:
                    pts[int(r["steps"])].append(float(r["retained_bytes"]))
            if not pts:
                continue
            ls = sorted(pts)
            ax.plot(ls, [med(pts[L]) / 1e6 for L in ls], marker=MARKERS[i],
                    color=SERIES[i], markersize=3.5, linewidth=1.2,
                    label=MOTION_LABEL[arm] if ci == 0 else None)
        ax.set_xscale("log")
        ax.set_ylabel("retained MB\n(%s)" % MOTION_CASE[c])
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    axes[-1][0].set_xlabel("steps in the run (change at the midpoint)")
    legend_below(fig, axes[-1][0], ncol=1 if args.width < DOUBLE_COL else 2)
    return fig, "motion_retention"


DECODE_SYSTEMS = ["ours", "torch-eager", "torch-compile", "torch-compile-ro"]


def fig_decode(out, args):
    """Next-token latency with a KV cache, per model, medians with sign-test
    intervals over fresh processes, one model per row."""
    rows = read_tsv(os.path.join(out, "decode.tsv"))
    if not rows:
        return None
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    systems = [s for s in DECODE_SYSTEMS if any(r["system"] == s for r in rows)]
    g = collections.defaultdict(list)
    for r in rows:
        try:
            g[(r["model"], r["system"])].append(float(r["token_us"]))
        except ValueError:
            pass
    fig, ax = plt.subplots(figsize=(args.width, 0.55 + 0.62 * len(models)))
    height = 0.8 / max(len(systems), 1)
    table = []
    for i, sysname in enumerate(systems):
        vals, lo, hi, ys = [], [], [], []
        for mi, m in enumerate(models):
            xs = g.get((m, sysname), [])
            v = med(xs) / 1e3 if xs else 0.0
            a, b = _sign_interval(xs) if xs else (0.0, 0.0)
            vals.append(v)
            lo.append(a / 1e3)
            hi.append(b / 1e3)
            ys.append(mi + (i - (len(systems) - 1) / 2.0) * height)
        colour, hatch, _ = style_of(sysname)
        ax.barh(ys, vals, height * 0.9, color=colour, linewidth=0,
                hatch=hatch if args.texture else None,
                label=SYSTEM_LABEL.get(sysname, sysname))
        ax.errorbar(vals, ys, xerr=err(vals, lo, hi), fmt="none", zorder=4,
                    **ERRBAR)
        for y, v, h in zip(ys, vals, hi):
            if v:
                ax.text(max(v, h) * 1.02 + 0.05, y, "%.2f" % v, va="center",
                        fontsize=5.5)
    ax.set_yticks(range(len(models)), models)
    ax.invert_yaxis()
    ax.set_xlabel("next-token latency, ms (lower is better)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(0, ax.get_xlim()[1] * 1.12)
    despine(ax)
    legend_below(fig, ax, ncol=1 if args.width < DOUBLE_COL else 2)
    for m in models:
        base = med(g.get((m, "ours"), [])) if g.get((m, "ours")) else 0.0
        for sysname in systems:
            xs = g.get((m, sysname), [])
            if not xs:
                continue
            a, b = _sign_interval(xs)
            rs = [r for r in rows if r["model"] == m and r["system"] == sysname]
            table.append([m, SYSTEM_LABEL.get(sysname, sysname),
                          "%.0f [%.0f, %.0f]" % (med(xs), a, b),
                          "%.2f" % (med(xs) / base) if base else "n/a",
                          "%s" % (rs[0].get("launches_per_token") or "-"),
                          "%d/%d" % (sum(r["tokens_match"] == "1" for r in rs),
                                     len(rs))])
    write_table(os.path.join(args.outdir, "decode.tex"),
                "next-token latency (us), greedy decode with a key/value "
                "cache, medians and sign-test intervals over fresh processes",
                ["model", "system", "token us", "vs ours", "launches/token",
                 "tokens match"], table)
    return fig, "decode"


def fig_deopt(out, args):
    """Two panels, same rows. Left: steady-state cost, i.e. whether a bridge
    recovered. Right: the most expensive single iteration (the deoptimization
    itself), which spans four orders of magnitude, so it is markers on a log axis."""
    rows = read_tsv(os.path.join(out, "deopt.tsv"))
    if not rows:
        return None
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for col in ("steady_us", "first_fail_us", "after_fail_us", "peak_us",
                    "launches_per_iter", "loops", "bridges", "cold_us"):
            v = r.get(col)
            if v not in (None, ""):
                g[(r["pattern"], r["system"])][col].append(float(v))
    patterns = [p for p in DEOPT_PATTERNS if any(k[0] == p for k in g)]
    systems = drawn([s for s in ("ours", "torch-eager", "torch-compile",
                                 "torch-compile-ro", "torch-compile-nobreak",
                                 "jax-host", "jax-static", "jax-cond")
                     if any(k[1] == s for k in g)], args)
    if not patterns:
        return None

    if args.width >= DOUBLE_COL:
        fig, (ax, ax2) = plt.subplots(
            1, 2, figsize=(args.width, 0.58 * len(patterns) + 1.2),
            gridspec_kw={"width_ratios": [1.25, 1]})
    else:
        # One column is too narrow for two panels beside the pattern names, so they stack.
        fig, (ax, ax2) = plt.subplots(
            2, 1, figsize=(args.width, 0.66 * len(patterns) + 1.5),
            sharey=True)
    height = 0.8 / len(systems)
    for si, system in enumerate(systems):
        colour, hatch, marker = style_of(system)
        ys, xs, lows, highs, pys, pxs = [], [], [], [], [], []
        for i, pattern in enumerate(patterns):
            d = g.get((pattern, system))
            if not d:
                continue
            y = i + ((len(systems) - 1) / 2.0 - si) * height
            m, lo, hi = band(d.get("steady_us", []))
            if m is not None:
                ys.append(y)
                xs.append(m)
                lows.append(lo)
                highs.append(hi)
            pm, _, _ = band(d.get("peak_us", []))
            if pm is not None:
                pys.append(y)
                pxs.append(pm)
        ax.barh(ys, xs, height=height * 0.85, color=colour, linewidth=0,
                label=SYSTEM_LABEL.get(system, system),
                hatch=hatch if args.texture else None)
        if xs:
            ax.errorbar(xs, ys, xerr=err(xs, lows, highs), fmt="none",
                        zorder=4, **ERRBAR)
        ax2.plot(pxs, pys, marker, color=colour, linestyle="none",
                 markersize=4, markeredgewidth=0)
    ticks = [DEOPT_LABEL.get(p, p) for p in patterns]
    ax.set_yticks(range(len(patterns)), ticks)
    ax.set_xscale("log")
    ax.set_xlabel("steady-state per iteration (\u00b5s, log scale), lower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.6, len(patterns) - 0.4)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    ax2.set_xscale("log")
    if args.width >= DOUBLE_COL:
        ax2.set_yticks(range(len(patterns)), ["" for _ in patterns])
    else:
        ax2.set_yticks(range(len(patterns)), ticks)
    ax2.set_xlabel("most expensive iteration (\u00b5s, log scale)\n"
                   "lower is better")
    ax2.xaxis.grid(True, zorder=0)
    ax2.set_axisbelow(True)
    ax2.set_ylim(-0.6, len(patterns) - 0.4)
    despine(ax2, keep=("left",))
    ax2.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("Cost of a guard failure")

    def cell(d, col, fmt="%.0f"):
        m, _, _ = band(d.get(col, []))
        return fmt % m if m is not None and m >= 0 else "n/a"

    table = []
    for pattern in patterns:
        for system in ("ours", "torch-eager", "torch-compile",
                       "torch-compile-ro", "torch-compile-nobreak",
                       "jax-host", "jax-static", "jax-cond"):
            d = g.get((pattern, system))
            if not d:
                continue
            table.append([SYSTEM_LABEL.get(system, system),
                          DEOPT_LABEL.get(pattern, pattern),
                          cell(d, "steady_us", "%.1f"),
                          cell(d, "first_fail_us"), cell(d, "after_fail_us"),
                          cell(d, "peak_us"), cell(d, "cold_us"),
                          cell(d, "launches_per_iter", "%.2f"),
                          cell(d, "loops"), cell(d, "bridges")])
    write_table(os.path.join(args.outdir, "deopt.tex"),
                "median over the rounds; first/after/peak are single "
                "iterations, peak is the deoptimization itself",
                ["system", "pattern", "steady us", "first fail", "after fail",
                 "peak", "cold", "launch/it", "loops", "bridges"], table)
    return fig, "deopt"


def fig_micro_baselines(out, args):
    """Every backend against the same fused baseline: grouped diverging bars,
    one row per point, same log-ratio convention as fig_micro_speedup."""
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    systems = ["torch-eager", "torch-compile", "torch-compile-ro", "torch-compile-mat",
               "jax", "iree", "triton", "torch-tensorrt"]
    pts = []
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused"):
            continue
        r = None
        if m.get("torch-compile"):
            r, _, _ = ratio_band(m["torch-compile"], m["fused"])
        pts.append((r, micro_label(v, k, n), m))
    if not pts:
        return None
    pts.sort(key=lambda p: (p[0] is None, p[0]))
    labels = [p[1] for p in pts]
    used = [s for s in systems if any(m.get(s) for _, _, m in pts)]
    if not used:
        return None

    # One marker per system at its ratio to MetaTensor, as in fig_micro_speedup.
    # Grouped bars for 28 points x 6 systems made the figure too tall.
    fig, ax = plt.subplots(figsize=(args.width, 0.16 * len(pts) + 1.25))
    y = list(range(len(pts)))
    for si_, s in enumerate(drawn(used, args)):
        ys, xs = [], []
        for i, (_, _, m) in enumerate(pts):
            fm = med(m.get("fused", []))
            sm = med(m.get(s, []))
            if fm and sm:
                ys.append(i)
                xs.append(sm / fm)
        if not xs:
            continue
        color, _, marker = style_of(s)
        ax.scatter(xs, ys, s=20, marker=marker,
                   facecolors=color, edgecolors="white",
                   linewidth=0.5, zorder=4, label=SYSTEM_LABEL.get(s, s))
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "×"))
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, len(pts) - 0.4)
    ax.set_xlabel("time relative to MetaTensor\n"
                  "log2 scale; right of 1x is slower")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Baselines relative to MetaTensor")

    header = ["benchmark", "MetaTensor (us)", "eager", "torch.compile"]
    if "jax" in used:
        header.append("JAX/XLA")
    if "iree" in used:
        header.append("IREE")
    if "triton" in used:
        header.append("Triton")
    header.append("MT/Triton")
    table_rows = []
    for _, lab, m in pts:
        fm = med(m.get("fused", []))
        te = med(m.get("torch-eager", []))
        tc = med(m.get("torch-compile", []))
        row = [lab, "%.0f" % fm if fm else "n/a",
               "%.0f" % te if te else "n/a", "%.0f" % tc if tc else "n/a"]
        if "jax" in used:
            jv = med(m.get("jax", []))
            row.append("%.0f" % jv if jv else "n/a")
        if "iree" in used:
            iv = med(m.get("iree", []))
            row.append("%.0f" % iv if iv else "n/a")
        tv = med(m.get("triton", []))
        if "triton" in used:
            row.append("%.0f" % tv if tv else "n/a")
        row.append("%.2f" % (fm / tv) if (fm and tv) else "n/a")
        table_rows.append(row)
    write_table(os.path.join(args.outdir, "micro_baselines.tex"),
                "median us per iteration, MetaTensor and baselines, float64",
                header, table_rows)
    return fig, "micro_baselines"


def fig_compile_overhead(out, args):
    """Compile and first-run cost per backend, and the break-even iteration
    count against eager, read from results.jsonl."""
    records = read_jsonl(os.path.join(out, "results.jsonl"))
    if not records:
        return None
    systems = ["ours", "torch-compile", "torch-compile-ro", "torch-compile-mat",
               "torch-tensorrt", "jax", "iree", "triton"]

    def workload_key(r):
        if r.get("kind") == "micro" and r.get("dtype") == "float64":
            return ("micro", r.get("variant"), r.get("k"), r.get("n"))
        if r.get("kind") == "models":
            return ("models", r.get("model"))
        return None

    def workload_label(key):
        if key[0] == "micro":
            return micro_label(key[1], key[2], key[3])
        return key[1]

    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in records:
        key = workload_key(r)
        if key is None:
            continue
        sysname = r.get("mode") if key[0] == "micro" else r.get("system")
        if not sysname:
            continue
        g[key][sysname].append(r)

    def medf(rs, field):
        vals = [r[field] for r in rs
                if isinstance(r.get(field), (int, float)) and r[field] >= 0]
        return statistics.median(vals) if vals else None

    rows = []
    for key in g:
        eager_rs = g[key].get("torch-eager", [])
        eager_first = medf(eager_rs, "first_run_ms")
        eager_steady = medf(eager_rs, "steady_us")
        for s in systems:
            rs = g[key].get(s)
            if not rs:
                continue
            compile_ms = medf(rs, "compile_ms")
            first_ms = medf(rs, "first_run_ms")
            steady = medf(rs, "steady_us")
            be = "n/a"
            if (first_ms is not None and eager_first is not None
                    and eager_steady is not None and steady is not None
                    and eager_steady - steady > 0):
                # compile_ms is paid before the first run where it is
                # reported; torch.compile folds it into first_run_ms.
                be = "%.0f" % ((first_ms + (compile_ms or 0) - eager_first)
                                * 1000 / (eager_steady - steady))
            rows.append((key, s, compile_ms, first_ms, steady, be))
    if not rows:
        return None

    write_table(os.path.join(args.outdir, "compile_overhead.tex"),
                "median compile / first-run cost and break-even iteration "
                "count against torch eager",
                ["workload", "system", "compile (ms)", "first run (ms)",
                 "steady (us)", "break-even (iters)"],
                [[workload_label(key), SYSTEM_LABEL.get(s, s),
                  "%.0f" % c if (c is not None and c >= 0) else "n/a",
                  "%.0f" % f if (f is not None and f >= 0) else "n/a",
                  "%.0f" % st if st is not None else "n/a", be]
                 for key, s, c, f, st, be in rows])

    # Model workloads only. micro_speedup and micro_baselines cover the micro points.
    model_keys = [key for key in dict.fromkeys(k for k, s, c, f, st, be in rows)
                  if key[0] == "models"]
    fallback = not model_keys
    if fallback:
        # No model rows in this run: fall back to the micro rows, capped to
        # the largest workloads so the figure still fits a page.
        keys_order = list(dict.fromkeys(k for k, s, c, f, st, be in rows))
        best = {}
        for key, s, c, f, st, be in rows:
            if f is not None and f > 0:
                best[key] = max(best.get(key, 0), f)
        keys_order = sorted(keys_order, key=lambda k: -best.get(k, 0))[:12]
    else:
        keys_order = model_keys
    plot_rows = [(key, s, f) for key, s, c, f, st, be in rows
                 if f is not None and f > 0 and key in keys_order]
    systems_used = [s for s in drawn(systems, args)
                    if any(pr[1] == s for pr in plot_rows)]
    if not plot_rows or not systems_used:
        return None

    h = 0.8 / len(systems_used)
    fig, ax = plt.subplots(figsize=(args.width, 0.42 * len(keys_order) + 1.3))
    y = list(range(len(keys_order)))
    for si_, s in enumerate(systems_used):
        off = ((len(systems_used) - 1) / 2 - si_) * h
        ys, xs = [], []
        for i, key in enumerate(keys_order):
            match = [f for k, ss, f in plot_rows if k == key and ss == s]
            if match:
                ys.append(i + off)
                xs.append(match[0])
        if not xs:
            continue
        ax.barh(ys, xs, height=h * 0.86, color=style_of(s)[0],
                linewidth=0,
                hatch=style_of(s)[1] if args.texture else None,
                label=SYSTEM_LABEL.get(s, s))
    ax.set_xscale("log")
    ax.set_yticks(y, [workload_label(k) for k in keys_order])
    ax.set_xlabel("first-run latency (ms, log scale)\nlower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Compile / first-run overhead")
    return fig, "compile_overhead"


# --- machines --------------------------------------------------------------


def fig_gap(out, args):
    """Two panels in different units against a shared model list: kernels launched
    per forward and the share of wall clock they occupy."""
    write_gemm_count(out, args)
    rows = read_tsv(os.path.join(out, "gap.tsv"))
    rows = [r for r in rows if r.get("launches_per_iter")]
    if not rows:
        return None
    g = collections.defaultdict(dict)
    for r in rows:
        g[r["model"]][r["system"]] = r
    systems = [s for s in ["ours", "torch-compile", "jax"]
               if any(s in d for d in g.values())]
    models = sorted(g, key=lambda m: float(g[m].get("ours", {}).get("steady_us") or 0))

    def col(sys_, model, key, default=0.0):
        v = g[model].get(sys_, {}).get(key)
        return float(v) if v else default

    wide = args.width >= DOUBLE_COL
    if wide:
        fig, axes = plt.subplots(1, 2, sharey=True,
                                 figsize=(args.width,
                                          0.42 * len(models) + 1.2))
    else:
        fig, axes = plt.subplots(2, 1, sharey=True,
                                 figsize=(args.width,
                                          0.60 * len(models) + 1.3))
    h = 0.8 / len(systems)
    y = list(range(len(models)))
    panels = [(axes[0], "launches_per_iter",
               "kernel launches per forward, lower is better"),
              (axes[1], "gpu_util", "GPU busy / wall time, higher is better")]
    for ax, key, xlabel in panels:
        for si_, s in enumerate(systems):
            off = ((len(systems) - 1) / 2 - si_) * h
            colour, hatch, _ = style_of(s)
            ax.barh([v + off for v in y], [col(s, m, key) for m in models],
                    height=h * 0.88, color=colour,
                    linewidth=0,
                    hatch=hatch if args.texture else None,
                    label=SYSTEM_LABEL[s])
        ax.set_xlabel(xlabel)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax, keep=("left",))
        ax.tick_params(axis="y", length=0)
    util_max = max([1.0] + [col(s, m, "gpu_util") for m in models for s in systems])
    axes[1].axvline(1, color=INK_2, linewidth=0.8, zorder=1)
    axes[1].set_xlim(0, util_max * 1.05)
    axes[0].set_yticks(y, models)
    if not wide:
        axes[1].set_yticks(y, models)
    legend_below(fig, axes[0], ncol=3)
    if args.titles:
        fig.suptitle("Where the gap comes from")

    header = ["model", "system", "us/iter", "launches", "kernels", "GPU us",
              "GPU util", "gemm", "split-K", "bmm", "gen", "copy", "other"]
    table_rows = []
    for m in models:
        for s in systems:
            r = g[m].get(s)
            if not r:
                continue
            mix = gap_mix(r)
            table_rows.append([m, SYSTEM_LABEL[s], r["steady_us"],
                               r["launches_per_iter"], r["kernels"],
                               r["gpu_busy_us"], r["gpu_util"]] +
                              [gap_num(mix[k])
                               for k in ("gemm", "splitk", "bmm", "gen",
                                         "copy", "other")])
    write_table(os.path.join(args.outdir, "gap.tex"),
                "per forward: nsys launch counts and GPU busy time against the "
                "measured wall time, and the launches attributed kernel by "
                "kernel - cuBLAS GEMMs, the cuBLASLt split-K reduce/epilogue "
                "kernels that finish some of them, generated "
                "(Triton/Inductor/XLA) kernels, copies, everything else; the "
                "five classes sum to the launch count (checked before "
                "rounding - these are per-forward averages over a differenced "
                "trace window and are fractional when cuBLAS does not pick "
                "the same kernel every iteration), and bmm is the batched "
                "(attention) subset of gemm, attributed by launch "
                "multiplicity - two products per layer, so the GEMM kernel "
                "launched 2L times on an L-layer model - and name-based where "
                "the kernel name says batched or attention",
                header, table_rows)
    return fig, "gap"


def gap_mix(r):
    """A gap.tsv row's launch attribution, as counts.

    Older rows carry the classes only in the free-text `notes` mix ("gemm=43
    gen=24"), parsed as a fallback. `bmm` exists only in newer rows.
    """
    mix = {}
    for part in (r.get("notes") or "").split():
        key, sep, value = part.partition("=")
        if sep and value.isdigit():
            mix[key] = int(value)
    out = {}
    for key in ("gemm", "splitk", "gen", "copy", "other", "bmm"):
        v = r.get(key)
        out[key] = float(v) if v not in (None, "") else float(mix.get(key, 0))
    return out


def gap_num(v):
    """A class count as printed. Per-forward averages can be fractional when
    cuBLAS picks a different kernel on some iterations; the caption's sum check
    uses the unrounded values."""
    return "%.1f" % v


def gap_rows(out):
    """gap.tsv for this run. `bench.sh gap` needs nsys; without it there is no
    gap.tsv and the table is skipped."""
    path = os.path.join(out, "gap.tsv")
    return [r for r in read_tsv(path) if r.get("launches_per_iter")]


def write_gemm_count(out, args):
    """figures/gemm_count.tex: how a forward's launches split into matrix
    products, generated kernels and everything else.

    The split comes from gap.tsv's mix column ("gemm=43 gen=24 copy=6 other=2"),
    attributed kernel by kernel by gap_analysis from the nsys trace."""
    rows = gap_rows(out)
    # gap.tsv is appended to, so the last row for a (model, system) is current
    # (fig_gap reads it the same way).
    latest = collections.OrderedDict()
    for r in rows:
        latest[(r["model"], r["system"])] = r
    header = ["model", "system", "gemm", "split-K", "bmm", "gen", "copy",
              "other", "total", "launches"]
    table = []
    for r in latest.values():
        mix = gap_mix(r)
        if not any(mix.values()):
            continue
        total = sum(mix[k] for k in ("gemm", "splitk", "gen", "copy", "other"))
        table.append([r["model"], SYSTEM_LABEL.get(r["system"], r["system"]),
                      gap_num(mix["gemm"]), gap_num(mix["splitk"]),
                      gap_num(mix["bmm"]), gap_num(mix["gen"]),
                      gap_num(mix["copy"]), gap_num(mix["other"]),
                      gap_num(total), r.get("launches_per_iter") or "-"])
    if not table:
        return
    caption = ("kernel launches per forward, attributed kernel by kernel: "
               "matrix products, the cuBLAS split-K reduce/epilogue kernels "
               "that finish some of them (kernels, not calls; a run traced "
               "before gap\\_analysis separated them reports 0 and folds them "
               "into gemm), the batched attention products inside gemm "
               "(bmm), generated elementwise/reduction kernels, copies, "
               "everything else. total is the five classes added up and "
               "equals the measured launch count, checked before rounding: "
               "the counts are per-forward averages over a differenced trace "
               "window and are fractional where cuBLAS does not pick the same "
               "kernel every iteration. bmm is a subset of gemm and is not "
               "added in; attention products are attributed by launch "
               "multiplicity, two per layer, so the GEMM kernel launched 2L "
               "times per forward on an L-layer model is the attention pair "
               "whatever cuBLAS named it, and name-based where the kernel "
               "name says batched or attention. gap\\_kernels.tsv is where a "
               "forward's launches are attributed kernel by kernel")
    write_table(os.path.join(args.outdir, "gemm_count.tex"), caption,
                header, table)


def machine_label(path):
    """The accelerator, never the host (a published figure should not name a
    server). Runs without machine.txt fall back to the directory."""
    gpu = None
    txt = os.path.join(path, "machine.txt")
    if os.path.exists(txt):
        for line in open(txt):
            key, _, value = line.partition(" ")
            if key == "gpu" and value.strip():
                gpu = value.strip()
    if not gpu:
        parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if "-" in parent:
            gpu = parent.partition("-")[2]
    return pretty_gpu(gpu or "unknown GPU")


def machine_labels(paths):
    """Distinct labels without falling back to a host; same-model runs are
    separated by their date."""
    labels = [machine_label(p) for p in paths]
    if len(set(labels)) == len(labels):
        return labels
    out = []
    for path, label in zip(paths, labels):
        if labels.count(label) > 1:
            out.append("%s, %s" % (label, run_date(path)))
        else:
            out.append(label)
    return out


def run_date(path):
    txt = os.path.join(path, "machine.txt")
    if os.path.exists(txt):
        for line in open(txt):
            key, _, value = line.partition(" ")
            if key == "date" and value.strip():
                return value.strip()[:10]
    base = os.path.basename(os.path.abspath(path))
    m = re.search(r"\d{4}-\d{2}-\d{2}", base)
    return m.group(0) if m else base


def pretty_gpu(name):
    name = name.replace("NVIDIA ", "").replace("GeForce ", "").strip()
    m = re.match(r"^rtx(\d{3,4})(ti|super)?$", name, re.I)
    if m:
        return "RTX %s%s" % (m.group(1), " Ti" if (m.group(2) or "").lower() == "ti"
                             else " SUPER" if m.group(2) else "")
    return name


def _speedup_rows(out):
    g = micro_values(read_tsv(os.path.join(out, "micro.tsv")))
    rows = {}
    for (dtype, v, k, n), m in g.items():
        if dtype != "float64" or not m.get("fused") or not m.get("torch-compile"):
            continue
        rows[(v, k, n)] = (micro_label(v, k, n),) + \
            ratio_band(m["torch-compile"], m["fused"])
    return rows


def _grouped_ratio_bars(ax, keys, per_machine, labels, args, xlabel, bands=None):
    y = list(range(len(keys)))
    h = 0.8 / len(labels)
    for mi, label in enumerate(labels):
        vals = [per_machine[mi][k] for k in keys]
        off = ((len(labels) - 1) / 2 - mi) * h
        ax.barh([i + off for i in y], [v - 1 for v in vals], left=1,
                height=h * 0.86, color=MACHINE_COLORS[mi], linewidth=0,
                hatch=MACHINE_HATCH[mi] if args.texture else None,
                label=label)
        if bands:
            ax.errorbar(vals, [i + off for i in y],
                        xerr=err(vals, [bands[mi][k][0] for k in keys],
                                 [bands[mi][k][1] for k in keys]),
                        fmt="none", zorder=4, **ERRBAR)
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "\u00d7"))
    ax.set_xlabel(xlabel)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)


def fig_compare_speedup(out, args):
    """The same speedup on each machine, to show whether the ordering holds."""
    outs = [out] + args.compare
    rows = [_speedup_rows(o) for o in outs]
    labels = machine_labels(outs)
    common = set(rows[0])
    for r in rows[1:]:
        common &= set(r)
    dropped = sorted(set().union(*(set(r) for r in rows)) - common)
    if not common:
        return None
    keys = sorted(common, key=lambda k: rows[0][k][1])
    per = [{k: r[k][1] for k in keys} for r in rows]
    bands = [{k: (r[k][2], r[k][3]) for k in keys} for r in rows]

    fig, ax = plt.subplots(figsize=(args.width, 0.20 * len(keys) + 1.2))
    _grouped_ratio_bars(ax, keys, per, labels, args,
                        "speedup over torch.compile\n"
                        "log2 scale; higher is better", bands)
    ax.set_yticks(list(range(len(keys))), [rows[0][k][0] for k in keys])
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("Microbenchmark speedup across machines")
    if dropped:
        print("compare_speedup: %d point(s) not measured at the same size on "
              "every machine, left out: %s"
              % (len(dropped), ", ".join(micro_label(*k) for k in dropped)),
              file=sys.stderr)
    write_table(os.path.join(args.outdir, "compare_speedup.tex"),
                "fused speedup over torch.compile, float64, with the range "
                "over the rounds",
                ["benchmark"] + [c for lab in labels for c in (lab, "range")],
                [[rows[0][k][0]]
                 + [c for p, b in zip(per, bands)
                    for c in ("%.2f" % p[k], spread_str(*b[k], fmt="%.2f"))]
                 for k in keys])
    return fig, "compare_speedup"


def fig_compare_models(out, args):
    """End-to-end ratio to torch.compile, per model, per machine."""
    outs = [out] + args.compare
    labels = machine_labels(outs)
    per, bands = [], []
    for o in outs:
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in read_tsv(os.path.join(o, "models.tsv")):
            g[r["model"]][r["system"]].append(float(r["steady_us"]))
        usable = {m: ratio_band(d["torch-compile"], d["ours"]) for m, d in g.items()
                  if d.get("ours") and d.get("torch-compile")}
        per.append({m: b[0] for m, b in usable.items()})
        bands.append({m: (b[1], b[2]) for m, b in usable.items()})
    common = set(per[0])
    for p in per[1:]:
        common &= set(p)
    if not common:
        return None
    keys = sorted(common, key=lambda m: per[0][m])

    fig, ax = plt.subplots(figsize=(args.width, 0.30 * len(keys) + 1.25))
    _grouped_ratio_bars(ax, keys, per, labels, args,
                        "speedup over torch.compile\n"
                        "log2 scale; higher is better", bands)
    ax.set_yticks(list(range(len(keys))), keys)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("End-to-end speedup across machines")
    write_table(os.path.join(args.outdir, "compare_models.tex"),
                "end-to-end speedup over torch.compile, with the range over "
                "the rounds",
                ["model"] + [c for lab in labels for c in (lab, "range")],
                [[m] + [c for p, b in zip(per, bands)
                        for c in ("%.2f" % p[m], spread_str(*b[m], fmt="%.2f"))]
                 for m in keys])
    return fig, "compare_models"


def fig_compare_dynamic(out, args):
    """Cost against sequence length, as a ratio so two GPUs share one axis."""
    outs = [out] + args.compare
    labels = machine_labels(outs)
    series, lengths = [], None
    for o in outs:
        g = collections.defaultdict(list)
        for r in read_tsv(os.path.join(o, "dynamic_summary.tsv")):
            g[(r["system"], int(r["length"]))].append(float(r["median_us"]))
        if not g:
            return None
        ls = sorted({k[1] for k in g})
        lengths = ls if lengths is None else [L for L in lengths if L in ls]
        series.append(g)
    if not lengths:
        return None

    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.72))
    markers = ["o", "s", "^", "D"]
    for mi, (g, label) in enumerate(zip(series, labels)):
        ys = []
        for L in lengths:
            ours = med(g.get(("ours", L), []))
            comp = med(g.get(("torch-compile-dynamic", L), []))
            ys.append(comp / ours if ours and comp else float("nan"))
        ax.plot(lengths, ys, color=MACHINE_COLORS[mi], linewidth=1.4,
                marker=markers[mi], markersize=3.4, markeredgewidth=0,
                label=label, clip_on=False)
    ax.axhline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xticks(lengths)
    ax.set_xlim(lengths[0] - 3, lengths[-1] + 3)
    ax.set_xlabel("sequence length (tokens)")
    ax.set_ylabel("speedup over torch.compile (\u00d7)\nhigher is better")
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v) + "\u00d7"))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("Dynamic sequence length across machines")
    write_table(os.path.join(args.outdir, "compare_dynamic.tex"),
                "speedup over torch.compile (dynamic) per sequence length",
                ["machine"] + [str(L) for L in lengths],
                [[label] + ["%.2f" % (med(g.get(("torch-compile-dynamic", L), []))
                                      / med(g.get(("ours", L), [])))
                            for L in lengths]
                 for g, label in zip(series, labels)])
    return fig, "compare_dynamic"


def fig_integration(out, args):
    """Language-integration cost: the five control-flow micro variants (guard,
    force, value guard, exception, side effect) at k=4 n=1M float64, one row
    per system. torch's graphs/breaks sit in the launches_per_iter/graphs tsv
    columns (see the README note on micro.tsv's shifted torch columns)."""
    rows = read_tsv(os.path.join(out, "micro.tsv"))
    variants, k, n = [1, 2, 3, 4, 5], 4, 1000000
    dtype_of = lambda r: r.get("breaks") or r.get("graphs") or "float64"
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        try:
            v, rk, rn = int(r["variant"]), int(r["k"]), int(r["n"])
        except (KeyError, ValueError, TypeError):
            continue
        if v in variants and rk == k and rn == n and dtype_of(r) == "float64":
            g[v][r["mode"]].append(r)
    if not g:
        return None

    def fnum(recs, field):
        xs = [float(x[field]) for x in recs
              if x.get(field) not in (None, "", "-1")]
        return statistics.median(xs) if xs else None

    table, bars = [], []
    for v in variants:
        m = g.get(v, {})
        ours_us = fnum(m.get("fused", []), "steady_us")
        ours_launches = fnum(m.get("fused", []), "launches_per_iter")
        tc_us = fnum(m.get("torch-compile", []), "steady_us")
        tc_graphs = fnum(m.get("torch-compile", []), "launches_per_iter")
        tc_breaks = fnum(m.get("torch-compile", []), "graphs")
        eager_us = fnum(m.get("torch-eager", []), "steady_us")
        jax_us = fnum(m.get("jax", []), "steady_us")
        ratio = tc_us / ours_us if tc_us and ours_us else None
        label = VARIANT_LABEL.get(v, "v%d" % v)
        table.append([
            label,
            "%.1f" % ours_us if ours_us else "n/a",
            "%.2f" % ours_launches if ours_launches is not None else "n/a",
            "%.1f" % tc_us if tc_us else "n/a",
            "%d" % tc_graphs if tc_graphs is not None else "n/a",
            "%d" % tc_breaks if tc_breaks is not None else "n/a",
            "%.1f" % eager_us if eager_us else "n/a",
            "%.1f" % jax_us if jax_us else "n/a",
            "%.2f" % ratio if ratio else "n/a",
        ])
        bars.append((label, {"ours": ours_us, "torch-compile": tc_us,
                             "torch-eager": eager_us, "jax": jax_us}))

    systems = [s for s in ["ours", "torch-compile", "torch-eager", "jax"]
               if any(b[1].get(s) for b in bars)]
    h = 0.8 / len(systems)
    fig, ax = plt.subplots(figsize=(args.width, 0.42 * len(bars) + 1.2))
    y = list(range(len(bars)))
    for si_, s in enumerate(systems):
        off = ((len(systems) - 1) / 2 - si_) * h
        vals = [b[1].get(s) or 0 for b in bars]
        color, hatch, _ = style_of(s)
        ax.barh([yv + off for yv in y], vals, height=h * 0.88,
                color=color, linewidth=0,
                hatch=hatch if args.texture else None,
                label=SYSTEM_LABEL.get(s, s))
    ax.set_yticks(y, [b[0] for b in bars])
    ax.set_xlabel("steady-state time per iteration (µs)\nlower is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("Language integration cost")

    write_table(os.path.join(args.outdir, "integration.tex"),
                "control-flow variants at k=4 n=1M float64: steady us, "
                "MetaTensor launches/iter, torch.compile graphs/breaks, "
                "ratio torch.compile/ours",
                ["variant", "MetaTensor us", "launches/iter",
                 "torch.compile us", "graphs", "breaks", "eager us",
                 "JAX us", "compile/ours"],
                table)
    return fig, "integration"


def fig_warmup(out, args):
    """Per-forward latency from a fresh process until steady state: one panel
    per model, cumulative time (log-log)."""
    rows = read_tsv(os.path.join(out, "warmup.tsv"))
    if not rows:
        return None
    series = collections.defaultdict(dict)  # (model,system,cache,round) -> {iter: us}
    for r in rows:
        key = (r["model"], r["system"], r["cache"], r["round"])
        series[key][int(r["iter"])] = float(r["us"])
    models = sorted({k[0] for k in series})
    systems = [s for s in ["torch-eager", "ours", "torch-compile",
                           "torch-compile-ro", "jax"]
               if any(k[1] == s for k in series)]

    if args.width >= DOUBLE_COL:
        fig, axes = plt.subplots(1, len(models), sharey=True, squeeze=False,
                                 figsize=(args.width, args.width * 0.78))
        axes = list(axes[0])
    else:
        fig, axes = plt.subplots(len(models), 1, sharex=True, sharey=True,
                                 squeeze=False,
                                 figsize=(args.width,
                                          1.35 * len(models) + 0.9))
        axes = [a[0] for a in axes]
    for ax, model in zip(axes, models):
        for si_, s in enumerate(systems):
            colour, _, _ = style_of(s)
            for cache, ls in (("warm", "-"), ("cold", "--")):
                keys = [k for k in series
                        if k[0] == model and k[1] == s and k[2] == cache]
                if not keys:
                    continue
                curves = []
                for k in keys:
                    d = series[k]
                    n = max(d) + 1
                    acc, cum = 0.0, []
                    for it in range(n):
                        acc += d.get(it, 0.0)
                        cum.append(acc / 1000.0)  # ms
                    curves.append(cum)
                n = min(len(c) for c in curves)
                med_curve = [statistics.median(c[i] for c in curves)
                            for i in range(n)]
                ax.plot(range(1, n + 1), med_curve, color=colour,
                        linestyle=ls, linewidth=1.2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(model, fontsize=8)
        despine(ax)
    if args.width >= DOUBLE_COL:
        for ax in axes:
            ax.set_xlabel("forward index (log scale)")
    else:
        axes[-1].set_xlabel("forward index (log scale)")
    axes[0].set_ylabel("cumulative ms (log)\nlower is better")
    # Colour is the system, dash is the kernel cache: ten curves need seven
    # legend entries, not ten.
    handles = [Line2D([], [], color=style_of(s)[0], lw=1.6,
                      label=SYSTEM_LABEL.get(s, s)) for s in systems]
    handles += [Line2D([], [], color=INK, lw=1.2, linestyle="-",
                       label="warm kernel cache"),
                Line2D([], [], color=INK, lw=1.2, linestyle="--",
                       label="cold kernel cache")]
    legend_below(fig, handles=handles,
                 labels=[h.get_label() for h in handles], ncol=3)
    if args.titles:
        fig.suptitle("Warm-up: cumulative time from a fresh process")

    jrows = [r for r in read_jsonl(os.path.join(out, "results.jsonl"))
            if r.get("kind") == "warmup"]
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in jrows:
        g[(r["model"], r["system"], r["cache"])]["first_us"].append(r.get("first_us"))
        g[(r["model"], r["system"], r["cache"])]["steady_us"].append(r.get("steady_us"))
        g[(r["model"], r["system"], r["cache"])]["steady_at"].append(r.get("steady_at"))
        g[(r["model"], r["system"], r["cache"])]["crossover"].append(r.get("crossover"))
    table = []
    for model in models:
        for s in systems:
            for cache in ("warm", "cold"):
                d = g.get((model, s, cache))
                if not d:
                    continue
                first = med(d["first_us"])
                steady = med(d["steady_us"])
                sa = [x for x in d["steady_at"] if isinstance(x, (int, float))]
                sa_str = "%.0f" % statistics.median(sa) if sa else "none"
                co = [x for x in d["crossover"] if isinstance(x, (int, float))]
                co_str = "%.0f" % statistics.median(co) if co else "none"
                table.append([model, SYSTEM_LABEL.get(s, s), cache,
                             "%.2f" % (first / 1000.0) if first else "n/a",
                             sa_str, "%.1f" % steady if steady else "n/a", co_str])
    write_table(os.path.join(args.outdir, "warmup.tex"),
               "first forward and steady-state onset from a fresh process, "
               "median over the rounds; crossover is the first forward at "
               "which cumulative time drops below torch-eager's",
               ["model", "system", "cache", "first forward (ms)",
                "steady_at (iters)", "steady (us)", "crossover vs eager (iters)"],
               table)
    return fig, "warmup"


def fig_batch(out, args):
    """Batch-size sweep: one panel per model, x = batch (log2),
    y = per-sequence microseconds (log), one line per system.

    Per sequence because per forward every line climbs. The point is where the
    lines meet: at batch 1 per-launch overhead separates the systems, and it
    amortises as the GEMMs grow.
    """
    rows = read_tsv(os.path.join(out, "batch.tsv"))
    if not rows:
        return None
    per_seq = collections.defaultdict(list)
    steady = collections.defaultdict(list)
    ident = collections.defaultdict(list)
    failed = collections.defaultdict(list)
    for r in rows:
        key = (r["model"], r["system"], int(r["batch"]))
        if (r.get("status") or "ok") != "ok" or not r.get("per_seq_us"):
            failed[key].append(1)
            continue
        per_seq[key].append(float(r["per_seq_us"]))
        steady[key].append(float(r["steady_us"]))
        if r.get("batch_rows_identical") not in (None, ""):
            ident[key].append(r["batch_rows_identical"])
    if not per_seq:
        return None
    models = sorted({k[0] for k in per_seq})
    systems = [s for s in ["ours", "torch-compile", "torch-compile-ro",
                           "torch-eager", "jax"]
               if any(k[1] == s for k in per_seq)]
    batches = sorted({k[2] for k in per_seq} | {k[2] for k in failed})

    if args.width >= DOUBLE_COL:
        fig, axes = plt.subplots(1, len(models), squeeze=False, sharey=True,
                                 figsize=(args.width, args.width * 0.42 + 0.5))
        axes = axes[0]
    else:
        # At one column two panels side by side leave 1.2in each; they stack.
        fig, axes = plt.subplots(len(models), 1, squeeze=False, sharex=True,
                                 figsize=(args.width,
                                          1.35 * len(models) + 0.95))
        axes = [a[0] for a in axes]
    for mi, model in enumerate(models):
        ax = axes[mi]
        for s in drawn(systems, args):
            pts = [(b, band(per_seq[(model, s, b)])) for b in batches
                   if (model, s, b) in per_seq]
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1][0] for p in pts]
            color, _, marker = style_of(s)
            ax.errorbar(xs, ys, yerr=err(ys, [p[1][1] for p in pts],
                                         [p[1][2] for p in pts]),
                        fmt="none", zorder=2, **ERRBAR)
            ax.plot(xs, ys, color=color, linewidth=1.4, marker=marker,
                    markersize=3.4, markeredgewidth=0, zorder=3,
                    label=SYSTEM_LABEL[s] if mi == 0 else None)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(batches)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%d" % v))
        if ax is axes[-1] or args.width >= DOUBLE_COL:
            ax.set_xlabel("batch size (sequences per forward)")
        ax.set_title(model, fontsize=8)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    for ax in (axes if args.width < DOUBLE_COL else axes[:1]):
        ax.set_ylabel("\u00b5s per sequence\n(log, lower is better)")
    legend_below(fig, axes[0], ncol=2)

    def cell(key, table, spec="%.0f"):
        v = med(table.get(key, []))
        return (spec % v) if v else "n/a"

    header = ["model", "batch"] + [SYSTEM_LABEL[s] for s in systems] + [
        "ours/compile", "ours/jax", "rows identical"]
    table_rows = []
    for model in models:
        for b in batches:
            row = [model, str(b)]
            for s in systems:
                key = (model, s, b)
                row.append("failed" if key in failed and key not in steady
                           else cell(key, steady))
            ours = med(steady.get((model, "ours", b), []))
            tc = med(steady.get((model, "torch-compile", b), []))
            jx = med(steady.get((model, "jax", b), []))
            row.append("%.2f" % (ours / tc) if ours and tc else "n/a")
            row.append("%.2f" % (ours / jx) if ours and jx else "n/a")
            marks = [v for s in systems for v in ident.get((model, s, b), [])]
            row.append("n/a" if not marks
                       else ("yes" if all(v == "1" for v in marks) else "NO"))
            table_rows.append(row)
    write_table(os.path.join(args.outdir, "batch.tex"),
                "steady-state us per forward (the whole batch), median over "
                "the rounds, and the ours/torch.compile and ours/JAX ratios; "
                "\"rows identical\" is the batching check - every sequence in "
                "the batch is the same token ids, so every output row block "
                "must match the first",
                header, table_rows)
    return fig, "batch"


def fig_correctness(out, args):
    """Worst case over the five derived inputs per model (see
    run_correctness.sh), not the average."""
    rows = read_tsv(os.path.join(out, "correctness.tsv"))
    rows = [r for r in rows if r.get("system") != "ours" and r.get("maxabsdiff")]
    if not rows:
        return None
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[r["model"]][r["system"]].append(r)
    systems = [s for s in ["torch-eager", "torch-compile", "jax"]
               if any(s in d for d in g.values())]
    # Older result sets still hold rows for models no longer measured; show
    # the population only.
    models = sorted(m for m in g if m not in PROBES)

    def worst(model, sys_):
        rs = g[model].get(sys_) or []
        if not rs:
            return None
        diffs = [float(r["maxabsdiff"]) for r in rs if r["maxabsdiff"]]
        matches = [float(r["argmax_match"]) for r in rs if r.get("argmax_match")]
        passes = [r.get("pass") for r in rs if r.get("pass") not in (None, "")]
        tol = next((r["tol"] for r in rs if r.get("tol")), "")
        return (max(diffs) if diffs else None,
                min(matches) if matches else None,
                tol, sum(1 for x in passes if x == "1"), len(rs))

    fig, ax = plt.subplots(figsize=(args.width, 0.36 * len(models) + 1.3))
    h = 0.8 / len(systems)
    y = list(range(len(models)))
    for si_, s in enumerate(systems):
        vals = []
        for m in models:
            w = worst(m, s)
            # A zero difference has no place on a log axis; floor it at the
            # smallest float32 step.
            vals.append(max(w[0], 1e-9) if w and w[0] is not None else 0.0)
        colour, hatch, _ = style_of(s)
        ax.barh([v + ((len(systems) - 1) / 2 - si_) * h for v in y], vals,
                height=h * 0.88, color=colour, linewidth=0,
                hatch=hatch if args.texture else None,
                label=SYSTEM_LABEL[s])
    for i, m in enumerate(models):
        row_tols = {float(worst(m, s)[2]) for s in systems
                   if worst(m, s) and worst(m, s)[2]}
        for t in row_tols:
            ax.plot([t, t], [i - 0.42, i + 0.42], color="0.3",
                   linewidth=0.8, linestyle="--", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("max |diff| vs MetaTensor over 5 inputs (log scale)\n"
                  "dashed: tolerance; lower is better")
    ax.set_yticks(y, models)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3)
    if args.titles:
        ax.set_title("Correctness on five derived inputs")

    header = ["model", "system", "max |diff|", "min argmax match", "tol",
              "pass"]
    table = []
    for m in models:
        for s in systems:
            w = worst(m, s)
            if not w:
                continue
            diff, match, tol, npass, n = w
            table.append([m, SYSTEM_LABEL[s],
                          "%.3g" % diff if diff is not None else "-",
                          "%.3f" % match if match is not None else "-",
                          tol or "-", "%d/%d" % (npass, n)])
    write_table(os.path.join(args.outdir, "correctness.tex"),
                "each system against MetaTensor on five inputs per model "
                "derived from the stored one (INPUT\\_SEED=1..5): the worst "
                "of the five maximum absolute logit differences, the smallest "
                "argmax agreement over the five (fraction of token positions, "
                "top-1 for the vision models), the tolerance for that "
                "(workload class, dtype), and how many of the five inputs "
                "passed it",
                header, table)
    return fig, "correctness"


# --- where the DAG lives ---------------------------------------------------

LAZY_ARMS = ["virtual", "deferred", "eager"]


def _lazy_group(rows, key):
    g = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        g[key(r)][r["system"]].append(r)
    return g


def _lazy_ratio_panel(ax, keys, groups, labels, args, field="steady_us"):
    """Bars per arm, relative to the virtual arm, with the range over rounds."""
    x = list(range(len(keys)))
    width = 0.38
    for i, arm in enumerate(["deferred", "eager"]):
        vals, lows, highs = [], [], []
        for k in keys:
            num = [r[field] for r in groups[k].get(arm, [])]
            den = [r[field] for r in groups[k].get("virtual", [])]
            m, lo, hi = ratio_band(num, den)
            vals.append(m or 0.0)
            lows.append(lo or 0.0)
            highs.append(hi or 0.0)
        colour, hatch, _ = style_of(arm)
        xs = [v + (i - 0.5) * width for v in x]
        ax.bar(xs, vals, width * 0.92, color=colour, linewidth=0,
               hatch=hatch if args.texture else None,
               label=SYSTEM_LABEL.get(arm, arm))
        ax.errorbar(xs, vals, yerr=err(vals, lows, highs), fmt="none",
                    zorder=4, **ERRBAR)
    ax.axhline(1.0, color=INK_2, linewidth=0.8, linestyle="--")
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_yscale("log")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)


def fig_lazy(out, args):
    """The experiment the deferred layer exists for: one binary, emitter, cache,
    allocator and set of forcing boundaries, with the operation DAG in three places."""
    rows = read_tsv(os.path.join(out, "lazy.tsv"))
    if not rows:
        return None
    groups = _lazy_group(rows, lambda r: r["model"])
    keys = [k for k in collections.OrderedDict.fromkeys(r["model"] for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(args.width, 2.5))
    _lazy_ratio_panel(axes[0], keys, groups, keys, args)
    axes[0].set_ylabel("latency relative to the DAG\nin the JIT (lower is better)")

    # The deferred arm rebuilds the DAG every iteration, which shows in the bytes.
    x = list(range(len(keys)))
    width = 0.27
    for i, arm in enumerate(LAZY_ARMS):
        vals = []
        for k in keys:
            vals.append(med([r["gc_ms"] for r in groups[k].get(arm, [])]) or 0.0)
        colour, hatch, _ = style_of(arm)
        axes[1].bar([v + (i - 1) * width for v in x], vals, width * 0.92,
                    color=colour, linewidth=0,
                    hatch=hatch if args.texture else None,
                    label=SYSTEM_LABEL.get(arm, arm))
    axes[1].set_ylabel("ms in the collector over the\ntimed loop (lower is better)")
    axes[1].set_xticks(x, keys, rotation=30, ha="right")
    axes[1].yaxis.grid(True, zorder=0)
    axes[1].set_axisbelow(True)
    despine(axes[1])
    handles, labels = axes[1].get_legend_handles_labels()
    legend_below(fig, handles=handles, labels=labels, ncol=3)
    if args.titles:
        axes[0].set_title("Where the operation DAG lives")

    table = []
    for k in keys:
        for arm in LAZY_ARMS:
            rs = groups[k].get(arm, [])
            if not rs:
                continue
            m, lo, hi = band([r["steady_us"] for r in rs])
            table.append([k, SYSTEM_LABEL.get(arm, arm), "%.1f" % m,
                          spread_str(lo, hi),
                          "%.1f" % (med([r["launches_per_iter"] for r in rs]) or 0),
                          "%d" % (med([r["kernels"] for r in rs]) or 0),
                          "%d" % (med([r["compiles"] for r in rs]) or 0),
                          "%d" % (med([r["gc_ms"] for r in rs]) or 0),
                          si(int(med([r.get("peak_bytes") for r in rs]) or 0)),
                          "%d" % (med([r["lazy_nodes"] for r in rs]) or 0),
                          "%d" % (med([r.get("loops") for r in rs]) or 0),
                          "%d" % (med([r.get("bridges") for r in rs]) or 0),
                          rs[0].get("argmax_match", "")])
    write_table(os.path.join(args.outdir, "lazy.tex"),
                "one binary, three places to keep the operation DAG. "
                "kernels and Triton compiles are cumulative process counters "
                "and are cardinalities: they say how many distinct kernels an "
                "arm compiled, not which, and do not identify kernel keys or "
                "order",
                ["model", "DAG lives in", "us/iter", "range", "launches/iter",
                 "kernels", "compiles", "gc ms", "peak bytes",
                 "nodes deferred", "loops", "bridges", "same argmax"], table)
    return fig, "lazy"


def fig_lazy_col(out, args):
    """The body's version: the latency panel alone, at one column."""
    rows = read_tsv(os.path.join(out, "lazy.tsv"))
    if not rows:
        return None
    groups = _lazy_group(rows, lambda r: r["model"])
    keys = [k for k in collections.OrderedDict.fromkeys(r["model"] for r in rows)]
    fig, ax = plt.subplots(figsize=(args.width, 2.4))
    _lazy_ratio_panel(ax, keys, groups, keys, args)
    ax.set_ylabel("latency relative to the DAG\nin the JIT (lower is better)")
    legend_below(fig, ax, ncol=2)
    return fig, "lazy_col"


def fig_lazy_micro(out, args):
    """The same three arms across the size range, where DAG rebuild cost
    dominates at one end and vanishes at the other."""
    rows = read_tsv(os.path.join(out, "lazy_micro.tsv"))
    if not rows:
        return None
    key = lambda r: (int(r["variant"]), int(r["k"]), int(r["n"]))
    groups = _lazy_group(rows, key)
    keys = list(collections.OrderedDict.fromkeys(key(r) for r in rows))
    labels = [micro_label(*k) for k in keys]
    fig, ax = plt.subplots(figsize=(args.width, 2.6))
    _lazy_ratio_panel(ax, keys, groups, labels, args)
    ax.set_ylabel("latency relative to\nthe DAG in the JIT\n(lower is better)")
    legend_below(fig, ax, ncol=2)
    write_table(os.path.join(args.outdir, "lazy_micro.tex"),
                "median us per iteration over the rounds, with launches per "
                "iteration",
                ["benchmark", "DAG lives in", "us/iter", "range",
                 "launches/iter"],
                [[labels[i], SYSTEM_LABEL.get(arm, arm),
                  "%.1f" % band([r["steady_us"] for r in groups[k][arm]])[0],
                  spread_str(*band([r["steady_us"] for r in groups[k][arm]])[1:]),
                  "%.2f" % (med([r["launches_per_iter"] for r in groups[k][arm]]) or 0)]
                 for i, k in enumerate(keys) for arm in LAZY_ARMS
                 if groups[k].get(arm)])
    return fig, "lazy_micro"


def fig_lazy_warmup(out, args):
    """Start-up, where the deferred arm should win: it fuses on the first
    forward, while the trace has to be recorded and optimized first."""
    rows = read_tsv(os.path.join(out, "lazy_warmup.tsv"))
    if not rows:
        return None
    models = list(collections.OrderedDict.fromkeys(r["model"] for r in rows))
    fig, axes = plt.subplots(len(models), 1, figsize=(args.width, 2.0 * len(models)),
                             sharex=True, squeeze=False)
    for ax, model in zip([a[0] for a in axes], models):
        for arm in LAZY_ARMS:
            series = collections.defaultdict(list)
            for r in rows:
                if r["model"] == model and r["system"] == arm:
                    series[int(r["iter"])].append(float(r["us"]))
            if not series:
                continue
            xs = sorted(series)
            colour, _, marker = style_of(arm)
            ax.plot(xs, [med(series[i]) for i in xs], color=colour,
                    linewidth=1.0, marker=marker, markersize=2.2,
                    markevery=max(1, len(xs) // 12),
                    label=SYSTEM_LABEL.get(arm, arm))
        ax.set_yscale("log")
        ax.set_ylabel("%s\nus per forward\n(lower is better)" % model)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    axes[-1][0].set_xlabel("forward, from a fresh process with an empty kernel cache")
    legend_below(fig, axes[0][0], ncol=3)
    return fig, "lazy_warmup"


# --- host-dependent control flow in a real model ---------------------------

def fig_control(out, args):
    """Early exit on distilgpt2: the exit layer is decided on the host from a
    value the model just computed."""
    rows = read_tsv(os.path.join(out, "control.tsv"))
    if not rows:
        return None
    regimes = list(collections.OrderedDict.fromkeys(r["regime"] for r in rows))
    systems = list(collections.OrderedDict.fromkeys(r["system"] for r in rows))
    fig, axes = plt.subplots(len(regimes), 1, figsize=(args.width, 1.5 * len(regimes) + 0.8),
                             squeeze=False, sharex=True)
    axes = axes.reshape(1, -1)
    for ax, regime in zip(axes[0], regimes):
        vals, p95s, colours, hatches, names = [], [], [], [], []
        for s in systems:
            rs = [r for r in rows if r["regime"] == regime and r["system"] == s]
            if not rs:
                continue
            vals.append(med([r["p50_us"] for r in rs]) or 0.0)
            p95s.append(med([r["p95_us"] for r in rs]) or 0.0)
            c, h, _ = style_of(s)
            colours.append(c)
            hatches.append(h)
            names.append(SYSTEM_LABEL.get(s, s))
        y = list(range(len(vals)))
        ax.barh(y, vals, 0.62, color=colours, linewidth=0,
                hatch=hatches if args.texture else None)
        for i, (v, p) in enumerate(zip(vals, p95s)):
            ax.plot([p], [i], marker="|", markersize=9, color=INK_2)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel("us per iteration, median\nand p95 tick (lower is better)")
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax, keep=("left",))
        ax.tick_params(axis="y", length=0)
        if args.titles or len(regimes) > 1:
            ax.set_title(regime, fontsize=7)
    table = []
    for regime in regimes:
        for s in systems:
            rs = [r for r in rows if r["regime"] == regime and r["system"] == s]
            if not rs:
                continue
            col = lambda c, spec="%.1f": (spec % med([r[c] for r in rs
                                                     if r.get(c) not in (None, "")])
                                          if any(r.get(c) not in (None, "") for r in rs)
                                          else "")
            table.append([regime, SYSTEM_LABEL.get(s, s), col("total_ms", "%.0f"),
                          col("p50_us"), col("p95_us"), col("max_us", "%.0f"),
                          col("bridges", "%.0f"), col("frame_compiles", "%.0f"),
                          rs[0].get("exit_seq", ""),
                          "yes" if all(r.get("pass") == "1" for r in rs) else "NO"])
    write_table(os.path.join(args.outdir, "control.tex"),
                "early exit on distilgpt2, median over three rounds. total ms "
                "is the whole phase including every compilation; frame "
                "compiles counts every frame Dynamo compiled inside that "
                "phase, first compilations included; the exit layers are the "
                "per-iteration sequence of exit depths, which every system has "
                "to reproduce or the row is not comparable",
                ["regime", "system", "total ms", "p50 us", "p95 us", "max us",
                 "bridges", "frame compiles", "exit layers", "pass"],
                table)
    return fig, "control"


def fig_control_cost(out, args):
    """Cumulative time from the first iteration, so compilation is inside the number."""
    rows = read_tsv(os.path.join(out, "control_series.tsv"))
    if not rows:
        return None
    regimes = list(collections.OrderedDict.fromkeys(r["regime"] for r in rows))
    systems = list(collections.OrderedDict.fromkeys(r["system"] for r in rows))
    if args.width >= DOUBLE_COL:
        fig, axes = plt.subplots(1, len(regimes), figsize=(args.width, 2.6),
                                 squeeze=False, sharey=True)
        axes = list(axes[0])
    else:
        fig, axes = plt.subplots(len(regimes), 1, squeeze=False,
                                 sharex=True, sharey=True,
                                 figsize=(args.width,
                                          1.4 * len(regimes) + 0.9))
        axes = [a[0] for a in axes]
    for ax, regime in zip(axes, regimes):
        for sysname in systems:
            series = collections.defaultdict(list)
            for r in rows:
                if r["regime"] == regime and r["system"] == sysname:
                    series[int(r["iter"])].append(float(r["us"]))
            if not series:
                continue
            xs = sorted(series)
            total, ys = 0.0, []
            for i in xs:
                total += med(series[i]) / 1000.0
                ys.append(total)
            colour, _, marker = style_of(sysname)
            ax.plot(xs, ys, color=colour, linewidth=1.0, marker=marker,
                    markersize=2.2, markevery=max(1, len(xs) // 10),
                    label=SYSTEM_LABEL.get(sysname, sysname))
        ax.set_title(regime, fontsize=7)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax)
    if args.width >= DOUBLE_COL:
        for ax in axes:
            ax.set_xlabel("iteration")
    else:
        axes[-1].set_xlabel("iteration")
    # Stacked, the label belongs to the figure: on one axes constrained_layout
    # sizes it against the top panel and clips the longer line.
    label = "cumulative ms, compilation\nincluded (lower is better)"
    if args.width >= DOUBLE_COL:
        axes[0].set_ylabel(label)
    else:
        fig.supylabel(label, fontsize=plt.rcParams["axes.labelsize"])
    legend_below(fig, axes[0], ncol=3)
    return fig, "control_cost"



SUITE_ORDER = ("torchbench", "huggingface", "timm")
SUITE_LABEL = {"torchbench": "TorchBench", "huggingface": "HuggingFace",
              "timm": "TIMM"}
SUITE_BASES = [("torch-inductor", "TorchInductor"),
              ("torch-eager", "PyTorch eager"),
              ("torch-inductor-cg", "TorchInductor (cudagraphs)"),
              ("jax-xla", "JAX/XLA")]


def _present(pairs, have):
    """The (system, label) pairs of `pairs` with any number at all, so older
    result sets plot without an empty legend entry."""
    return [(s, label) for s, label in pairs if have(s)]


def _suite_order(suites):
    return sorted(suites, key=lambda s: SUITE_ORDER.index(s)
                 if s in SUITE_ORDER else len(SUITE_ORDER))


def _suites_groups(out):
    """{(suite, model): {system: [rows]}} from suites.tsv.

    If a (suite, model, system, run) index repeats (a re-run appended to the
    same file), only the last occurrence is kept."""
    rows = read_tsv(os.path.join(out, "suites.tsv"))
    g = collections.defaultdict(lambda: collections.defaultdict(dict))
    for r in rows:
        g[(r["suite"], r["model"])][r["system"]][r.get("run")] = r
    return {key: {s: list(runs.values()) for s, runs in systems.items()}
           for key, systems in g.items()}


def _suites_rows(out):
    """{(suite, model): {...}} from suites.tsv, aggregated as summarize.py's
    table is: "ours" counts only if every run passed the accuracy check with no
    CPU fallback, else median ms per forward over the runs. jax-xla counts only
    if every run timed and passed the same check."""
    out_g = {}
    for key, systems in _suites_groups(out).items():
        ours = systems.get("ours", [])
        good = [x for x in ours if x.get("pass") == "1"
               and x.get("cpu_fallback") == "0"]
        ok = bool(ours) and len(good) == len(ours)
        d = {"ok": ok,
            "ours": med([x["median_ms"] for x in good]) if ok else None}
        for s, _ in SUITE_BASES:
            d[s] = med([x["median_ms"] for x in systems.get(s, [])
                       if x.get("median_ms")])
        jax = systems.get("jax-xla", [])
        if not jax or not all(x.get("median_ms") and x.get("pass") == "1"
                              for x in jax):
            d["jax-xla"] = None
        first = (ours or systems.get("torch-eager") or [{}])[0]
        d["batch"] = first.get("batch", "")
        out_g[key] = d
    return out_g


SUITE_MARKS = [("torch-eager", "eager"), ("jax-xla", "JAX/XLA")]


def _suite_mark_style(s):
    """(edge colour, marker, face) of a baseline's marker in
    suites_speedup: eager keeps its open ink circle, JAX/XLA its identity
    colour and marker."""
    if s == "torch-eager":
        return INK, "o", "white"
    colour, _, marker = style_of(s)
    return colour, marker, colour


def fig_suites_speedup(out, args):
    """Per-model speedup on the PyTorch 2 dashboard suites: diverging bars around
    1.0 as in micro_speedup, one panel per suite so vision models (TIMM) do not
    share a scale with language models (HuggingFace, TorchBench). A model that
    fails the accuracy check gets a fail mark in its sorted place. Eager torch
    and, where it ran and passed, JAX/XLA are markers on the same ratio axis."""
    g = _suites_rows(out)
    if not g:
        return None
    suites = _suite_order({k[0] for k in g})
    panels = []
    for suite in suites:
        keys = [k for k in g if k[0] == suite]
        ok_keys = sorted(
            (k for k in keys if g[k]["ok"] and g[k]["torch-inductor"]),
            key=lambda k: g[k]["torch-inductor"] / g[k]["ours"])
        bad_keys = sorted(k for k in keys if k not in ok_keys)
        panels.append((suite, bad_keys + ok_keys))
    panels = [(s, ks) for s, ks in panels if ks]
    if not panels:
        return None

    heights = [len(ks) for _, ks in panels]
    fig, axes = plt.subplots(
        len(panels), 1, squeeze=False,
        figsize=(args.width, 0.145 * sum(heights) + 0.85 * len(panels) + 0.5),
        gridspec_kw={"height_ratios": heights})
    axes = [a[0] for a in axes]
    table_rows = []
    for ax, (suite, keys) in zip(axes, panels):
        y = list(range(len(keys)))
        vals, labels, fails = [], [], []
        marks = {s: ([], []) for s, _ in SUITE_MARKS}
        for i, k in enumerate(keys):
            d = g[k]
            labels.append(k[1])
            if d["ok"] and d["torch-inductor"]:
                vals.append(d["torch-inductor"] / d["ours"])
                for s, (mx, my) in marks.items():
                    if d[s]:
                        mx.append(d[s] / d["ours"])
                        my.append(i)
            else:
                vals.append(None)
                fails.append(i)
        ok_y = [i for i in y if vals[i] is not None]
        ok_v = [vals[i] for i in ok_y]
        colors = [POS if v >= 1 else NEG for v in ok_v]
        ax.barh(ok_y, [v - 1 for v in ok_v], left=1, height=0.62,
               color=colors, linewidth=0,
               hatch=(HATCH[0] if not args.texture else None))
        lo = min(ok_v + [1.0]) if ok_v else 1.0
        hi = max(ok_v + [1.0]) if ok_v else 1.0
        xlo, xhi = lo / 1.3, hi * 1.3
        for s, _ in SUITE_MARKS:
            ink, marker, face = _suite_mark_style(s)
            in_x, in_y, off = [], [], []
            for x, yv in zip(*marks[s]):
                if xlo <= x <= xhi:
                    in_x.append(x)
                    in_y.append(yv)
                else:
                    off.append((yv, x))
            if in_x:
                ax.scatter(in_x, in_y, s=14, marker=marker, facecolors=face,
                          edgecolors=ink, linewidth=0.8, zorder=6)
            for yv, x in off:
                above = x > xhi
                edge = xhi / 1.05 if above else xlo * 1.05
                ax.scatter([edge], [yv], s=22, facecolors=ink, edgecolors=ink,
                          marker=(">" if above else "<"), zorder=7)
                ax.annotate("%.0f×" % x if x >= 10 else "%.1f×" % x,
                           (edge, yv), xytext=(-4 if above else 4, 0),
                           textcoords="offset points", va="center",
                           ha=("right" if above else "left"), fontsize=6,
                           color=ink)
        if fails:
            ax.scatter([1] * len(fails), fails, marker="x", s=22,
                      color=INK_2, linewidth=1.1, zorder=6)
            for i in fails:
                ax.annotate("accuracy fail", (1, i), xytext=(4, 0),
                           textcoords="offset points", va="center",
                           ha="left", fontsize=6, color=INK_2)
        ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ratio_ticks(lo, hi))
        ax.set_xlim(xlo, xhi)
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "×"))
        ax.set_yticks(y, labels, fontsize=5.5)
        ax.set_ylim(-0.6, len(keys) - 0.4)
        if ok_y:
            for i in (ok_y[0], ok_y[-1]):
                v = vals[i]
                ax.annotate("%.2f×" % v, (max(v, 1.0), i), xytext=(4, 0),
                           textcoords="offset points", va="center",
                           ha="left", fontsize=6, color=INK_2)
        gm = _geomean(ok_v) if ok_v else None
        ax.set_title("%s (gmean %s over inductor)" %
                     (SUITE_LABEL.get(suite, suite),
                      "%.2f×" % gm if gm else "n/a"), fontsize=7)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax, keep=("left",))
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("speedup over TorchInductor, float32\n"
                        "log2 scale; higher is better")
    legend_handles = [
        Line2D([], [], color=POS, lw=4, label="MetaTensor faster"),
        Line2D([], [], color=NEG, lw=4, label="TorchInductor faster"),
    ] + [
        Line2D([], [], marker=_suite_mark_style(s)[1], linestyle="none",
              markerfacecolor=_suite_mark_style(s)[2],
              markeredgecolor=_suite_mark_style(s)[0], markersize=5,
              label="speedup over " + label)
        for s, label in SUITE_MARKS
        if any(g[k][s] for k in g if g[k]["ok"])
    ] + [
        Line2D([], [], marker=">", linestyle="none", color=INK,
              markersize=5, label="marker off-scale (labelled)"),
        Line2D([], [], marker="x", linestyle="none", color=INK_2,
              markersize=5, label="accuracy fail"),
    ]
    legend_below(fig, handles=legend_handles,
                labels=[h.get_label() for h in legend_handles], ncol=2)
    if args.titles:
        fig.suptitle("PyTorch 2 benchmark suites")

    for suite, keys in panels:
        for k in keys:
            d = g[k]
            table_rows.append([
                SUITE_LABEL.get(suite, suite), k[1], d["batch"],
                "%.3f" % d["ours"] if d["ours"] else "n/a",
                "%.3f" % d["torch-eager"] if d["torch-eager"] else "n/a",
                "%.3f" % d["torch-inductor"] if d["torch-inductor"] else "n/a",
                "%.3f" % d["torch-inductor-cg"]
                if d["torch-inductor-cg"] else "n/a",
                "%.2fx" % (d["torch-inductor"] / d["ours"])
                if d["ok"] and d["ours"] and d["torch-inductor"] else "n/a",
                "%.2fx" % (d["torch-eager"] / d["ours"])
                if d["ok"] and d["ours"] and d["torch-eager"] else "n/a",
                "%.3f" % d["jax-xla"] if d["jax-xla"] else "n/a",
                "%.2fx" % (d["jax-xla"] / d["ours"])
                if d["ok"] and d["ours"] and d["jax-xla"] else "n/a",
                "pass" if d["ok"] else "fail",
            ])
    write_table(os.path.join(args.outdir, "suites_speedup.tex"),
               "median ms per forward over the runs, as the inductor "
               "dashboard measures it (float32, TF32 matmuls); a model "
               "counts as passing only if every run cleared the accuracy "
               "check with nothing falling back to the CPU. speedup "
               "columns are baseline time over MetaTensor time, so >1x "
               "means MetaTensor is faster. JAX/XLA is n/a where it did "
               "not run or its outputs failed the same accuracy check",
               ["suite", "model", "batch", "MetaTensor", "eager",
                "inductor", "inductor+cg", "speedup vs inductor",
                "speedup vs eager", "JAX/XLA", "speedup vs JAX/XLA",
                "accuracy"],
               table_rows)
    return fig, "suites_speedup"


def fig_suites_summary(out, args):
    """One geomean speedup per suite per baseline, a compact grouped-bar summary
    of the per-model figure. Tallies bars cannot carry (models faster, models
    passing accuracy) go in the table beside it."""
    g = _suites_rows(out)
    if not g:
        return None
    suites = _suite_order({k[0] for k in g})
    if not suites:
        return None
    bases = _present(SUITE_BASES, lambda s: any(d[s] for d in g.values()))
    per = {base: [] for base, _ in bases}
    counts = {base: [] for base, _ in bases}
    pass_counts = []
    for suite in suites:
        all_keys = [k for k in g if k[0] == suite]
        ok_keys = [k for k in all_keys if g[k]["ok"]]
        pass_counts.append((len(ok_keys), len(all_keys)))
        for base, _ in bases:
            ratios = [g[k][base] / g[k]["ours"] for k in ok_keys
                     if g[k][base]]
            per[base].append(_geomean(ratios))
            counts[base].append((sum(1 for r in ratios if r > 1.0),
                                 len(ratios)))

    fig, ax = plt.subplots(figsize=(args.width, 0.62 * len(suites) + 1.1))
    y = list(range(len(suites)))
    h = 0.8 / len(bases)
    for bi, (base, label) in enumerate(bases):
        vals = [v if v else 1.0 for v in per[base]]
        off = ((len(bases) - 1) / 2 - bi) * h
        color, hatch, _ = style_of(base)
        ax.barh([v + off for v in y], [v - 1 for v in vals], left=1,
               height=h * 0.86, color=color, linewidth=0,
               hatch=hatch if args.texture else None, label=label)
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    flat = [v for vs in per.values() for v in vs if v] + [1.0]
    ax.set_xticks(ratio_ticks(min(flat), max(flat)))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "×"))
    ax.set_yticks(y, [SUITE_LABEL.get(s, s) for s in suites])
    ax.set_xlabel("geomean speedup, float32\nlog2 scale; higher is better")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=2)
    if args.titles:
        ax.set_title("PyTorch 2 benchmark suites, summary")

    header = ["suite", "models (pass/total)"]
    for _, label in bases:
        header += [label + " gmean", "faster on"]
    table_rows = []
    for i, suite in enumerate(suites):
        row = [SUITE_LABEL.get(suite, suite), "%d/%d" % pass_counts[i]]
        for base, _ in bases:
            v = per[base][i]
            row += ["%.2fx" % v if v else "n/a", "%d/%d" % counts[base][i]]
        table_rows.append(row)
    write_table(os.path.join(args.outdir, "suites_summary.tex"),
               "geomean speedup of MetaTensor over each baseline, per "
               "suite, and how many models it wins on out of the models "
               "both ran (JAX/XLA only where it ran and passed the same "
               "accuracy check)",
               header, table_rows)
    return fig, "suites_summary"


FIRST_SYSTEMS = [("ours", "MetaTensor"), ("torch-eager", "PyTorch eager"),
                 ("torch-inductor", "TorchInductor"),
                 ("torch-inductor-cg", "TorchInductor (cudagraphs)"),
                 ("jax-xla", "JAX/XLA")]


def fig_suites_first(out, args):
    """First-call latency. What the first call contains differs by system, so this
    is not a speedup figure and no ratio between systems is meaningful.

    MetaTensor's first_ms is the first forward before warm-up (a plain
    interpreter pass, nothing compiled yet). torch-eager's is also a first call
    with nothing to compile. torch-inductor and torch-inductor-cg are compile
    numbers (run_torch.py resets dynamo and times the next call), but no on-disk
    kernel cache (torch._inductor, Triton) is cleared between runs, so they are
    mostly warm-cache loads, not from-scratch compiles. JAX/XLA's first_ms is
    torchax tracing plus the XLA compile and one run, with JAX's persistent
    compilation cache off, so it is a from-scratch compile.

    One dot per system on a log time axis (a bar must start at zero, which is
    not meaningful here), one row per model, panelled by suite like
    suites_speedup.
    """
    g = _suites_groups(out)
    if not g:
        return None

    def first_of(key, s):
        vals = [float(r["first_ms"]) for r in g[key].get(s, [])
               if r.get("first_ms")]
        return med(vals) if vals else None

    systems = _present(FIRST_SYSTEMS,
                       lambda s: any(first_of(k, s) for k in g))
    suites = _suite_order({k[0] for k in g})
    panels = []
    for suite in suites:
        keys = [k for k in g if k[0] == suite and first_of(k, "ours")]
        keys.sort(key=lambda k: first_of(k, "ours"))
        panels.append((suite, keys))
    panels = [(s, ks) for s, ks in panels if ks]
    if not panels:
        return None

    heights = [len(ks) for _, ks in panels]
    fig, axes = plt.subplots(
        len(panels), 1, squeeze=False,
        figsize=(args.width, 0.145 * sum(heights) + 0.85 * len(panels) + 0.5),
        gridspec_kw={"height_ratios": heights})
    axes = [a[0] for a in axes]
    for ax, (suite, keys) in zip(axes, panels):
        y = list(range(len(keys)))
        for s, label in systems:
            xs, ys = [], []
            for i, k in enumerate(keys):
                v = first_of(k, s)
                if v is not None:
                    xs.append(v)
                    ys.append(i)
            if not xs:
                continue
            colour, _, marker = style_of(s)
            ax.scatter(xs, ys, s=16, color=colour, marker=marker,
                      linewidth=0, zorder=4, label=label)
        ax.set_xscale("log")
        ax.set_yticks(y, [k[1] for k in keys], fontsize=5.5)
        ax.set_ylim(-0.6, len(keys) - 0.4)
        ax.set_title(SUITE_LABEL.get(suite, suite), fontsize=7)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax, keep=("left",))
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("first-call latency (ms, log scale)\n"
                        "not steady state, and not the same thing per "
                        "system - see the caption")
    handles = [Line2D([], [], color=style_of(s)[0], marker=style_of(s)[2],
                      linestyle="none", markersize=5, label=label)
              for s, label in systems]
    legend_below(fig, handles=handles,
                labels=[h.get_label() for h in handles], ncol=2)
    if args.titles:
        fig.suptitle("PyTorch 2 benchmark suites: first-call latency")

    table_rows = []
    for suite, keys in panels:
        for k in keys:
            row = [SUITE_LABEL.get(suite, suite), k[1]]
            for s, _ in systems:
                v = first_of(k, s)
                row.append("%.0f" % v if v is not None else "n/a")
            table_rows.append(row)
    write_table(os.path.join(args.outdir, "suites_first.tex"),
               "median first-call latency in ms over the runs, mostly on "
               "a warm on-disk kernel cache rather than a from-scratch "
               "compile. What the first call contains differs by system "
               "and the columns are not a fair ratio of each other: "
               "MetaTensor is a plain interpreter pass, before the JIT "
               "has traced or compiled anything; torch eager is likewise "
               "just its first call; torch-inductor and its cudagraphs "
               "variant include dynamo tracing and inductor compilation; "
               "JAX/XLA includes the torchax trace and a from-scratch XLA "
               "compile",
               ["suite", "model"] + [label for _, label in systems],
               table_rows)
    return fig, "suites_first"


def fig_suites_accuracy(out, args):
    """Numerical accuracy against the float64 reference.

    suite_check.py runs the dashboard's own same() verdict and also the RMSE of
    ours and eager torch against that reference. The RMSE is a diagnostic, not
    the verdict, since same() looks at more than one number. The two RMSEs share
    a reference and output range per model, so the pair is comparable row to
    row (not across models). A dumbbell, as in fusion. One row per model,
    panelled by suite like suites_speedup; a failed model is marked, not dropped.
    """
    g = _suites_groups(out)
    if not g:
        return None

    def acc_of(key):
        ours = g[key].get("ours", [])
        rmse = next((float(r["rmse"]) for r in ours if r.get("rmse")), None)
        trmse = next((float(r["torch_rmse"]) for r in ours
                     if r.get("torch_rmse")), None)
        passed = next((r.get("pass") for r in ours
                      if r.get("pass") not in (None, "")), None)
        nan = next((r.get("eager_nan") for r in ours
                   if r.get("eager_nan") not in (None, "")), None)
        return rmse, trmse, passed, nan

    suites = _suite_order({k[0] for k in g})
    panels = []
    for suite in suites:
        keys = [k for k in g if k[0] == suite]
        ok_keys = sorted(
            (k for k in keys if acc_of(k)[0] and acc_of(k)[1]),
            key=lambda k: acc_of(k)[0] / acc_of(k)[1])
        bad_keys = sorted(k for k in keys if k not in ok_keys)
        panels.append((suite, bad_keys + ok_keys))
    panels = [(s, ks) for s, ks in panels if ks]
    if not panels:
        return None

    heights = [len(ks) for _, ks in panels]
    fig, axes = plt.subplots(
        len(panels), 1, squeeze=False,
        figsize=(args.width, 0.145 * sum(heights) + 0.85 * len(panels) + 0.5),
        gridspec_kw={"height_ratios": heights})
    axes = [a[0] for a in axes]
    for ax, (suite, keys) in zip(axes, panels):
        y = list(range(len(keys)))
        ours_x, ours_y, eager_x, eager_y, fails, nans = [], [], [], [], [], []
        for i, k in enumerate(keys):
            rmse, trmse, passed, nan = acc_of(k)
            if rmse and trmse:
                ax.plot([rmse, trmse], [i, i], color=GRID, linewidth=1.2,
                       zorder=1, solid_capstyle="round")
                ours_x.append(rmse)
                ours_y.append(i)
                eager_x.append(trmse)
                eager_y.append(i)
            if passed is not None and passed != "1":
                fails.append(i)
            if nan == "1":
                nans.append(i)
        ax.scatter(ours_x, ours_y, s=14, color=style_of("ours")[0],
                  marker=style_of("ours")[2], zorder=3, linewidth=0,
                  label="MetaTensor")
        ax.scatter(eager_x, eager_y, s=14, color=style_of("torch-eager")[0],
                  marker=style_of("torch-eager")[2], zorder=3, linewidth=0,
                  label="PyTorch eager")
        if fails:
            ax.scatter([max(ours_x + eager_x + [1e-9])] * len(fails), fails,
                      marker="x", s=22, color=INK_2, linewidth=1.1, zorder=6)
        for i in nans:
            ax.annotate("eager NaN", (eager_x[eager_y.index(i)], i),
                       xytext=(4, 0), textcoords="offset points",
                       va="center", ha="left", fontsize=6, color=INK_2)
        ax.set_xscale("log")
        ax.set_yticks(y, [k[1] for k in keys], fontsize=5.5)
        ax.set_ylim(-0.6, len(keys) - 0.4)
        ratios = [acc_of(k)[0] / acc_of(k)[1] for k in keys
                 if acc_of(k)[0] and acc_of(k)[1]]
        gm = _geomean(ratios) if ratios else None
        ax.set_title("%s (gmean %s of eager's RMSE)" %
                     (SUITE_LABEL.get(suite, suite),
                      "%.2f×" % gm if gm else "n/a"), fontsize=7)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        despine(ax, keep=("left",))
        ax.tick_params(axis="y", length=0)
    axes[-1].set_xlabel("RMSE vs the float64 reference (log scale)\n"
                        "diagnostic only; same() is the pass/fail verdict")
    legend_handles = [
        Line2D([], [], marker=style_of("ours")[2], linestyle="none",
              color=style_of("ours")[0], markersize=5, label="MetaTensor"),
        Line2D([], [], marker=style_of("torch-eager")[2], linestyle="none",
              color=style_of("torch-eager")[0], markersize=5,
              label="PyTorch eager"),
        Line2D([], [], marker="x", linestyle="none", color=INK_2,
              markersize=5, label="accuracy fail"),
    ]
    legend_below(fig, handles=legend_handles,
                labels=[h.get_label() for h in legend_handles], ncol=3)
    if args.titles:
        fig.suptitle("PyTorch 2 benchmark suites: numerical accuracy")

    table_rows = []
    for suite, keys in panels:
        for k in keys:
            rmse, trmse, passed, nan = acc_of(k)
            row = [SUITE_LABEL.get(suite, suite), k[1],
                  "%.4g" % rmse if rmse is not None else "n/a",
                  "%.4g" % trmse if trmse is not None else "n/a",
                  "%.2fx" % (rmse / trmse) if rmse and trmse else "n/a",
                  "pass" if passed == "1" else "fail" if passed else "n/a"]
            if nan is not None:
                row.append("yes" if nan == "1" else "no")
            table_rows.append(row)
    header = ["suite", "model", "rmse", "torch_rmse", "ratio", "accuracy"]
    if any(len(r) > 6 for r in table_rows):
        header.append("eager nan")
    write_table(os.path.join(args.outdir, "suites_accuracy.tex"),
               "RMSE against the float64 reference, ours and eager torch, "
               "both from suite_check.py's diagnostic; ratio is ours over "
               "eager's, so 1x is the same error eager torch already has",
               header, table_rows)
    return fig, "suites_accuracy"


def _fence_range(found):
    """The 1.5-IQR fence over every box in `found` (matplotlib's whisker rule),
    not the raw min/max of the jittered points, which one outlier would stretch."""
    fenced = []
    for pairs in found.values():
        vals = [v for _, v in pairs]
        if len(vals) > 1:
            q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive")
            iqr = q3 - q1
            lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            fenced += [v for v in vals if lo_f <= v <= hi_f]
        else:
            fenced += vals
    fenced = fenced or [1.0]
    return min(fenced), max(fenced)


def _mark_offscale(ax, found, positions, ylo, yhi):
    """Arrow and model name (as in suites_speedup) for a jittered point outside
    [ylo, yhi], which set_ylim would clip. The exact value is in the table. Each
    direction stacks outward in point order so neighbouring labels do not collide."""
    stack = collections.defaultdict(int)
    for key, pairs in sorted(found.items(), key=lambda kv: positions[kv[0]]):
        pos = positions[key]
        group = key[0]
        for name, v in pairs:
            if ylo <= v <= yhi:
                continue
            above = v > yhi
            edge = yhi / 1.05 if above else ylo * 1.05
            ax.scatter([pos], [edge], marker=("^" if above else "v"), s=20,
                      facecolors=INK, edgecolors=INK, zorder=7)
            gap = 4 + stack[(group, above)] * 9
            stack[(group, above)] += 1
            ax.annotate(name, (pos, edge),
                       xytext=(0, gap if above else -gap),
                       textcoords="offset points", ha="center",
                       va=("bottom" if above else "top"), fontsize=6,
                       color=INK_2)


def _box_stats(vals):
    q1, med_, q3 = statistics.quantiles(vals, n=4, method="inclusive") \
        if len(vals) > 1 else (vals[0], vals[0], vals[0])
    return len(vals), min(vals), q1, med_, q3, max(vals)


def _grouped_boxplot(ax, groups, series, values_fn, geomean=False, jitter=0.7):
    """One box per (group, series) pair, side by side within each group and
    coloured by style_of(series key). Every value is also a jittered point (fixed
    seed per box, so renders are identical), hence showfliers is off. Returns
    {(group, key): [(name, value)]} and each box's x position."""
    n = len(series)
    width = 0.8 / max(n, 1)
    found, positions = {}, {}
    for gi, group in enumerate(groups):
        for si, (skey, _) in enumerate(series):
            pairs = values_fn(group, skey)
            vals = [v for _, v in pairs]
            pos = gi + (si - (n - 1) / 2) * width
            found[(group, skey)] = pairs
            positions[(group, skey)] = pos
            if not vals:
                continue
            colour, _, _ = style_of(skey)
            ax.boxplot([vals], positions=[pos], widths=width * 0.82,
                      whis=1.5, showfliers=False, patch_artist=True,
                      boxprops=dict(facecolor=colour, alpha=0.28,
                                    edgecolor=colour, linewidth=1.0),
                      medianprops=dict(color=colour, linewidth=1.4),
                      whiskerprops=dict(color=colour, linewidth=1.0),
                      capprops=dict(color=colour, linewidth=1.0), zorder=2)
            rng = random.Random(gi * 97 + si * 31 + 1)
            xs = [pos + (rng.random() - 0.5) * width * jitter for _ in vals]
            ax.scatter(xs, vals, s=5, color=colour, alpha=0.55, zorder=3,
                      linewidth=0)
            if geomean:
                ax.scatter([pos], [_geomean(vals)], marker="D", s=16,
                          facecolors="white", edgecolors=INK,
                          linewidth=1.0, zorder=5)
    return found, positions


def fig_suites_speedup_box(out, args):
    """The per-suite distribution behind suites_speedup: a box (median, IQR,
    whiskers at 1.5x IQR) per suite per baseline, every model a jittered point
    over its box, and an "all" group pooling the population. A single-column
    companion to the per-model figure. Only a point clipped to the axis edge is named."""
    g = _suites_rows(out)
    if not g:
        return None
    suites = _suite_order({k[0] for k in g})
    if not suites:
        return None
    groups = suites + ["all"]

    def values(group, base):
        keys = [k for k in g if (group == "all" or k[0] == group)
               and g[k]["ok"] and g[k][base]]
        return [(k[1], g[k][base] / g[k]["ours"]) for k in keys]

    bases = _present(SUITE_BASES, lambda s: any(d[s] for d in g.values()))
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.95))
    found, positions = _grouped_boxplot(ax, groups, bases, values,
                                        geomean=True)
    ax.axhline(1, color=INK_2, linewidth=0.8, zorder=1)
    ax.set_yscale("log", base=2)
    fence_lo, fence_hi = _fence_range(found)
    ylo, yhi = min(fence_lo, 1.0) / 1.3, max(fence_hi, 1.0) * 1.3
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(ratio_ticks(ylo, yhi))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "×"))
    _mark_offscale(ax, found, positions, ylo, yhi)
    base0 = bases[0][0]

    def tick(g_):
        t = "%s\nn=%d" % (SUITE_LABEL.get(g_, "All"), len(found[(g_, base0)]))
        if (g_, "jax-xla") in found:
            t += "\nJAX n=%d" % len(found[(g_, "jax-xla")])
        return t
    ax.set_xticks(list(range(len(groups))), [tick(g_) for g_ in groups])
    ax.set_ylabel("speedup, float32\nlog2 scale; higher is better")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)

    handles = [Line2D([], [], color=style_of(s)[0], lw=4, label=label)
              for s, label in bases]
    legend_below(fig, handles=handles,
                labels=[h.get_label() for h in handles],
                ncol=2 if args.width < DOUBLE_COL else 4)
    if args.titles:
        ax.set_title("PyTorch 2 benchmark suites, speedup distribution")

    table_rows = []
    for group in groups:
        for base, label in bases:
            vals = [v for _, v in found[(group, base)]]
            if not vals:
                continue
            n, mn, q1, med_, q3, mx = _box_stats(vals)
            table_rows.append([SUITE_LABEL.get(group, "All"), label,
                              "%d" % n, "%.2fx" % mn, "%.2fx" % q1,
                              "%.2fx" % med_, "%.2fx" % q3, "%.2fx" % mx,
                              "%.2fx" % _geomean(vals)])
    write_table(os.path.join(args.outdir, "suites_speedup_box.tex"),
               "per-model speedup distribution (baseline time over "
               "MetaTensor time, so >1x means MetaTensor is faster), "
               "whiskers at 1.5x the interquartile range; JAX/XLA over "
               "the models it ran and passed the accuracy check on",
               ["suite", "baseline", "n", "min", "q1", "median", "q3",
                "max", "geomean"],
               table_rows)
    return fig, "suites_speedup_box"


def fig_suites_first_box(out, args):
    """The per-suite distribution behind suites_first's per-model dots. As there,
    what a first call contains differs by system (MetaTensor and eager are
    uncompiled; TorchInductor and its cudagraphs variant include the dynamo
    trace and inductor compile, JAX/XLA the torchax trace and XLA compile), so
    these are each system's own numbers, not a ratio. Box per suite per system,
    every model a jittered point, plus an "all" group."""
    g = _suites_groups(out)
    if not g:
        return None

    def first_of(key, s):
        vals = [float(r["first_ms"]) for r in g[key].get(s, [])
               if r.get("first_ms")]
        return med(vals) if vals else None

    suites = _suite_order({k[0] for k in g})
    if not suites:
        return None
    groups = suites + ["all"]

    def values(group, s):
        keys = [k for k in g if group == "all" or k[0] == group]
        out = []
        for k in keys:
            v = first_of(k, s)
            if v is not None:
                out.append((k[1], v))
        return out

    systems = _present(FIRST_SYSTEMS, lambda s: values("all", s))
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.95))
    found, _ = _grouped_boxplot(ax, groups, systems, values)
    ax.set_yscale("log")
    def tick(g_):
        t = "%s\nn=%d" % (SUITE_LABEL.get(g_, "All"), len(values(g_, "ours")))
        if (g_, "jax-xla") in found:
            t += "\nJAX n=%d" % len(found[(g_, "jax-xla")])
        return t
    ax.set_xticks(list(range(len(groups))), [tick(g_) for g_ in groups])
    ax.set_ylabel("first-call latency (ms, log scale)\n"
                 "not steady state, and not the same thing per system")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    handles = [Line2D([], [], color=style_of(s)[0], lw=4, label=label)
              for s, label in systems]
    legend_below(fig, handles=handles,
                labels=[h.get_label() for h in handles], ncol=2)
    if args.titles:
        ax.set_title("PyTorch 2 benchmark suites, first-call distribution")

    table_rows = []
    for group in groups:
        for s, label in systems:
            vals = [v for _, v in found[(group, s)]]
            if not vals:
                continue
            n, mn, q1, med_, q3, mx = _box_stats(vals)
            table_rows.append([SUITE_LABEL.get(group, "All"), label,
                              "%d" % n, "%.0f" % mn, "%.0f" % q1,
                              "%.0f" % med_, "%.0f" % q3, "%.0f" % mx])
    write_table(os.path.join(args.outdir, "suites_first_box.tex"),
               "first-call latency in ms, whiskers at 1.5x the "
               "interquartile range. What the first call contains "
               "differs by system (see suites_first), so no ratio "
               "between systems is implied by sharing this table",
               ["suite", "system", "n", "min", "q1", "median", "q3", "max"],
               table_rows)
    return fig, "suites_first_box"


def fig_suites_accuracy_box(out, args):
    """The per-suite distribution behind suites_accuracy: ours' RMSE over eager's
    against the float64 reference (suite_check.py), a diagnostic, not the
    pass/fail verdict from same(). Box per suite, every model a jittered point,
    plus an "all" group."""
    g = _suites_groups(out)
    if not g:
        return None

    def acc_ratio(key):
        ours = g[key].get("ours", [])
        rmse = next((float(r["rmse"]) for r in ours if r.get("rmse")), None)
        trmse = next((float(r["torch_rmse"]) for r in ours
                     if r.get("torch_rmse")), None)
        return rmse / trmse if rmse and trmse else None

    suites = _suite_order({k[0] for k in g})
    if not suites:
        return None
    groups = suites + ["all"]
    series = [("ours", "ours / eager RMSE")]

    def values(group, _key):
        keys = [k for k in g if group == "all" or k[0] == group]
        out = []
        for k in keys:
            r = acc_ratio(k)
            if r is not None:
                out.append((k[1], r))
        return out

    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.9))
    found, positions = _grouped_boxplot(ax, groups, series, values,
                                        geomean=True)
    ax.axhline(1, color=INK_2, linewidth=0.8, zorder=1)
    ax.set_yscale("log")
    fence_lo, fence_hi = _fence_range(found)
    ylo, yhi = min(fence_lo, 1.0) / 1.8, max(fence_hi, 1.0) * 1.8
    ax.set_ylim(ylo, yhi)
    ax.set_yticks(ratio_ticks(ylo, yhi))
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "×"))
    _mark_offscale(ax, found, positions, ylo, yhi)
    counts = {group: len(found[(group, "ours")]) for group in groups}
    ax.set_xticks(list(range(len(groups))),
                 ["%s\nn=%d" % (SUITE_LABEL.get(g_, "All"), counts[g_])
                  for g_ in groups])
    ax.set_ylabel("RMSE vs float64 reference, ours / eager (log scale)\n"
                 "1× = same error as eager")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax)
    if args.titles:
        ax.set_title("PyTorch 2 benchmark suites, accuracy distribution")

    table_rows = []
    for group in groups:
        vals = [v for _, v in found[(group, "ours")]]
        if not vals:
            continue
        n, mn, q1, med_, q3, mx = _box_stats(vals)
        table_rows.append([SUITE_LABEL.get(group, "All"), "%d" % n,
                          "%.2fx" % mn, "%.2fx" % q1, "%.2fx" % med_,
                          "%.2fx" % q3, "%.2fx" % mx,
                          "%.2fx" % _geomean(vals)])
    write_table(os.path.join(args.outdir, "suites_accuracy_box.tex"),
               "distribution of ours' RMSE over eager's, against the "
               "float64 reference; same() is the pass/fail verdict, this "
               "ratio is diagnostic. whiskers at 1.5x the interquartile "
               "range",
               ["suite", "n", "min", "q1", "median", "q3", "max",
                "geomean"],
               table_rows)
    return fig, "suites_accuracy_box"


ABL_ARMS = [("nofuse", "tensor pass off"),
            ("deferred", "deferred library (DAG at run time)"),
            ("nojit", "JIT off"),
            ("split-branch", "region ends at every guard"),
            ("split-all", "... and at every exception edge")]


def _ablation_suites(out):
    rows = read_tsv(os.path.join(out, "ablation_suites.tsv"))
    g = collections.OrderedDict()
    for r in rows:
        g.setdefault(r["model"], collections.defaultdict(list))[r["arm"]].append(r)
    return g


def _abl_ok(rs):
    return bool(rs) and all(r.get("median_ms") and r.get("cpu_fallback") == "0"
                            for r in rs) and rs[0].get("pass") == "1"


def fig_ablation_suites(out, args):
    g = _ablation_suites(out)
    if not g:
        return None
    models = [m for m in g if _abl_ok(g[m].get("fused"))]
    arms = [(a, l) for a, l in ABL_ARMS
            if any(_abl_ok(g[m].get(a)) for m in models)]
    ratios = {a: {} for a, _ in arms}
    for a, _ in arms:
        for m in models:
            if _abl_ok(g[m].get(a)):
                ratios[a][m] = ratio_band([r["median_ms"] for r in g[m][a]],
                                          [r["median_ms"] for r in g[m]["fused"]])
    keys = models + ["geomean"]
    fig, ax = plt.subplots(figsize=(args.width, 0.3 * len(keys) * len(arms) / 2 + 1.2))
    h = 0.8 / len(arms)
    flat = [1.0]
    for i, (a, label) in enumerate(arms):
        ys, vals, lows, highs = [], [], [], []
        for j, m in enumerate(keys):
            if m == "geomean":
                v = _geomean([b[0] for b in ratios[a].values()])
                lo = hi = v
            elif m in ratios[a]:
                v, lo, hi = ratios[a][m]
            else:
                continue
            ys.append(j + (i - (len(arms) - 1) / 2) * h)
            vals.append(v)
            lows.append(lo)
            highs.append(hi)
        flat += lows + highs
        colour, hatch, _ = style_of({"nojit": "nojit-port", "split-branch": "deferred",
                                     "split-all": "nojit-port"}.get(a, a))
        ax.barh(ys, [v - 1 for v in vals], left=1, height=h * 0.86,
                color=colour, linewidth=0, hatch=hatch if args.texture else None,
                label=label)
        ax.errorbar(vals, ys, xerr=err(vals, lows, highs), fmt="none",
                    zorder=4, **ERRBAR)
        for y, v, hi in zip(ys, vals, highs):
            ax.text(hi * 1.06, y, "%.2f×" % v, va="center", fontsize=5.5)
    ax.axvline(1, color=INK_2, linewidth=0.8, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ratio_ticks(min(flat), max(flat)))
    ax.set_xlim(min(flat) / 1.1, max(flat) * 2.2)
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: ("%g" % v).rstrip("0").rstrip(".") + "×"))
    ax.set_yticks(range(len(keys)), keys)
    ax.invert_yaxis()
    ax.set_xlabel("latency relative to MetaTensor\n(log2 scale)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("left",))
    ax.tick_params(axis="y", length=0)
    legend_below(fig, ax, ncol=3)

    table = []
    for m in models:
        f = g[m]["fused"]
        fm, flo, fhi = band([r["median_ms"] for r in f])
        row = [m, f[0]["batch"], "%.3f" % fm, spread_str(flo, fhi, "%.3f")]
        for a, _ in arms:
            row.append("%.2fx" % ratios[a][m][0] if m in ratios[a] else "n/a")
        for a in ["fused"] + [a for a, _ in arms]:
            rs = g[m].get(a)
            row.append("%.1f" % med([r["launches"] for r in rs])
                       if _abl_ok(rs) else "n/a")
        d = g[m].get("deferred")
        if any(a == "deferred" for a, _ in arms):
            row.append("%.0f" % med([r["lazy_nodes"] for r in d]) if _abl_ok(d) else "n/a")
        row.append("/".join(g[m][a][0].get("pass") or "-" for a in
                            ["fused"] + [a for a, _ in arms] if g[m].get(a)))
        table.append(row)
    gm = ["geomean", "", "", ""]
    for a, _ in arms:
        gm.append("%.2fx" % _geomean([b[0] for b in ratios[a].values()]))
    table.append(gm + [""] * (len(table[0]) - len(gm)))
    write_table(os.path.join(args.outdir, "ablation_suites.tex"),
                "TorchBench models at the dashboard batch: median ms per "
                "forward of MetaTensor (median over processes, with range), "
                "and each arm's latency relative to it; launches per forward "
                "per arm; deferred DAG nodes per forward; accuracy verdict "
                "per arm",
                ["model", "batch", "fused ms", "range"] +
                [l for _, l in arms] +
                ["launches " + a for a in ["fused"] + [a for a, _ in arms]] +
                (["nodes deferred"] if any(a == "deferred" for a, _ in arms) else []) +
                ["pass"], table)

    mi = [("mi4", 4), ("mi6", 6), ("mi8", 8), ("fused", 16)]
    t2 = []
    for m in models:
        row = [m]
        for a, _ in mi:
            rs = g[m].get(a)
            row += (["%.3f" % med([r["median_ms"] for r in rs]),
                     "%.1f" % med([r["launches"] for r in rs])]
                    if _abl_ok(rs) else ["n/a", "n/a"])
        t2.append(row)
    if any(g[m].get("mi4") for m in models):
        write_table(os.path.join(args.outdir, "ablation_inputs.tex"),
                    "fused-kernel input cap: median ms and launches per "
                    "forward at each cap (16 is the default)",
                    ["model"] + [x for _, n in mi for x in
                                 ("ms (%d)" % n, "launches (%d)" % n)], t2)
    _split_tables(out, args)
    return fig, "ablation_suites"


def _split_tables(out, args):
    spans = read_tsv(os.path.join(out, "split_spans.tsv"))
    if spans:
        tot = collections.Counter()
        models = collections.defaultdict(set)
        for r in spans:
            tot[r["guard"]] += int(r["spans"])
            models[r["guard"]].add(r["model"])
        n = sum(tot.values())
        write_table(os.path.join(args.outdir, "ablation_spans.tex"),
                    "guards that a fused region spans in the compiled traces "
                    "of the PT2 suite models (every trace, preamble and "
                    "bridge, counted once when it is optimized): how many, "
                    "their share, and in how many of the %d models"
                    % len({r["model"] for r in spans}),
                    ["guard", "spanned", "share", "models"],
                    [[k, "%d" % v, "%.1f%%" % (100.0 * v / n), "%d" % len(models[k])]
                     for k, v in tot.most_common()] +
                    [["total", "%d" % n, "", ""]])
    micro = read_tsv(os.path.join(out, "split_micro.tsv"))
    control = read_tsv(os.path.join(out, "split_control.tsv"))
    if not micro and not control:
        return
    names = {"0": "one trace", "1": "split at guards", "3": "split at guards and exceptions"}
    rows = []
    g = collections.OrderedDict()
    for r in micro:
        g.setdefault(("micro " + VARIANT_LABEL.get(int(r["variant"]), r["variant"]),
                      r["split"]), []).append(r)
    for r in control:
        g.setdefault(("early exit, " + r["regime"], r["split"]), []).append(r)
    base = {}
    for (w, m), rs in g.items():
        if "steady_us" in rs[0]:
            t = med([r["steady_us"] for r in rs])
            l = "%.2f" % med([r["launches"] for r in rs])
            ok = len(set(r["acc"] for r in rs)) == 1
        else:
            t = med([r["p50_us"] for r in rs])
            l = ""
            ok = all(r["pass"] == "1" for r in rs)
        if m == "0":
            base[w] = t
        rows.append([w, names.get(m, m), "%.1f" % t,
                     "%.2fx" % (t / base[w]) if w in base else "", l,
                     "yes" if ok else "no"])
    write_table(os.path.join(args.outdir, "ablation_split.tex"),
                "same binary, fused regions ended at every guard (and at "
                "every exception edge): median us per iteration (early exit: "
                "p50 over the steady iterations), ratio to one trace, "
                "launches per iteration, same result as one trace",
                ["workload", "arm", "us/iter", "vs one trace", "launches/iter",
                 "same result"], rows)


DYN_TORCH = [("eager", "eager"), ("compile-static", "static"),
             ("compile-default", "default"), ("compile-dynamic", "dynamic")]
DYN_PHASES = ["first", "revisit", "unseen"]


def fig_dynamic_suites(out, args):
    rows = read_tsv(os.path.join(out, "dynamic_suites.tsv"))
    if not rows:
        return None
    checks = read_tsv(os.path.join(out, "dynamic_check.tsv"))
    g = collections.defaultdict(list)
    for r in rows:
        if r.get("median_ms"):
            g[(r["model"], r["arm"], r["cache"], r["phase"], int(r["size"]))].append(r)
    models = list(collections.OrderedDict.fromkeys(r["model"] for r in rows))

    def total(model, arm, phase, field):
        per_run = collections.defaultdict(float)
        for (m, a, c, ph, sz), rs in g.items():
            if m == model and a == arm and ph == phase:
                for r in rs:
                    per_run[r["run"]] += float(r[field] or 0)
        return med(list(per_run.values())) if per_run else None

    def first_ms(model, arm, phase):
        xs = [med([r["first_ms"] for r in rs]) for (m, a, c, ph, sz), rs in g.items()
              if m == model and a == arm and ph == phase]
        return max(xs) if xs else None

    main = []
    for m in models:
        for ph in DYN_PHASES:
            row = [m, ph]
            for f in ("loops", "bridges", "kernels", "compiles"):
                v = total(m, "ours", ph, f)
                row.append("%d" % v if v is not None else "n/a")
            v = first_ms(m, "ours", ph)
            row.append("%.0f" % v if v is not None else "n/a")
            for arm, _ in DYN_TORCH[1:]:
                v = total(m, arm, ph, "graphs")
                t = first_ms(m, arm, ph)
                row.append("%d / %.0f" % (v, t) if v is not None else "n/a")
            main.append(row)
    write_table(os.path.join(args.outdir, "dynamic_suites.tex"),
                "shape sweep, one process per run, caches empty at its start: "
                "compilations summed over the five sizes of each phase "
                "(median over runs) and the slowest first call in the phase "
                "in ms; torch.compile columns are new Dynamo graphs / slowest "
                "first call",
                ["model", "phase", "loops", "bridges", "kernels", "compiles",
                 "first ms"] + ["%s graphs / first ms" % l for _, l in DYN_TORCH[1:]],
                main)

    d3, d3f, detail = [], [], []
    for m in models:
        rs, fs = [], []
        sizes = sorted({k[4] for k in g if k[0] == m and k[1] == "ours"}, reverse=True)
        for sz in sizes:
            ours = [r["median_ms"] for ph in DYN_PHASES if ph != "revisit"
                    for r in g.get((m, "ours", "cold", ph, sz), [])]
            sr = g.get((m, "ours-spec", "warm", "fixed", sz), [])
            last = max([int(r["run"]) for r in sr] or [0])
            spec = [r["median_ms"] for r in sr if int(r["run"]) == last]
            fixed = [r["median_ms"] for r in g.get((m, "ours-fixed", "warm", "fixed", sz), [])]
            o = med(ours)
            if o and spec:
                rs.append(o / med(spec))
            if o and fixed:
                fs.append(o / med(fixed))
            row = [m, "%d" % sz, "%.3f" % o if o else "n/a",
                   "%.3f" % med(spec) if spec else "n/a",
                   "%.3f" % med(fixed) if fixed else "n/a"]
            for arm, _ in DYN_TORCH:
                xs = [r["median_ms"] for ph in DYN_PHASES if ph != "revisit"
                      for r in g.get((m, arm, "cold", ph, sz), [])]
                row.append("%.3f" % med(xs) if xs else "n/a")
            cold = [r["first_ms"] for r in g.get((m, "ours-fixed", "cold", "fixed", sz), [])]
            row.append("%.0f" % med(cold) if cold else "n/a")
            ok = [c["pass"] for c in checks if c["model"] == m and
                  c["point"].endswith("-%d" % sz)]
            row.append("%d/%d" % (ok.count("1"), len(ok)) if ok else "n/a")
            detail.append(row)
        if rs:
            d3.append([m, "%.3fx" % _geomean(rs), "%.3fx" % min(rs), "%.3fx" % max(rs),
                       "%.3fx" % _geomean(fs) if fs else "n/a"])
    if d3:
        def gm(rows, col):
            xs = [float(r[col][:-1]) for r in rows if r[col] != "n/a"]
            return "%.3fx" % _geomean(xs) if xs else "n/a"
        stable = [r for r in d3 if float(r[3][:-1]) / float(r[2][:-1]) < 2]
        out_rows = list(d3)
        out_rows.append(["geomean", gm(d3, 1), "", "", gm(d3, 4)])
        if len(stable) < len(d3):
            out_rows.append(["geomean, max/min < 2", gm(stable, 1), "", "", gm(stable, 4)])
        d3 = out_rows
    write_table(os.path.join(args.outdir, "dynamic_d3.tex"),
                "steady-state cost of size-independent kernels: median ms of "
                "the sweep (first and unseen phases) over a process "
                "specialized to that one size with the size policy's limit "
                "lifted (spec) and over a fresh process at that size under the "
                "default policy (fixed); geomean, min and max over the ten sizes",
                ["model", "sweep / spec", "min", "max", "sweep / fixed"], d3)
    write_table(os.path.join(args.outdir, "dynamic_sizes.tex"),
                "per size: steady median ms of the sweep, of the specialized "
                "and the fixed process, of torch; the fixed process's first "
                "call with empty caches; accuracy points passing same() "
                "against eager with a float64 reference",
                ["model", "size", "sweep", "spec", "fixed"] +
                [l for _, l in DYN_TORCH] + ["cold first ms", "pass"], detail)
    return None


PROF_PARTS = [("gemm", "GEMM / conv"), ("attention", "attention"), ("reduce", "reductions"),
              ("pointwise", "elementwise"), ("layout", "layout / copies"),
              ("other", "other kernels"), ("host", "GPU idle (host)")]


def _profile_rows(out):
    rows = read_tsv(os.path.join(out, "profile.tsv"))
    g = collections.OrderedDict()
    for r in rows:
        if r.get("wall_ms"):
            g.setdefault((r["suite"], r["model"]), {})[r["system"]] = r
    return g


def _parts(r):
    d = dict((k, float(r[k + "_ms"])) for k, _ in PROF_PARTS if k not in ("host", "gemm"))
    d["gemm"] = float(r["matmul_ms"]) + float(r["conv_ms"])
    d["host"] = max(0.0, float(r["wall_ms"]) - float(r["busy_ms"]))
    return d


def fig_profile(out, args):
    g = _profile_rows(out)
    if not g:
        return None
    base = "torch-inductor"
    table, deltas, lead = [], collections.defaultdict(list), collections.Counter()
    for (suite, model), d in g.items():
        if "ours" not in d or "inductor" not in d:
            continue
        o, i = _parts(d["ours"]), _parts(d["inductor"])
        wo, wi = float(d["ours"]["wall_ms"]), float(d["inductor"]["wall_ms"])
        diff = dict((k, i[k] - o[k]) for k, _ in PROF_PARTS)
        if wi > wo:
            lead[(suite, "faster", max(diff, key=lambda k: diff[k]))] += 1
        else:
            lead[(suite, "slower", min(diff, key=lambda k: diff[k]))] += 1
        for k, _ in PROF_PARTS:
            deltas[(suite, k)].append(diff[k] / wi)
        table.append([suite, model, "%.2f" % wo, "%.2f" % wi, "%.2fx" % (wi / wo)] +
                     ["%+.2f" % diff[k] for k, _ in PROF_PARTS] +
                     ["%.0f / %.0f" % (float(d["ours"]["events"]), float(d["inductor"]["events"]))])
    write_table(os.path.join(args.outdir, "profile.tex"),
                "per forward, median over the timed loop under nsys: wall ms of "
                "MetaTensor and TorchInductor, their ratio, and where TorchInductor's "
                "extra time goes (ms, positive = TorchInductor spends more), by kernel "
                "class and GPU idle time; kernels launched (ours / inductor). MetaTensor's "
                "conv and matmul kernels carry their fused epilogues, so their "
                "elementwise work is counted under GEMM / conv; cuDNN runs some convolutions "
                "as plain GEMM kernels, so the two are reported together",
                ["suite", "model", "ours ms", "inductor ms", "speedup"] +
                [l for _, l in PROF_PARTS] + ["kernels"], table)
    suites = _suite_order({s for s, _ in deltas})
    summ = []
    for s in suites:
        row = [SUITE_LABEL.get(s, s)]
        for k, _ in PROF_PARTS:
            xs = deltas[(s, k)]
            row.append("%+.1f%%" % (100 * statistics.median(xs)) if xs else "")
        for side in ("faster", "slower"):
            items = sorted(((n, k) for (ss, sd, k), n in lead.items() if ss == s and sd == side), reverse=True)
            row.append(", ".join("%s %d" % (dict(PROF_PARTS)[k], n) for n, k in items) or "-")
        summ.append(row)
    write_table(os.path.join(args.outdir, "profile_summary.tex"),
                "median over the suite's models of TorchInductor's extra time in each "
                "class as a share of its wall time, and the class with the largest "
                "difference per model, split by whether MetaTensor is faster or slower",
                ["suite"] + [l for _, l in PROF_PARTS] + ["faster: main class", "slower: main class"],
                summ)

    keys = [k for k in g if "ours" in g[k] and "inductor" in g[k]]
    if not keys:
        return None
    keys.sort(key=lambda k: (suites.index(k[0]) if k[0] in suites else 9,
                             float(g[k]["inductor"]["wall_ms"]) / float(g[k]["ours"]["wall_ms"])))
    fig, ax = plt.subplots(figsize=(args.width, 0.16 * len(keys) * 2 + 1.4))
    colours = ["#1a4fa0", "#4fa8e0", "#1baf7a", "#eda100", "#d94f70", "#9a9994", "#3f3f3f"]
    y = 0
    ticks, labels = [], []
    for k in keys:
        wi = float(g[k]["inductor"]["wall_ms"])
        for si, sysname in enumerate(("ours", "inductor")):
            p = _parts(g[k][sysname])
            left = 0.0
            for (part, label), c in zip(PROF_PARTS, colours):
                v = p[part] / wi
                ax.barh(y + si * 0.4, v, left=left, height=0.38, color=c, linewidth=0,
                        label=label if y == 0 and si == 0 else None)
                left += v
        ticks.append(y + 0.2)
        labels.append(k[1][:28])
        y += 1
    ax.set_yticks(ticks, labels, fontsize=4.5)
    ax.invert_yaxis()
    ax.axvline(1.0, color=INK_2, linewidth=0.6, linestyle="--")
    ax.set_xlabel("time per forward, as a fraction of TorchInductor's\n"
                  "(upper bar MetaTensor, lower bar TorchInductor)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))
    legend_below(fig, ax, ncol=3)
    _profile_suites(g, suites, args)
    _profile_host(out, args)
    return fig, "profile"


def _profile_suites(g, suites, args):
    colours = ["#1a4fa0", "#4fa8e0", "#1baf7a", "#eda100", "#d94f70", "#9a9994", "#3f3f3f"]
    fig, ax = plt.subplots(figsize=(args.width, 0.55 * len(suites) * 2 + 1.0))
    ticks, labels = [], []
    for si, suite in enumerate(suites):
        keys = [k for k in g if k[0] == suite and "ours" in g[k] and "inductor" in g[k]]
        for bi, sysname in enumerate(("ours", "inductor")):
            y = si * 2.6 + bi
            left = 0.0
            for (part, label), c in zip(PROF_PARTS, colours):
                v = sum(_parts(g[k][sysname])[part] / float(g[k]["inductor"]["wall_ms"])
                        for k in keys) / len(keys)
                ax.barh(y, v, left=left, height=0.8, color=c, linewidth=0,
                        label=label if si == 0 and bi == 0 else None)
                left += v
            ax.text(left + 0.01, y, "%.2f" % left, va="center", fontsize=6)
            ticks.append(y)
            labels.append("%s, %s (%d)" % (SUITE_LABEL.get(suite, suite),
                                           "MetaTensor" if sysname == "ours" else "TorchInductor",
                                           len(keys)))
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(1.0, color=INK_2, linewidth=0.6, linestyle="--")
    ax.set_xlabel("time per forward as a fraction of TorchInductor's,\n"
                  "averaged over the suite's models")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))
    legend_below(fig, ax, ncol=3)
    for fmt in args.format.split(","):
        fig.savefig(os.path.join(args.outdir, "profile_suites." + fmt), bbox_inches="tight")
    plt.close(fig)


TRAFFIC_PARTS = [("gemm", "GEMM / conv"), ("attention", "attention"), ("reduce", "reductions"),
                 ("pointwise", "elementwise"), ("layout", "layout / copies")]


def fig_profile_traffic(out, args):
    rows = read_tsv(os.path.join(out, "ncu.tsv"))
    if not rows:
        return None
    g = collections.OrderedDict()
    for r in rows:
        part = "gemm" if r["class"] in ("matmul", "conv") else r["class"]
        part = part if part in dict(TRAFFIC_PARTS) else "layout"
        d = g.setdefault((r["suite"], r["model"]), {}).setdefault(
            r["system"], collections.defaultdict(float))
        d[part] += (float(r["read_mb"]) + float(r["write_mb"])) / 1e3
        d["ms"] += float(r["time_ms"])
        d["kernels"] += float(r["kernels"])
    keys = [k for k in g if "ours" in g[k] and "inductor" in g[k]]
    if not keys:
        return None
    tot = lambda d: sum(d[p] for p, _ in TRAFFIC_PARTS)
    table = []
    for k in keys:
        o, i = g[k]["ours"], g[k]["inductor"]
        table.append([SUITE_LABEL.get(k[0], k[0]), k[1].split(".")[0],
                      "%.0f / %.0f" % (o["kernels"], i["kernels"]),
                      "%.2f / %.2f" % (o["ms"], i["ms"])] +
                     ["%.2f / %.2f" % (o[p], i[p]) for p, _ in TRAFFIC_PARTS] +
                     ["%.2f / %.2f" % (tot(o), tot(i)), "%.2f" % (tot(o) / tot(i))])
    write_table(os.path.join(args.outdir, "profile_traffic.tex"),
                "one steady-state forward under Nsight Compute (clocks locked, so times "
                "run longer than under nsys): kernels, GPU ms and DRAM GB read plus written "
                "by kernel class, MetaTensor / TorchInductor, and MetaTensor's total "
                "traffic as a share of TorchInductor's. MetaTensor's conv kernels carry "
                "their fused epilogues",
                ["suite", "model", "kernels", "GPU ms"] + [l + " GB" for _, l in TRAFFIC_PARTS] +
                ["total GB", "ratio"], table)
    fig, ax = plt.subplots(figsize=(args.width, 0.32 * len(keys) * 2 + 1.2))
    colours = ["#1a4fa0", "#4fa8e0", "#1baf7a", "#eda100", "#d94f70"]
    ticks, labels = [], []
    for y, k in enumerate(keys):
        base = tot(g[k]["inductor"])
        for si, sysname in enumerate(("ours", "inductor")):
            left = 0.0
            for (part, label), c in zip(TRAFFIC_PARTS, colours):
                v = g[k][sysname][part] / base
                ax.barh(y + si * 0.4, v, left=left, height=0.38, color=c, linewidth=0,
                        label=label if y == 0 and si == 0 else None)
                left += v
            if si == 0:
                ax.text(left + 0.01, y, "%.2f" % left, va="center", fontsize=6)
        ticks.append(y + 0.2)
        labels.append(k[1].split(".")[0])
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.invert_yaxis()
    ax.axvline(1.0, color=INK_2, linewidth=0.6, linestyle="--")
    ax.set_xlabel("DRAM bytes per forward / TorchInductor's\n"
                  "(upper bar MetaTensor, lower TorchInductor)")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    despine(ax, keep=("bottom",))
    legend_below(fig, ax, ncol=3)
    return fig, "profile_traffic"


def fig_profile_efficiency(out, args):
    g = _profile_rows(out)
    flops = dict(((r["suite"], r["model"]), r) for r in read_tsv(os.path.join(out, "flops_all.tsv")))
    if not g or not flops:
        return None
    pts = collections.defaultdict(list)
    for k, d in g.items():
        if "ours" not in d or "inductor" not in d or k not in flops:
            continue
        f = flops[k]
        for part, fl, cols in (("gemm", float(f["matmul"]) + float(f["conv"]), ("matmul_ms", "conv_ms")),
                               ("attention", float(f["attention"]), ("attention_ms",))):
            ts = [sum(float(d[s][c]) for c in cols) for s in ("ours", "inductor")]
            if fl > 1e9 and all(ts):
                pts[(k[0], part)].append((fl / ts[0] / 1e9, fl / ts[1] / 1e9, k[1]))
    suites = _suite_order({s for s, _ in pts})
    table = []
    for s in suites:
        row = [SUITE_LABEL.get(s, s)]
        for part in ("gemm", "attention"):
            xs = pts.get((s, part), [])
            row += ["%d" % len(xs)]
            row += (["%.1f" % _geomean([x[0] for x in xs]), "%.1f" % _geomean([x[1] for x in xs]),
                     "%.2fx" % _geomean([x[0] / x[1] for x in xs])] if xs else ["-"] * 3)
        table.append(row)
    write_table(os.path.join(args.outdir, "profile_efficiency.tex"),
                "achieved TFLOP/s, geomean over the suite's models: FLOPs counted by "
                "torch.utils.flop_counter on the eager forward, divided by the GPU time of "
                "the class under nsys. MetaTensor's conv time includes its fused epilogues; "
                "attention FLOPs count the full score matrix, so a causal kernel that skips "
                "masked blocks shows above its true rate",
                ["suite", "GEMM / conv models", "ours", "inductor", "ratio",
                 "attention models", "ours", "inductor", "ratio"], table)
    fig, axes = plt.subplots(1, 2, figsize=(args.width, 2.6))
    marks = {"torchbench": "o", "huggingface": "s", "timm": "^"}
    for ax, part, title in zip(axes, ("gemm", "attention"), ("GEMM / conv", "attention")):
        hi = 1.0
        for s in suites:
            xs = pts.get((s, part), [])
            if xs:
                ax.scatter([x[1] for x in xs], [x[0] for x in xs], s=10, marker=marks.get(s, "o"),
                           label=SUITE_LABEL.get(s, s), linewidths=0)
                hi = max(hi, max(max(x[0], x[1]) for x in xs))
        ax.plot([0, hi * 1.05], [0, hi * 1.05], color=INK_2, linewidth=0.6, linestyle="--")
        ax.set_xlim(0, hi * 1.05)
        ax.set_ylim(0, hi * 1.05)
        ax.set_xlabel("TorchInductor TFLOP/s")
        ax.set_title(title, fontsize=8)
        despine(ax)
    axes[0].set_ylabel("MetaTensor TFLOP/s")
    legend_below(fig, axes[0], ncol=3)
    return fig, "profile_efficiency"


HOST_FIELDS = [("lead_ms", "first kernel ms", "%.3f"), ("tail_ms", "sync tail ms", "%.3f"),
               ("api_us_per_launch", "launch API us", "%.1f"), ("api_ms", "launch API ms", "%.2f"),
               ("n_small", "kernels < 10 us", "%.0f"), ("bubble_ms", "GPU gaps ms", "%.3f")]


def _profile_host(out, args):
    rows = read_tsv(os.path.join(out, "profile_detail.tsv"))
    if not rows:
        return None
    g = collections.defaultdict(list)
    for r in rows:
        suite = r["name"].split("_", 1)[0]
        g[(suite, r["system"])].append(r)
    table = []
    for s in _suite_order({s for s, _ in g}):
        for sysname, label in (("ours", "MetaTensor"), ("inductor", "TorchInductor"),
                               ("inductor-cg", "TorchInductor + CUDA graphs")):
            rs = g.get((s, sysname))
            if rs:
                table.append([SUITE_LABEL.get(s, s), label, "%d" % len(rs)] +
                             [fmt % statistics.median(float(r[f]) for r in rs)
                              for f, _, fmt in HOST_FIELDS])
    write_table(os.path.join(args.outdir, "profile_host.tex"),
                "host side of one forward under nsys, median over the suite's models of the "
                "per-model median: time from the forward's start to its first kernel, from the "
                "last kernel's end to the synchronisation returning, launch API time per launch "
                "and per forward (one graph launch for CUDA graphs), kernels shorter than 10 us, "
                "and GPU idle gaps longer than 1 us between kernels",
                ["suite", "system", "models"] + [l for _, l, _ in HOST_FIELDS], table)


FIGURES = collections.OrderedDict([
    ("micro_speedup", fig_micro_speedup),
    ("integration", fig_integration),
    ("fusion", fig_fusion),
    ("fusion_stats", fig_fusion_stats),
    ("models", fig_models),
    ("model_speedup", fig_model_speedup),
    ("dynamic", fig_dynamic),
    ("batch", fig_batch),
    ("precision", fig_precision),
    ("ablation", fig_ablation),
    ("explain", fig_explain),
    ("model_inventory", fig_model_inventory),
    ("model_provenance", fig_model_provenance),
    ("op_inventory", fig_op_inventory),
    ("deopt", fig_deopt),
    ("transition", fig_transition),
    ("motion", fig_motion),
    ("motion_retention", fig_motion_retention),
    ("decode", fig_decode),
    ("micro_baselines", fig_micro_baselines),
    ("compile_overhead", fig_compile_overhead),
    ("gap", fig_gap),
    ("correctness", fig_correctness),
    ("lazy", fig_lazy),
    ("lazy_col", fig_lazy_col),
    ("lazy_micro", fig_lazy_micro),
    ("lazy_warmup", fig_lazy_warmup),
    ("control", fig_control),
    ("control_cost", fig_control_cost),
    ("warmup", fig_warmup),
    ("suites_speedup", fig_suites_speedup),
    ("suites_summary", fig_suites_summary),
    ("suites_speedup_box", fig_suites_speedup_box),
    ("suites_first", fig_suites_first),
    ("suites_first_box", fig_suites_first_box),
    ("suites_accuracy", fig_suites_accuracy),
    ("suites_accuracy_box", fig_suites_accuracy_box),
    ("ablation_suites", fig_ablation_suites),
    ("dynamic_suites", fig_dynamic_suites),
    ("profile", fig_profile),
    ("profile_traffic", fig_profile_traffic),
    ("profile_efficiency", fig_profile_efficiency),
])

COMPARE_FIGURES = collections.OrderedDict([
    ("compare_speedup", fig_compare_speedup),
    ("compare_models", fig_compare_models),
    ("compare_dynamic", fig_compare_dynamic),
])


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", nargs="?", default=os.environ.get("OUT"),
                   help="result directory holding micro.tsv etc.")
    p.add_argument("--outdir", help="where the figures go (default $OUT/figures)")
    p.add_argument("--compare", action="append", default=[], metavar="DIR",
                   help="a second result directory to compare against "
                        "(repeatable, at most three in total)")
    p.add_argument("--only", action="append",
                   choices=list(FIGURES) + list(COMPARE_FIGURES),
                   help="render just this figure (repeatable)")
    p.add_argument("--format", default="pdf",
                   help="comma-separated: pdf, png, svg (default pdf)")
    p.add_argument("--column", choices=["single", "double"],
                   help="figure width: one column or the full text width "
                        "(default: whichever suits the figure)")
    p.add_argument("--font", choices=["serif", "sans-serif"], default="serif")
    p.add_argument("--texture", action="store_true",
                   help="hatch the fills, for grayscale print and full CVD")
    p.add_argument("--titles", action="store_true",
                   help="draw titles on the axes (for slides; documents use captions)")
    p.add_argument("--iree", action="store_true",
                   help="include IREE in the figures (files get an _iree suffix); "
                        "the tables always carry it")
    args = p.parse_args(argv)

    if not args.out:
        p.error("no result directory given and $OUT is unset")
    if not os.path.isdir(args.out):
        p.error("%s is not a directory" % args.out)
    for d in args.compare:
        if not os.path.isdir(d):
            p.error("%s is not a directory" % d)
    # The categorical palette is never cycled, so a fourth machine would need a
    # hue that does not exist.
    if len(args.compare) > 2:
        p.error("at most three result directories can share a figure")
    args.outdir = args.outdir or os.path.join(args.out, "figures")
    explicit_width = args.column is not None
    os.makedirs(args.outdir, exist_ok=True)
    style(args.font)

    table = dict(FIGURES)
    table.update(COMPARE_FIGURES)
    default = list(COMPARE_FIGURES) if args.compare else list(FIGURES)
    wanted = args.only or default
    written = []
    for name in wanted:
        if name in COMPARE_FIGURES and not args.compare:
            p.error("%s needs --compare" % name)
        if explicit_width:
            args.width = SINGLE_COL if args.column == "single" else DOUBLE_COL
        else:
            args.width = PAGE_WIDTH.get(
                name, DOUBLE_COL if name in WIDE else SINGLE_COL)
        made = table[name](args.out, args)
        if made is None:
            print("skip %s (no rows in %s)" % (name, args.out), file=sys.stderr)
            continue
        fig, stem = made
        if args.iree and stem in ("models", "micro_baselines", "compile_overhead"):
            stem += "_iree"
        for ext in args.format.split(","):
            path = os.path.join(args.outdir, "%s.%s" % (stem, ext.strip()))
            fig.savefig(path)
            written.append(path)
        plt.close(fig)
        # models and micro_speedup also get a single-column variant regardless of
        # --column, so both widths exist and --column single does not clobber
        # the double-column default.
        if name in ("models", "micro_speedup"):
            saved_width = args.width
            args.width = PAGE_WIDTH.get(name + "_col", SINGLE_COL)
            made_col = table[name](args.out, args)
            args.width = saved_width
            if made_col is not None:
                fig_col, stem_col = made_col
                for ext in args.format.split(","):
                    path = os.path.join(args.outdir,
                                         "%s_col.%s" % (stem_col, ext.strip()))
                    fig_col.savefig(path)
                    written.append(path)
                plt.close(fig_col)
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
