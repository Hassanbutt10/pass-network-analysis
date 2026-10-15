"""
Pass Network Analysis - Cycle 7, Day 6 project
Builds Argentina's passing network for Copa America 2024 and finds the hub
player: the one whose removal would cut the most passing routes. This is the
player to deny the ball in a pressing plan.

How it works:
  - Nodes: the 11 players with the most completed open-play passes, placed at
    the average position they passed from
  - Edges: completed passes between two players (both directions combined),
    shown only when a pair connected at least MIN_EDGE times
  - Passing volume: weighted degree (total passes on a player's links)
  - Hub: highest betweenness centrality, where a stronger link counts as a
    shorter route. A player with high betweenness sits on many of the shortest
    routes between other players
  - Only the first 90 minutes of each match, open play only (no corners,
    free kicks, goal kicks or throw-ins)

Limitations: positions are averaged across 6 matches with changing line-ups,
so the picture blends formations. The network shows structure, not
cause and effect: nothing here proves that denying the hub breaks the
team's buildup. Betweenness also depends on the MIN_EDGE cutoff and on a
small sample of matches.

Outputs: argentina_pass_network.png, volume_vs_centrality.png,
pass_network_animation.mp4 (a replay of the real passes: the ball travels
between players and a line is drawn at every pass), pass_network_nodes.csv, pass_network_edges.csv

Reuses: copa_america_2024_matches.csv, copa_america_2024_events.csv
"""

from collections import Counter

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Circle
import imageio_ffmpeg

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

FOCUS_TEAM = "Argentina"
N_NODES = 11
MIN_EDGE = 12  # completed passes between a pair over the tournament

matches = pd.read_csv("copa_america_2024_matches.csv")
events = pd.read_csv("copa_america_2024_events.csv", low_memory=False)
events = events[events["period"].isin([1, 2])].copy()
if "pass_type" not in events.columns:
    events["pass_type"] = np.nan

# ---------- the team's matches in order ----------
tm = matches[(matches["home_team"] == FOCUS_TEAM) | (matches["away_team"] == FOCUS_TEAM)]
tm = tm.sort_values("match_date")
order = tm["match_id"].tolist()
opp_of = {
    r.match_id: (r.away_team if r.home_team == FOCUS_TEAM else r.home_team)
    for r in tm.itertuples()
}
n_matches = len(order)

# ---------- completed open-play passes ----------
p = events[
    (events["team"] == FOCUS_TEAM)
    & (events["type"] == "Pass")
    & events["pass_outcome"].isna()
    & events["pass_recipient"].notna()
].copy()
p = p[p["pass_type"].isna() | p["pass_type"].isin(["Recovery", "Interception"])]
loc = p["location"].astype(str).str.extract(r"\[\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)")
p["x"] = pd.to_numeric(loc[0], errors="coerce")
p["y"] = pd.to_numeric(loc[1], errors="coerce")
p = p.dropna(subset=["x", "y"])

passes_made = p.groupby("player").size().sort_values(ascending=False)
nodes = list(passes_made.index[:N_NODES])
pos = p[p["player"].isin(nodes)].groupby("player")[["x", "y"]].mean()

# ---------- short display names ----------
OVERRIDES = {
    "Lionel Andrés Messi Cuccittini": "Messi",
    "Ángel Fabián Di María Hernández": "Di María",
    "Nahuel Molina Lucero": "Molina",
    "Damián Emiliano Martínez": "E. Martínez",
    "Lautaro Javier Martínez": "Lautaro Martínez",
    "Lisandro Martínez": "Lisandro Martínez",
}
PARTICLES = {"de", "da", "di", "dos", "van", "von", "mac", "lo", "la", "del", "le"}


def short_name(full):
    if full in OVERRIDES:
        return OVERRIDES[full]
    parts = full.split()
    if len(parts) >= 3 and parts[-2].lower() in PARTICLES:
        return " ".join(parts[-2:])
    return parts[-1]


names = {n: short_name(n) for n in nodes}
dupes = Counter(names.values())
for n in nodes:
    if dupes[names[n]] > 1:
        names[n] = f"{n.split()[0]} {names[n]}"

# ---------- edges, cumulative match by match ----------
p2 = p[p["player"].isin(nodes) & p["pass_recipient"].isin(nodes)].copy()
p2["u"] = np.where(p2["player"] < p2["pass_recipient"], p2["player"], p2["pass_recipient"])
p2["v"] = np.where(p2["player"] < p2["pass_recipient"], p2["pass_recipient"], p2["player"])
per_match = p2.groupby(["u", "v", "match_id"]).size().unstack("match_id", fill_value=0)
per_match = per_match.reindex(columns=order, fill_value=0)
cum = per_match.cumsum(axis=1)

final = cum.iloc[:, -1]
edges = list(final[final >= MIN_EDGE].index)
if not edges:
    raise ValueError("No pairs reach MIN_EDGE, lower it")
cum_w = cum.loc[edges].values  # rows = edges, columns = matches

node_idx = {nm: i for i, nm in enumerate(nodes)}
edge_u = np.array([node_idx[u] for u, v in edges])
edge_v = np.array([node_idx[v] for u, v in edges])

# cumulative passes made per player, per match (for "most passes so far")
made_pm = p[p["player"].isin(nodes)].groupby(["player", "match_id"]).size().unstack(
    "match_id", fill_value=0
).reindex(index=nodes, columns=order, fill_value=0).cumsum(axis=1)


def network_stats(weights):
    """strength and betweenness per node for one set of edge weights."""
    strength = np.zeros(len(nodes))
    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    for (u, v), w in zip(zip(edge_u, edge_v), weights):
        if w > 0:
            strength[u] += w
            strength[v] += w
            G.add_edge(int(u), int(v), dist=1.0 / w)
    bet = nx.betweenness_centrality(G, weight="dist", normalized=True)
    bet = np.array([bet[i] for i in range(len(nodes))])
    return strength, bet


def pick_hub(strength, bet):
    if bet.max() > 0:
        return int(np.argmax(bet))
    return int(np.argmax(strength))


step_stats = [network_stats(cum_w[:, k]) for k in range(n_matches)]
strength_f, bet_f = step_stats[-1]
hub_f = pick_hub(strength_f, bet_f)
leader_f = int(np.argmax(made_pm.iloc[:, -1].values))

# ---------- print and save ----------
tbl = pd.DataFrame({
    "player": [names[n] for n in nodes],
    "passes_made": made_pm.iloc[:, -1].values,
    "link_strength": strength_f,
    "betweenness": bet_f.round(3),
}).sort_values("betweenness", ascending=False)
print(f"\n{FOCUS_TEAM} pass network, {n_matches} matches, first 90 minutes, open play:")
print(tbl.to_string(index=False))
tbl.to_csv("pass_network_nodes.csv", index=False)

edge_tbl = pd.DataFrame({
    "player_a": [names[u] for u, v in edges],
    "player_b": [names[v] for u, v in edges],
    "passes": cum_w[:, -1],
}).sort_values("passes", ascending=False)
print("\nStrongest links:")
print(edge_tbl.head(6).to_string(index=False))
edge_tbl.to_csv("pass_network_edges.csv", index=False)

print("\nHub after each match:")
for k in range(n_matches):
    s, b = step_stats[k]
    print(f"  after match {k + 1} (vs {opp_of[order[k]]}): {names[nodes[pick_hub(s, b)]]}")

print(f"\nHub (betweenness): {names[nodes[hub_f]]}")
print(f"Most completed passes: {names[nodes[leader_f]]}")

# ---------- drawing ----------
BG = "#0d1b2a"
LINE = "#35506b"
BLUE = "#4a90d9"
ORANGE = "#f4a300"
MAX_W = cum_w[:, -1].max()
MAX_S = strength_f.max()


def draw_pitch(ax):
    ax.set_facecolor("#10253a")
    kw = dict(fill=False, edgecolor=LINE, linewidth=1.6)
    ax.add_patch(Rectangle((0, 0), 120, 80, **kw))
    ax.plot([60, 60], [0, 80], color=LINE, linewidth=1.6)
    ax.add_patch(Circle((60, 40), 10, **kw))
    ax.add_patch(Rectangle((0, 18), 18, 44, **kw))
    ax.add_patch(Rectangle((102, 18), 18, 44, **kw))
    ax.add_patch(Rectangle((0, 30), 6, 20, **kw))
    ax.add_patch(Rectangle((114, 30), 6, 20, **kw))
    ax.set_xlim(-2, 122)
    ax.set_ylim(82, -2)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


fig = plt.figure(figsize=(10, 10), dpi=108)  # 1080 x 1080 pixels
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0.03, 0.22, 0.94, 0.58])

fig.text(0.5, 0.94, f"Who Does {FOCUS_TEAM}'s Passing Run Through?", ha="center",
         color="white", fontsize=25, fontweight="bold")
banner_t = fig.text(0.5, 0.895, "", ha="center", color="#a8b8c8", fontsize=15)
line1_t = fig.text(0.5, 0.155, "", ha="center", color=ORANGE, fontsize=19, fontweight="bold")
line2_t = fig.text(0.5, 0.105, "", ha="center", color="white", fontsize=15)
fig.text(0.5, 0.035,
         "Dot size = passing volume. Orange = hub (highest betweenness). Line width = passes between two players.\n"
         "First 90 minutes, open play, StatsBomb open data, Copa America 2024.",
         ha="center", color="#a8b8c8", fontsize=10)


def place_labels(xs, ys, sizes, labels):
    """Pick a label position for each node that avoids other labels and dots.
    Units are pitch units (120 x 80). Computed once, so labels stay still.
    Returns {node: (label_x, label_y, needs_leader_line)}."""
    px_per_unit = 7.45
    radius = [np.sqrt(sz / np.pi) * 108 / 72 / px_per_unit for sz in sizes]
    placed = []  # (cx, cy, w, h)
    result = {}
    crowd = [sum(np.hypot(xs[i] - xs[j], ys[i] - ys[j]) < 12 for j in range(len(xs)) if j != i)
             for i in range(len(xs))]
    for i in sorted(range(len(xs)), key=lambda k: -crowd[k]):
        w = 1.4 * len(labels[i]) + 1.0
        h = 3.4
        r = radius[i]
        near = [(0, r + h / 2 + 1.0), (0, -(r + h / 2 + 1.0)),
                (r + w / 2 + 1.0, 0), (-(r + w / 2 + 1.0), 0),
                (r * 0.7 + w * 0.28, r * 0.7 + h / 2 + 0.6), (-(r * 0.7 + w * 0.28), r * 0.7 + h / 2 + 0.6),
                (r * 0.7 + w * 0.28, -(r * 0.7 + h / 2 + 0.6)), (-(r * 0.7 + w * 0.28), -(r * 0.7 + h / 2 + 0.6))]
        cands = [(dx, dy, 1.0) for dx, dy in near] + [(dx * 2.0, dy * 2.0, 2.0) for dx, dy in near]
        best = None
        for ci, (dx, dy, sc) in enumerate(cands):
            cx, cy = xs[i] + dx, ys[i] + dy
            pen = (ci % 8) * 0.4 + (sc - 1.0) * 4.0
            if cx - w / 2 < -2 or cx + w / 2 > 122 or cy - h / 2 < -2 or cy + h / 2 > 82:
                pen += 50
            for (ox, oy, ow, oh) in placed:
                ovx = max(0.0, (w + ow) / 2 - abs(cx - ox))
                ovy = max(0.0, (h + oh) / 2 - abs(cy - oy))
                pen += 12 * ovx * ovy
            for j in range(len(xs)):
                ovx = max(0.0, w / 2 + radius[j] - abs(cx - xs[j]))
                ovy = max(0.0, h / 2 + radius[j] - abs(cy - ys[j]))
                pen += 6 * ovx * ovy
            if best is None or pen < best[0]:
                best = (pen, cx, cy)
        _, cx, cy = best
        gap = np.hypot(max(abs(cx - xs[i]) - w / 2, 0.0), max(abs(cy - ys[i]) - h / 2, 0.0))
        result[i] = (cx, cy, gap > r + 1.5)
        placed.append((cx, cy, w, h))
    return result


def draw_state(sizes, widths, hub, banner, line1, line2, ball=None, flight=None):
    """ball = (x, y) of the ball. flight = (passer_index, (x, y)) draws the pass in flight."""
    ax.clear()
    draw_pitch(ax)
    xs = pos.loc[nodes, "x"].values
    ys = pos.loc[nodes, "y"].values
    for i in range(len(edges)):
        if widths[i] > 0.05:
            ax.plot([xs[edge_u[i]], xs[edge_v[i]]], [ys[edge_u[i]], ys[edge_v[i]]],
                    color="#9db4c9", linewidth=widths[i], alpha=0.65, solid_capstyle="round", zorder=2)
    if flight is not None:
        a_i, (bx, by) = flight
        ax.plot([xs[a_i], bx], [ys[a_i], by], color="white", linewidth=3.2,
                solid_capstyle="round", zorder=5)
    colors = [ORANGE if i == hub else BLUE for i in range(len(nodes))]
    ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="white", linewidths=1.5, zorder=3)
    if hub is not None:
        ax.scatter([xs[hub]], [ys[hub]], s=sizes[hub] * 1.9, facecolors="none",
                   edgecolors=ORANGE, linewidths=3, zorder=3)
    for i, nm in enumerate(nodes):
        lx, ly, leader = label_pos[i]
        if leader:
            ax.plot([xs[i], lx], [ys[i], ly], color="#c9d6e2", linewidth=1, alpha=0.7, zorder=1.5)
        t = ax.text(lx, ly, names[nm], ha="center", va="center",
                    color="white", fontsize=11, fontweight="bold", zorder=4)
        t.set_path_effects([pe.withStroke(linewidth=3, foreground=BG)])
    if ball is not None:
        ax.scatter([ball[0]], [ball[1]], s=190, c="white", edgecolors=BG,
                   linewidths=2, zorder=6)
    banner_t.set_text(banner)
    line1_t.set_text(line1)
    line2_t.set_text(line2)


def size_of(strength):
    return 260 + 1330 * strength / MAX_S


def width_of(w):
    return np.where(w > 0, 1.4 + 10.6 * w / MAX_W, 0.0)


def lines_for(hub_i, leader_i, when):
    l1 = f"Hub {when}: {names[nodes[hub_i]]}"
    if hub_i == leader_i:
        l2 = f"Also the most completed passes: {names[nodes[leader_i]]}"
    else:
        l2 = f"Most completed passes {when}: {names[nodes[leader_i]]}"
    return l1, l2


# label positions are worked out once from the final dot sizes, so they never jump
label_pos = place_labels(pos.loc[nodes, "x"].values, pos.loc[nodes, "y"].values,
                         size_of(strength_f), [names[n] for n in nodes])

# static PNG of the final network
l1, l2 = lines_for(hub_f, leader_f, "overall")
draw_state(size_of(strength_f), width_of(cum_w[:, -1]), hub_f,
           f"{n_matches} matches, Copa America 2024", l1, l2)
net_png = f"{FOCUS_TEAM.lower().replace(' ', '_')}_pass_network.png"
fig.savefig(net_png, dpi=108, facecolor=BG)
print(f"\nSaved chart: {net_png}")

# ---------- animation: replay of the real passes ----------
# Phase 1: the first passes one at a time, the ball visibly travels and a line
# is drawn. Phase 2: the rest of the tournament fast-forwarded in order.
# Phase 3: hold on the finished network.
xs_n = pos.loc[nodes, "x"].values
ys_n = pos.loc[nodes, "y"].values

p2["m"] = p2["match_id"].map({mid: i for i, mid in enumerate(order)})
if "index" in p2.columns:
    sort_cols = ["m", "index"]
elif all(c in p2.columns for c in ["period", "minute", "second"]):
    sort_cols = ["m", "period", "minute", "second"]
else:
    sort_cols = ["m"]
seq = p2.sort_values(sort_cols)
edge_index = {e: i for i, e in enumerate(edges)}
seq_rows = []
for player, recip, u, v, m in seq[["player", "pass_recipient", "u", "v", "m"]].values:
    if (u, v) in edge_index:
        seq_rows.append((node_idx[player], node_idx[recip], edge_index[(u, v)], int(m)))
print(f"\nReplaying {len(seq_rows)} passes between the 11 players.")


def ease(t):
    return t * t * (3 - 2 * t)


HUB_AFTER = 100  # do not name a hub until this many passes have been played


def hub_now(W):
    st, bt = network_stats(W)
    if W.sum() < HUB_AFTER or bt.max() <= 0:
        return None, st
    return int(np.argmax(bt)), st


def banner_for(k):
    return f"Match {k + 1} of {n_matches}: vs {opp_of[order[k]]}"


def lines_now(hub_i, n_played):
    l1 = f"Hub so far: {names[nodes[hub_i]]}" if hub_i is not None else "Building the network..."
    return l1, f"Passes played: {n_played:,}"


PH1, TRAVEL, HOLD, STEPS, FINAL_HOLD = 12, 7, 2, 130, 50
PH1 = min(PH1, len(seq_rows))
W = np.zeros(len(edges))
states = []

for j in range(PH1):
    a, b2, e, k = seq_rows[j]
    hub_i, st = hub_now(W)
    l1, l2 = lines_now(hub_i, j)
    for f in range(TRAVEL):
        t = ease((f + 1) / TRAVEL)
        ball = (xs_n[a] + (xs_n[b2] - xs_n[a]) * t, ys_n[a] + (ys_n[b2] - ys_n[a]) * t)
        states.append((size_of(st), width_of(W), hub_i, banner_for(k), l1, l2, ball, (a, ball)))
    W[e] += 1
    hub_i, st = hub_now(W)
    l1, l2 = lines_now(hub_i, j + 1)
    for _ in range(HOLD):
        states.append((size_of(st), width_of(W), hub_i, banner_for(k), l1, l2,
                       (xs_n[b2], ys_n[b2]), None))

rest = seq_rows[PH1:]
bounds = np.linspace(0, len(rest), STEPS + 1).astype(int)
for si in range(STEPS):
    batch = rest[bounds[si]:bounds[si + 1]]
    if not batch:
        continue
    for (a, b2, e, k) in batch:
        W[e] += 1
    a, b2, e, k = batch[-1]
    hub_i, st = hub_now(W)
    l1, l2 = lines_now(hub_i, PH1 + bounds[si + 1])
    states.append((size_of(st), width_of(W), hub_i, banner_for(k), l1, l2,
                   (xs_n[b2], ys_n[b2]), (a, (xs_n[b2], ys_n[b2]))))

final_banner = f"Press target: {names[nodes[hub_f]]}"
l1, l2 = lines_for(hub_f, leader_f, "overall")
states += [(size_of(strength_f), width_of(cum_w[:, -1]), hub_f, final_banner, l1, l2, None, None)] * FINAL_HOLD


def update(i):
    draw_state(*states[i])


FPS = 20
anim = FuncAnimation(fig, update, frames=len(states), interval=1000 / FPS)
writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=4000,
                      extra_args=["-pix_fmt", "yuv420p"])
print(f"Rendering {len(states)} frames ({len(states) / FPS:.0f} seconds), this takes a minute...")
anim.save("pass_network_animation.mp4", writer=writer, dpi=108,
          savefig_kwargs={"facecolor": BG})
print("Saved video: pass_network_animation.mp4")
plt.close(fig)

# ---------- chart 2: passing volume vs network centrality ----------
c = tbl.copy()
c["vol_pct"] = c["link_strength"] / c["link_strength"].max() * 100
c["bet_pct"] = c["betweenness"] / max(c["betweenness"].max(), 1e-9) * 100
c = c.sort_values("bet_pct", ascending=True)

fig2, ax2 = plt.subplots(figsize=(10, 8))
fig2.patch.set_facecolor(BG)
ax2.set_facecolor(BG)
yy = np.arange(len(c))
ax2.barh(yy + 0.2, c["vol_pct"], height=0.38, color=BLUE, label="Passing volume (link strength)")
ax2.barh(yy - 0.2, c["bet_pct"], height=0.38, color=ORANGE, label="Network centrality (betweenness)")
ax2.set_yticks(yy)
ax2.set_yticklabels(c["player"], color="white", fontsize=12)
ax2.set_xlabel("% of the highest value in each group", color="white", fontsize=11)
ax2.tick_params(axis="x", colors="white")
ax2.tick_params(axis="y", length=0)
for spine in ax2.spines.values():
    spine.set_color("#3a4a5a")
ax2.legend(facecolor=BG, labelcolor="white", edgecolor="#3a4a5a", loc="lower right")
fig2.suptitle(f"{FOCUS_TEAM}: Passing Volume vs Network Centrality", color="white",
              fontsize=16, fontweight="bold", y=0.97)
ax2.set_title("Link strength and betweenness, each scaled so the highest player = 100%",
              color="#a8b8c8", fontsize=11, pad=10)
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig2.savefig("volume_vs_centrality.png", dpi=150, facecolor=BG)
print("Saved chart: volume_vs_centrality.png")