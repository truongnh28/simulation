"""Additional figures for dtree.html, knn.html, svm.html"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    'figure.dpi': 160, 'savefig.dpi': 160,
    'savefig.bbox': 'tight', 'savefig.transparent': False,
    'figure.facecolor': '#0f172a', 'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#475569', 'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8', 'ytick.color': '#94a3b8',
    'text.color': '#e2e8f0', 'grid.color': '#334155', 'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans', 'font.size': 11,
})
DARK_BG = '#0f172a'; PANEL = '#1e293b'
GREEN = '#4ade80'; BLUE = '#60a5fa'; YELLOW = '#fbbf24'
RED = '#f87171'; PURPLE = '#c084fc'; TEAL = '#2dd4bf'; ORANGE = '#fb923c'

def save(name):
    plt.savefig(f'/Users/lap13954/Documents/q/MLA/imgs/{name}', facecolor=DARK_BG)
    plt.close()
    print(f'  saved {name}')

# ─────────────────────────────────────────────────────────────────────────────
# A. ENTROPY & GINI vs P(+)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_facecolor(PANEL)
p = np.linspace(0.001, 0.999, 500)
entropy = -p*np.log2(p) - (1-p)*np.log2(1-p)
gini    = 1 - (p**2 + (1-p)**2)
err     = np.minimum(p, 1-p)

ax.plot(p, entropy, color=BLUE,   lw=2.5, label='Entropy H(p) = −p log₂p − (1−p)log₂(1−p)')
ax.plot(p, gini,    color=GREEN,  lw=2.5, label='Gini G(p) = 1 − p² − (1−p)²')
ax.plot(p, err,     color=ORANGE, lw=2.0, linestyle='--', label='Misclassification Error = min(p, 1−p)')

ax.axvline(0.5, color='#475569', lw=0.8, linestyle=':')
ax.annotate('Max (p=0.5)', xy=(0.5, 1.0), xytext=(0.6, 0.9),
            color=BLUE, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))

ax.fill_between(p, 0, entropy, alpha=0.05, color=BLUE)
ax.set_xlabel('P(+) — xác suất lớp dương', fontsize=12)
ax.set_ylabel('Impurity measure', fontsize=12)
ax.set_title('Entropy vs Gini vs Misclassification Error', color='#e2e8f0', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.3, loc='upper center')
ax.set_xlim(0,1); ax.set_ylim(0, 1.1)
ax.grid(True)
ax.text(0.02, 0.05, 'Pure\n(H=0, G=0)', color='#94a3b8', fontsize=8, transform=ax.transAxes)
ax.text(0.85, 0.05, 'Pure\n(H=0, G=0)', color='#94a3b8', fontsize=8, transform=ax.transAxes)
plt.tight_layout()
save('entropy_gini.png')

# ─────────────────────────────────────────────────────────────────────────────
# B. KNN DECISION BOUNDARY — K=1 vs K=15
# ─────────────────────────────────────────────────────────────────────────────

np.random.seed(42)
N = 80
X_pos = np.random.randn(N//2, 2) * 0.9 + np.array([1.0, 1.0])
X_neg = np.random.randn(N//2, 2) * 0.9 + np.array([-0.5, -0.5])
X_all = np.vstack([X_pos, X_neg])
y_all = np.array([1]*(N//2) + [0]*(N//2))

# KNN decision boundary — manual KNN without sklearn
def knn_predict(X_train, y_train, X_test, k):
    dists = np.sqrt(((X_test[:,None,:] - X_train[None,:,:])**2).sum(axis=2))
    idx = np.argsort(dists, axis=1)[:, :k]
    votes = y_train[idx]
    return (votes.mean(axis=1) >= 0.5).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(DARK_BG)

xx, yy = np.meshgrid(np.linspace(-3, 4, 120), np.linspace(-3, 4, 120))
grid = np.c_[xx.ravel(), yy.ravel()]

for ax, k, title_note in zip(axes, [1, 15],
    ['K=1 → Overfit\n(jagged boundary)', 'K=15 → Smoother\n(less variance)']):
    ax.set_facecolor(PANEL)
    Z = knn_predict(X_all, y_all, grid, k).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.35, levels=1, colors=['#1a2a4a', '#1a3a2a'])
    ax.contour(xx, yy, Z, colors=['#60a5fa'], linewidths=1.8, alpha=0.9)

    ax.scatter(X_pos[:,0], X_pos[:,1], c=GREEN,  edgecolors='white', lw=0.5, s=50, zorder=5, label='Class +1')
    ax.scatter(X_neg[:,0], X_neg[:,1], c=RED,    edgecolors='white', lw=0.5, s=50, zorder=5, label='Class −1')
    ax.set_title(f'KNN  K={k} — {title_note}', color='#e2e8f0', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('Feature 1'); ax.set_ylabel('Feature 2')
    ax.legend(fontsize=9, framealpha=0.3, loc='lower right')

fig.suptitle('KNN Decision Boundaries: Effect of K on Bias-Variance', color='#e2e8f0', fontsize=13, fontweight='bold')
plt.tight_layout()
save('knn_decision_boundary.png')

# ─────────────────────────────────────────────────────────────────────────────
# C. KD-TREE STRUCTURE — building diagram
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(DARK_BG)

points = {
    'A': (6, 7), 'B': (2, 4), 'C': (3, 2),
    'D': (8, 5), 'E': (4, 1), 'F': (9, 7)
}

# Left: 2D space with split lines
ax = axes[0]; ax.set_facecolor(PANEL)
ax.set_xlim(0, 11); ax.set_ylim(0, 9)

# Plot points
colors_pts = {'A': YELLOW, 'B': GREEN, 'C': GREEN, 'D': ORANGE, 'E': GREEN, 'F': ORANGE}
for name, (x,y) in points.items():
    ax.plot(x, y, 'o', color=colors_pts[name], ms=12, zorder=5)
    ax.text(x+0.2, y+0.2, name, color=colors_pts[name], fontsize=11, fontweight='bold', zorder=6)

# Split 1: vertical x=6 (median of all x)
ax.axvline(6, color=BLUE, lw=2, linestyle='-', zorder=3)
ax.text(6.1, 8.5, 'x=6 (split 1)\naxis=x', color=BLUE, fontsize=8.5)

# Split 2 left: horizontal y=4 (median of {B(2,4), C(3,2), E(4,1)})
ax.hlines(4, 0, 6, color=GREEN, lw=1.8, linestyle='--', zorder=3)
ax.text(0.1, 4.2, 'y=4 (split 2L)\naxis=y', color=GREEN, fontsize=8)

# Split 2 right: horizontal y=6 (median of {A(6,7), D(8,5), F(9,7)}) — approx
ax.hlines(6, 6, 11, color=ORANGE, lw=1.8, linestyle='--', zorder=3)
ax.text(6.1, 6.2, 'y=6 (split 2R)\naxis=y', color=ORANGE, fontsize=8)

ax.set_title('KD-Tree: 2D Space Partitioning', color='#e2e8f0', fontsize=11, fontweight='bold')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.grid(True, alpha=0.3)

# Right: tree structure
ax = axes[1]; ax.set_facecolor(DARK_BG); ax.axis('off')
ax.set_xlim(0, 10); ax.set_ylim(0, 9)

def node_box(ax, x, y, text, color, w=2.2, h=0.65):
    box = FancyBboxPatch((x-w/2, y-h/2), w, h,
                          boxstyle='round,pad=0.1', facecolor=color,
                          edgecolor='white', linewidth=1.2, zorder=5)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9,
            color='white', fontweight='bold', zorder=6)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2+0.33), xytext=(x1, y1-0.33),
                arrowprops=dict(arrowstyle='->', color='#94a3b8', lw=1.2))

# Tree nodes
node_box(ax, 5, 8.2, 'A (6,7)\naxis=x, split=6', BLUE)
node_box(ax, 2.5, 6.0, 'B (2,4)\naxis=y, split=4', GREEN)
node_box(ax, 7.5, 6.0, 'D (8,5)\naxis=y, split=6', ORANGE)
node_box(ax, 1.5, 3.8, 'C (3,2)\nleaf', '#166534')
node_box(ax, 3.5, 3.8, 'E (4,1)\nleaf', '#166534')
node_box(ax, 6.5, 3.8, 'F (9,7)\nleaf', '#92400e')

arrow(ax, 5, 8.2, 2.5, 6.0)
arrow(ax, 5, 8.2, 7.5, 6.0)
arrow(ax, 2.5, 6.0, 1.5, 3.8)
arrow(ax, 2.5, 6.0, 3.5, 3.8)
arrow(ax, 7.5, 6.0, 6.5, 3.8)

ax.text(3.5, 7.2, 'x ≤ 6', color='#94a3b8', fontsize=9)
ax.text(6.2, 7.2, 'x > 6', color='#94a3b8', fontsize=9)
ax.text(0.8, 4.9, 'y ≤ 4', color='#94a3b8', fontsize=9)
ax.text(3.0, 4.9, 'y > 4', color='#94a3b8', fontsize=9)
ax.text(5.8, 4.9, 'y ≤ 6', color='#94a3b8', fontsize=9)

ax.set_title('KD-Tree Structure', color='#e2e8f0', fontsize=11, fontweight='bold')

fig.suptitle('KD-Tree: Space Partitioning & Tree Construction', color='#e2e8f0', fontsize=13, fontweight='bold')
plt.tight_layout()
save('kdtree.png')

# ─────────────────────────────────────────────────────────────────────────────
# D. DECISION TREE SPLIT EXAMPLE
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
fig.patch.set_facecolor(DARK_BG)

np.random.seed(5)
# Simple dataset: Outlook, Temperature → Play Tennis
# Simplified: x=Humidity, y=Temp, label=Play/No-play
X_play  = np.column_stack([np.random.uniform(30, 75, 9),  np.random.uniform(65, 90, 9)])
X_noplay= np.column_stack([np.random.uniform(70, 95, 5),  np.random.uniform(60, 85, 5)])

ax = axes[0]; ax.set_facecolor(PANEL)
ax.scatter(X_play[:,0],   X_play[:,1],   c=GREEN, s=80, edgecolors='white', lw=0.5, zorder=5, label='Play Tennis')
ax.scatter(X_noplay[:,0], X_noplay[:,1], c=RED,   s=80, edgecolors='white', lw=0.5, zorder=5, marker='s', label='No Play')

# Best split line
ax.axvline(72, color=YELLOW, lw=2.2, linestyle='--', zorder=3, label='Best split: Humidity ≤ 72')
ax.text(73, 91, 'High Humidity\n→ No Play', color=RED, fontsize=8.5)
ax.text(40, 91, 'Low Humidity\n→ Play', color=GREEN, fontsize=8.5)
ax.set_xlabel('Humidity (%)'); ax.set_ylabel('Temperature (°F)')
ax.set_title('Decision Tree: Finding Best Split\n(max Information Gain)', color='#e2e8f0', fontsize=10.5, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.3, loc='lower right')
ax.grid(True, alpha=0.3)

# Right: Information Gain for different split thresholds
ax = axes[1]; ax.set_facecolor(PANEL)
thresholds = np.linspace(35, 95, 50)
X = np.vstack([X_play, X_noplay])
y = np.array([1]*9 + [0]*5)
def ig(X, y, thresh):
    mask = X[:,0] <= thresh
    n = len(y); n_l = mask.sum(); n_r = (~mask).sum()
    if n_l==0 or n_r==0: return 0
    def ent(labels):
        _, c = np.unique(labels, return_counts=True)
        p = c/c.sum()
        return -np.sum(p*np.log2(p+1e-12))
    return ent(y) - (n_l/n)*ent(y[mask]) - (n_r/n)*ent(y[~mask])

igs = [ig(X, y, t) for t in thresholds]
ax.plot(thresholds, igs, color=YELLOW, lw=2.5)
ax.fill_between(thresholds, 0, igs, alpha=0.15, color=YELLOW)
best_t = thresholds[np.argmax(igs)]
ax.axvline(best_t, color=GREEN, lw=1.5, linestyle='--')
ax.annotate(f'Best split\nHumidity≤{best_t:.0f}%', xy=(best_t, max(igs)),
            xytext=(best_t+8, max(igs)*0.85),
            color=GREEN, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.0))
ax.set_xlabel('Split threshold — Humidity (%)')
ax.set_ylabel('Information Gain (bits)')
ax.set_title('Information Gain for All Thresholds\n(Exhaustive search per feature)', color='#e2e8f0', fontsize=10.5, fontweight='bold')
ax.grid(True)

fig.suptitle('Decision Tree: Best Split Selection via Information Gain', color='#e2e8f0', fontsize=13, fontweight='bold')
plt.tight_layout()
save('dtree_split.png')

# ─────────────────────────────────────────────────────────────────────────────
# E. KNN: Effect of distance metric
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.set_facecolor(PANEL)

# Show unit "circles" for L1, L2, Linf
theta = np.linspace(0, 2*np.pi, 500)
# L2
ax.plot(np.cos(theta), np.sin(theta), color=BLUE, lw=2.5, label='L₂ (Euclidean) — circle')
# L1 (diamond): |x|+|y|=1
l1x = [1,0,-1,0,1]; l1y = [0,1,0,-1,0]
ax.plot(l1x, l1y, color=GREEN, lw=2.5, label='L₁ (Manhattan) — diamond')
# Linf (square): max(|x|,|y|)=1
ax.plot([1,1,-1,-1,1], [1,-1,-1,1,1], color=YELLOW, lw=2.5, label='L∞ (Chebyshev) — square')

ax.axhline(0, color='#475569', lw=0.6)
ax.axvline(0, color='#475569', lw=0.6)
ax.set_aspect('equal')
ax.set_xlim(-1.5, 1.5); ax.set_ylim(-1.5, 1.5)
ax.set_title('Unit "Ball" for Different Distance Metrics\n(All points within distance=1 from origin)',
             color='#e2e8f0', fontsize=12, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.3, loc='upper right')
ax.grid(True)

# Annotations
ax.text(0.15, 0.95, 'd=√(x²+y²)', color=BLUE, fontsize=8.5, transform=ax.transAxes)
ax.text(0.15, 0.88, 'd=|x|+|y|', color=GREEN, fontsize=8.5, transform=ax.transAxes)
ax.text(0.15, 0.81, 'd=max(|x|,|y|)', color=YELLOW, fontsize=8.5, transform=ax.transAxes)
plt.tight_layout()
save('distance_metrics.png')

print('\nAll additional figures generated!')
