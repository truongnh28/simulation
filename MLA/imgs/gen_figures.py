"""
Generate all MLA figures as high-quality PNGs.
Run from: /Users/lap13954/Documents/q/MLA/imgs/
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe

# ── style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 160,
    'savefig.dpi': 160,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'figure.facecolor': '#0f172a',
    'axes.facecolor': '#1e293b',
    'axes.edgecolor': '#475569',
    'axes.labelcolor': '#e2e8f0',
    'xtick.color': '#94a3b8',
    'ytick.color': '#94a3b8',
    'text.color': '#e2e8f0',
    'grid.color': '#334155',
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
    'font.size': 11,
})

DARK_BG  = '#0f172a'
PANEL    = '#1e293b'
GREEN    = '#4ade80'
BLUE     = '#60a5fa'
YELLOW   = '#fbbf24'
RED      = '#f87171'
PURPLE   = '#c084fc'
TEAL     = '#2dd4bf'
ORANGE   = '#fb923c'
GRID_C   = '#334155'

def save(name):
    plt.savefig(f'/Users/lap13954/Documents/q/MLA/imgs/{name}', facecolor=DARK_BG)
    plt.close()
    print(f'  saved {name}')

# ─────────────────────────────────────────────────────────────────────────────
# 1. ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def sigmoid(z): return 1/(1+np.exp(-z))
def sigmoid_d(z): s=sigmoid(z); return s*(1-s)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)
z = np.linspace(-5, 5, 400)

# Left: activation values
ax = axes[0]
ax.set_facecolor(PANEL)
ax.plot(z, sigmoid(z),                  color=BLUE,   lw=2.2, label='Sigmoid  σ(z)')
ax.plot(z, np.tanh(z),                  color=GREEN,  lw=2.2, label='Tanh')
ax.plot(z, np.maximum(0,z),             color=YELLOW, lw=2.2, label='ReLU')
ax.plot(z, np.where(z>=0,z,0.01*z),     color=ORANGE, lw=2.2, label='Leaky ReLU (α=0.01)')
# GELU approx
gelu = 0.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+0.044715*z**3)))
ax.plot(z, gelu,                         color=PURPLE, lw=2.2, label='GELU')
ax.axhline(0, color='#475569', lw=0.8)
ax.axvline(0, color='#475569', lw=0.8)
ax.set_xlim(-5,5); ax.set_ylim(-1.6, 1.6)
ax.set_xlabel('z'); ax.set_ylabel('f(z)')
ax.set_title('Activation Functions', color='#e2e8f0', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.3, loc='upper left')
ax.grid(True)

# Right: derivatives
ax = axes[1]
ax.set_facecolor(PANEL)
drelu   = np.where(z>=0, 1.0, 0.0)
dleaky  = np.where(z>=0, 1.0, 0.01)
dgelu_approx = sigmoid(1.702*z)*(1+1.702*z*(1-sigmoid(1.702*z)))
ax.plot(z, sigmoid_d(z),    color=BLUE,   lw=2.2, label="σ'(z)  ≤ 0.25")
ax.plot(z, 1-np.tanh(z)**2, color=GREEN,  lw=2.2, label="tanh'(z) ≤ 1")
ax.plot(z, drelu,            color=YELLOW, lw=2.2, label="ReLU'(z)")
ax.plot(z, dleaky,           color=ORANGE, lw=2.2, label="Leaky'(z)")
ax.plot(z, dgelu_approx,     color=PURPLE, lw=2.2, label="GELU'(z)")
ax.axhline(0, color='#475569', lw=0.8)
ax.axhline(0.25, color=BLUE, lw=0.8, linestyle='--', alpha=0.5)
ax.annotate('max = 0.25', xy=(-4.8, 0.27), color=BLUE, fontsize=8)
ax.set_xlim(-5,5); ax.set_ylim(-0.1, 1.15)
ax.set_xlabel('z'); ax.set_ylabel("f'(z)")
ax.set_title('Derivatives (Gradient Flow)', color='#e2e8f0', fontsize=13, fontweight='bold')
ax.legend(fontsize=9, framealpha=0.3, loc='upper right')
ax.grid(True)

fig.suptitle('Activation Functions & Vanishing Gradient Risk', color='#e2e8f0', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save('activation_functions.png')

# ─────────────────────────────────────────────────────────────────────────────
# 2. VANISHING GRADIENT — MAGNITUDE PER LAYER
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_facecolor(PANEL)
layers = np.arange(1, 21)
sigmoid_decay = 0.25**layers
tanh_decay    = 0.9**layers      # tanh max derivative ~1 but typical ~0.8
relu_flat     = np.ones_like(layers, dtype=float)
relu_flat[relu_flat>0] = 1.0     # ReLU keeps gradient if neuron active

ax.semilogy(layers, sigmoid_decay, 'o-', color=BLUE,   lw=2, ms=5, label='Sigmoid  ∝ (0.25)ᴸ')
ax.semilogy(layers, tanh_decay,    's-', color=GREEN,  lw=2, ms=5, label='Tanh  ∝ (0.9)ᴸ')
ax.semilogy(layers, relu_flat,     '^-', color=YELLOW, lw=2, ms=5, label='ReLU  ≈ 1 (nếu không chết)')
# Residual: gradient highway
res_decay = 1 + sigmoid_decay   # simplified residual
ax.semilogy(layers, np.ones_like(layers)*0.9, '--', color=TEAL, lw=2, label='ResNet (gradient highway ≥ 1)')

ax.axhline(1e-3, color='#ef4444', lw=0.8, linestyle=':', alpha=0.7)
ax.annotate('Vanishing zone', xy=(12, 1.2e-3), color='#ef4444', fontsize=8)
ax.set_xlabel('Layer depth L (from output)', fontsize=11)
ax.set_ylabel('|∂L/∂W⁽¹⁾|  (log scale)', fontsize=11)
ax.set_title('Vanishing Gradient: Magnitude Decay per Layer', color='#e2e8f0', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.3)
ax.grid(True, which='both')
ax.set_xlim(1, 20)
plt.tight_layout()
save('vanishing_gradient.png')

# ─────────────────────────────────────────────────────────────────────────────
# 3. WEIGHT INITIALIZATION — XAVIER VS HE
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
n_in = 512
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.patch.set_facecolor(DARK_BG)

configs = [
    ('Zero init\nw = 0', np.zeros(n_in*n_in), RED,
     'Symmetry problem:\nAll neurons identical!'),
    ('Xavier init\nVar = 2/(n_in+n_out)', np.random.randn(n_in*n_in)*np.sqrt(2/(n_in+n_in)), GREEN,
     'Good for Sigmoid/Tanh\nVar(output) ≈ Var(input)'),
    ('He init\nVar = 2/n_in', np.random.randn(n_in*n_in)*np.sqrt(2/n_in), YELLOW,
     'Good for ReLU\nCompensates 50% zeros'),
]

for ax, (title, data, color, note) in zip(axes, configs):
    ax.set_facecolor(PANEL)
    if np.std(data) < 1e-10:
        ax.bar([0], [n_in*n_in], color=RED, width=0.02)
        ax.set_xlim(-0.1, 0.1)
        ax.text(0, n_in*n_in*0.5, 'All = 0', ha='center', va='center', color='#fff', fontsize=11)
    else:
        ax.hist(data, bins=80, color=color, alpha=0.8, density=True, edgecolor='none')
        mu, sig = np.mean(data), np.std(data)
        x_range = np.linspace(mu-4*sig, mu+4*sig, 200)
        def norm_pdf(x, mu, sig): return np.exp(-0.5*((x-mu)/sig)**2)/(sig*np.sqrt(2*np.pi))
        ax.plot(x_range, norm_pdf(x_range, mu, sig), color='white', lw=1.5, alpha=0.7)
        ax.text(0.97, 0.95, f'σ={sig:.3f}', transform=ax.transAxes,
                ha='right', va='top', color='white', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0f172a', alpha=0.7))
    ax.set_title(title, color='#e2e8f0', fontsize=10.5, fontweight='bold')
    ax.text(0.5, -0.22, note, transform=ax.transAxes, ha='center', va='top',
            color='#94a3b8', fontsize=8.5, style='italic')
    ax.grid(True, axis='y')
    ax.set_ylabel('Density')

fig.suptitle('Weight Initialization Strategies (n_in = 512)', color='#e2e8f0', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0,0.05,1,1])
save('weight_init.png')

# ─────────────────────────────────────────────────────────────────────────────
# 4. BATCH NORMALIZATION: ACTIVATION DISTRIBUTION BEFORE/AFTER
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(7)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.patch.set_facecolor(DARK_BG)

# Layer 1 activations - some shifted/skewed
z1 = np.random.normal(3.5, 2.0, 5000)          # shifted mean
z2 = np.random.normal(0, 0.25, 5000)           # very small variance (saturated)
# After BN
gamma, beta = 1.2, 0.1
z_norm = (z1 - z1.mean())/z1.std()
z_bn   = gamma * z_norm + beta

ax = axes[0]; ax.set_facecolor(PANEL)
ax.hist(z1, bins=60, color=RED, alpha=0.8, density=True, edgecolor='none')
ax.set_title('Before BN — Layer 5\n(Mean shift: μ=3.5)', color='#e2e8f0', fontsize=10.5, fontweight='bold')
ax.axvline(z1.mean(), color='white', lw=1.5, linestyle='--')
ax.text(z1.mean()+0.15, ax.get_ylim()[1]*0.9, f'μ={z1.mean():.1f}', color='white', fontsize=9)
ax.grid(True, axis='y'); ax.set_xlabel('Activation value')

ax = axes[1]; ax.set_facecolor(PANEL)
ax.hist(z_norm, bins=60, color=GREEN, alpha=0.8, density=True, edgecolor='none')
ax.set_title('After Normalize\n(μ≈0, σ²≈1)', color='#e2e8f0', fontsize=10.5, fontweight='bold')
ax.axvline(0, color='white', lw=1.5, linestyle='--')
ax.text(0.15, ax.get_ylim()[1]*0.9, 'μ≈0', color='white', fontsize=9)
ax.grid(True, axis='y'); ax.set_xlabel('Activation value')

ax = axes[2]; ax.set_facecolor(PANEL)
ax.hist(z_bn, bins=60, color=TEAL, alpha=0.8, density=True, edgecolor='none')
ax.set_title(f'After Scale & Shift\n(γ={gamma}, β={beta} — learnable)', color='#e2e8f0', fontsize=10.5, fontweight='bold')
ax.axvline(z_bn.mean(), color='white', lw=1.5, linestyle='--')
ax.text(z_bn.mean()+0.1, ax.get_ylim()[1]*0.9, f'μ≈{z_bn.mean():.2f}', color='white', fontsize=9)
ax.grid(True, axis='y'); ax.set_xlabel('Activation value')

fig.suptitle('Batch Normalization: Internal Covariate Shift Fix', color='#e2e8f0', fontsize=13, fontweight='bold')
plt.tight_layout()
save('batchnorm.png')

# ─────────────────────────────────────────────────────────────────────────────
# 5. SGD vs MOMENTUM vs ADAM — CONVERGENCE ON RAVINE LOSS LANDSCAPE
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(1)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.patch.set_facecolor(DARK_BG)

# Elongated quadratic (ravine)  f(x,y) = 50*x^2 + y^2
X = np.linspace(-1.2, 1.2, 300)
Y = np.linspace(-12, 12, 300)
Xg, Yg = np.meshgrid(X, Y)
Z = 50*Xg**2 + Yg**2

def grad_f(x, y): return np.array([100*x, 2*y])

# SGD
def sgd_path(x0, y0, lr=0.009, steps=60):
    path = [(x0, y0)]
    x, y = x0, y0
    for _ in range(steps):
        g = grad_f(x,y)
        x -= lr*g[0]; y -= lr*g[1]
        path.append((x,y))
    return path

def momentum_path(x0, y0, lr=0.009, mu=0.9, steps=60):
    path = [(x0, y0)]
    x, y = x0, y0; vx, vy = 0, 0
    for _ in range(steps):
        g = grad_f(x,y)
        vx = mu*vx - lr*g[0]; vy = mu*vy - lr*g[1]
        x += vx; y += vy
        path.append((x,y))
    return path

def adam_path(x0, y0, lr=0.08, b1=0.9, b2=0.999, eps=1e-8, steps=60):
    path = [(x0, y0)]
    x, y = x0, y0; mx, my, vx, vy = 0,0,0,0
    for t in range(1, steps+1):
        g = grad_f(x,y)
        mx = b1*mx+(1-b1)*g[0]; my = b1*my+(1-b1)*g[1]
        vx = b2*vx+(1-b2)*g[0]**2; vy = b2*vy+(1-b2)*g[1]**2
        mxh = mx/(1-b1**t); myh = my/(1-b1**t)
        vxh = vx/(1-b2**t); vyh = vy/(1-b2**t)
        x -= lr*mxh/(np.sqrt(vxh)+eps); y -= lr*myh/(np.sqrt(vyh)+eps)
        path.append((x,y))
    return path

titles = ['SGD (lr=0.009)\nzig-zag oscillation', 'SGD + Momentum (μ=0.9)\nsmooth trajectory', 'Adam\nfast adaptive convergence']
paths_fn = [lambda: sgd_path(-1,10), lambda: momentum_path(-1,10), lambda: adam_path(-1,10)]
colors  = [RED, YELLOW, GREEN]

for ax, title, path_fn, col in zip(axes, titles, paths_fn, colors):
    ax.set_facecolor(PANEL)
    cnt = ax.contourf(Xg, Yg, np.log(Z+1), levels=20, cmap='Blues', alpha=0.7)
    ax.contour(Xg, Yg, np.log(Z+1), levels=10, colors='#475569', linewidths=0.4)
    path = path_fn()
    xs, ys = zip(*path)
    ax.plot(xs, ys, '-', color=col, lw=1.8, alpha=0.9)
    ax.plot(xs[0], ys[0], 'o', color='white', ms=7, zorder=5)
    ax.plot(0, 0, '*', color='#fbbf24', ms=12, zorder=5)
    ax.set_title(title, color='#e2e8f0', fontsize=10.5, fontweight='bold')
    ax.set_xlabel('w₁ (slow axis)'); ax.set_ylabel('w₂ (fast axis)')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-12, 12)
    ax.annotate('start', xy=(xs[0], ys[0]), xytext=(0.3, 8), color='white', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='white', lw=0.8))
    ax.annotate('★ min', xy=(0,0), xytext=(0.3, -5), color='#fbbf24', fontsize=8,
                arrowprops=dict(arrowstyle='->', color='#fbbf24', lw=0.8))

fig.suptitle('Optimization Algorithms on Elongated Loss Landscape (Ravine)', color='#e2e8f0', fontsize=13, fontweight='bold')
plt.tight_layout()
save('optimization_paths.png')

# ─────────────────────────────────────────────────────────────────────────────
# 6. BIAS-VARIANCE TRADEOFF
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_facecolor(PANEL)

complexity = np.linspace(0.1, 10, 200)
bias2      = 4.0 / complexity**1.2
variance   = 0.08 * complexity**1.8
noise      = 0.5 * np.ones_like(complexity)
total      = bias2 + variance + noise

ax.fill_between(complexity, total, noise, color=RED,    alpha=0.15)
ax.plot(complexity, bias2,    color=BLUE,   lw=2.5, label='Bias²  (↓ with complexity)')
ax.plot(complexity, variance, color=ORANGE, lw=2.5, label='Variance  (↑ with complexity)')
ax.plot(complexity, noise,    color='#64748b', lw=1.8, linestyle='--', label='Irreducible noise  σ²ε')
ax.plot(complexity, total,    color=GREEN,  lw=3,   label='Total Error = Bias²+Var+σ²ε')

# Optimum
opt_idx = np.argmin(total)
ax.axvline(complexity[opt_idx], color=YELLOW, lw=1.5, linestyle=':', alpha=0.8)
ax.annotate(f'Optimum\n(min total error)', xy=(complexity[opt_idx], total[opt_idx]),
            xytext=(complexity[opt_idx]+1.2, total[opt_idx]+0.8),
            color=YELLOW, fontsize=9,
            arrowprops=dict(arrowstyle='->', color=YELLOW, lw=1.0))

ax.text(1.5, 2.8, 'Underfitting\n(High Bias)', ha='center', color=BLUE, fontsize=9,
        bbox=dict(boxstyle='round', facecolor=DARK_BG, alpha=0.6))
ax.text(8.5, 2.8, 'Overfitting\n(High Variance)', ha='center', color=ORANGE, fontsize=9,
        bbox=dict(boxstyle='round', facecolor=DARK_BG, alpha=0.6))

ax.set_xlabel('Model Complexity', fontsize=12)
ax.set_ylabel('Expected Error', fontsize=12)
ax.set_title('Bias-Variance Tradeoff', color='#e2e8f0', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, framealpha=0.3, loc='upper center')
ax.set_ylim(0, 6.5); ax.set_xlim(0.1, 10)
ax.grid(True)
plt.tight_layout()
save('bias_variance.png')

# ─────────────────────────────────────────────────────────────────────────────
# 7. DROPOUT — EXPECTED ACTIVATION
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor(DARK_BG)

np.random.seed(42)
n_neurons = 10
p = 0.5
activations = np.random.uniform(0.3, 1.2, n_neurons)

def plot_neurons(ax, title, vals, colors, scale_note=''):
    ax.set_facecolor(PANEL)
    for i, (v, c) in enumerate(zip(vals, colors)):
        bar = ax.bar(i, v, color=c, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.65)
    ax.set_ylim(0, 1.8)
    ax.set_xlim(-0.5, n_neurons-0.5)
    ax.set_xlabel('Neuron index'); ax.set_ylabel('Activation value')
    ax.set_title(title, color='#e2e8f0', fontsize=10.5, fontweight='bold')
    ax.axhline(np.mean(vals[vals>0] if hasattr(vals,'__len__') else [vals]),
               color=YELLOW, lw=1.5, linestyle='--', alpha=0.7)
    if scale_note:
        ax.text(0.5, 1.05, scale_note, transform=ax.transAxes, ha='center', color='#94a3b8', fontsize=8)
    ax.grid(True, axis='y', alpha=0.4)

# No dropout scaling: mask some neurons, no scale
mask = np.random.binomial(1, 1-p, n_neurons).astype(float)
dropped = activations * mask
cols_drop = [GREEN if m>0 else RED for m in mask]
plot_neurons(axes[0],
             f'Training (p=0.5, NO inverted scaling)\nE[ã] = (1−p)·a ≠ a  ← mismatch!',
             dropped, cols_drop, '→ Inference uses full activations → scale mismatch')

# Inverted dropout: scale by 1/(1-p)
inv = dropped / (1-p)
cols_inv = [TEAL if m>0 else '#475569' for m in mask]
plot_neurons(axes[1],
             f'Training (Inverted Dropout ÷(1−p)=×2)\nE[ã] = (1−p)·(a/(1−p)) = a  ✓',
             inv, cols_inv, '→ Inference: all neurons on, NO scaling needed')

for ax in axes:
    ax.legend(handles=[
        mpatches.Patch(color=GREEN, label='Active neuron'),
        mpatches.Patch(color=RED,   label='Dropped (×0)'),
    ], fontsize=8, framealpha=0.3, loc='upper right')

fig.suptitle('Dropout: Standard vs Inverted Scaling', color='#e2e8f0', fontsize=13, fontweight='bold')
plt.tight_layout()
save('dropout.png')

# ─────────────────────────────────────────────────────────────────────────────
# 8. NN ARCHITECTURE — simple MLP diagram using matplotlib patches
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_facecolor(DARK_BG)
fig.patch.set_facecolor(DARK_BG)
ax.set_xlim(0, 10); ax.set_ylim(-0.5, 6)
ax.axis('off')

layers = [
    (1.0,  [1,2,3,4],   BLUE,   'Input layer\n(x₁, x₂, x₃, x₄)'),
    (3.5,  [0.5,1.5,2.5,3.5,4.5], GREEN,  'Hidden layer 1\n(n₁=5, ReLU)'),
    (6.0,  [1,2,3,4],   TEAL,   'Hidden layer 2\n(n₂=4, ReLU)'),
    (8.5,  [1.5, 3.0],   YELLOW,'Output layer\n(n₃=2, Softmax)'),
]

node_r = 0.22
all_coords = []
for x, ys, col, label in layers:
    coords = [(x, y) for y in ys]
    all_coords.append(coords)
    for (cx, cy) in coords:
        circle = plt.Circle((cx, cy), node_r, color=col, zorder=5, alpha=0.9)
        ax.add_patch(circle)
    ax.text(x, -0.25, label, ha='center', va='top', color='#e2e8f0', fontsize=8.5, zorder=6)

# Draw connections
for i in range(len(all_coords)-1):
    src = all_coords[i]; dst = all_coords[i+1]
    for (sx, sy) in src:
        for (dx, dy) in dst:
            ax.plot([sx+node_r, dx-node_r], [sy, dy], color='#475569', lw=0.5, alpha=0.4, zorder=3)

# Forward pass annotation
ax.annotate('', xy=(8.0, 4.8), xytext=(1.5, 4.8),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=2))
ax.text(4.75, 5.05, 'Forward Pass: z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾,   a⁽ˡ⁾ = f(z⁽ˡ⁾)',
        ha='center', color=GREEN, fontsize=9)

ax.annotate('', xy=(1.5, 5.6), xytext=(8.0, 5.6),
            arrowprops=dict(arrowstyle='->', color=RED, lw=2))
ax.text(4.75, 5.75, 'Backprop: δ⁽ˡ⁾ = [(W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾] ⊙ f\'(z⁽ˡ⁾)',
        ha='center', color=RED, fontsize=9)

# Loss label
ax.text(9.5, 2.5, 'Loss\nL(ŷ,y)', ha='center', color=ORANGE, fontsize=9.5, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='#1e293b', edgecolor=ORANGE, alpha=0.8))

ax.set_title('Multi-Layer Perceptron (MLP) — Forward & Backpropagation',
             color='#e2e8f0', fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
save('mlp_architecture.png')

# ─────────────────────────────────────────────────────────────────────────────
# 9. LEARNING CURVES — overfitting vs good fit
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
fig.patch.set_facecolor(DARK_BG)
epochs = np.arange(1, 101)

# Overfitting
ax = axes[0]; ax.set_facecolor(PANEL)
train_loss  = 2.5*np.exp(-0.05*epochs)+0.05
val_loss_of = 1.5*np.exp(-0.03*epochs)+0.6*(1-np.exp(-0.02*(epochs-30)))
val_loss_of = np.where(epochs<30, 1.5*np.exp(-0.03*epochs), val_loss_of)
ax.plot(epochs, train_loss,  color=GREEN, lw=2.2, label='Train Loss')
ax.plot(epochs, val_loss_of, color=RED,   lw=2.2, label='Val Loss')
ax.axvline(30, color=YELLOW, lw=1.2, linestyle=':', alpha=0.8)
ax.text(32, 1.7, 'Overfitting\nbegins', color=YELLOW, fontsize=8.5)
ax.fill_between(epochs[29:], val_loss_of[29:], train_loss[29:], alpha=0.15, color=RED)
ax.set_title('Overfitting Pattern', color='#e2e8f0', fontsize=11, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(fontsize=9, framealpha=0.3); ax.grid(True)

# Good fit
ax = axes[1]; ax.set_facecolor(PANEL)
train_loss2 = 2.5*np.exp(-0.05*epochs)+0.2
val_loss2   = 2.8*np.exp(-0.04*epochs)+0.25
ax.plot(epochs, train_loss2, color=GREEN, lw=2.2, label='Train Loss')
ax.plot(epochs, val_loss2,   color=BLUE,  lw=2.2, label='Val Loss')
ax.fill_between(epochs, val_loss2, train_loss2, alpha=0.08, color=BLUE)
ax.set_title('Good Fit (Generalization)', color='#e2e8f0', fontsize=11, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(fontsize=9, framealpha=0.3); ax.grid(True)

fig.suptitle('Learning Curves: Diagnosing Model Fit', color='#e2e8f0', fontsize=13, fontweight='bold')
plt.tight_layout()
save('learning_curves.png')

print('\nAll figures generated successfully!')
