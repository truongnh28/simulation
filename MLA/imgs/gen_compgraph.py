"""
Computational graph figures for nn.html
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams.update({
    'figure.dpi':160,'savefig.dpi':160,'savefig.bbox':'tight',
    'figure.facecolor':'#0f172a','axes.facecolor':'#0f172a',
    'text.color':'#e2e8f0','font.family':'DejaVu Sans','font.size':10,
})
D='#0f172a'; PANEL='#1e293b'; G='#4ade80'; B='#60a5fa'
Y='#fbbf24'; R='#f87171'; T='#2dd4bf'; O='#fb923c'; PU='#c084fc'
EDGE='#475569'; WHITE='#e2e8f0'

def save(n):
    plt.savefig(f'/Users/lap13954/Documents/q/MLA/imgs/{n}', facecolor=D)
    plt.close()
    print(f'  saved {n}')

def node(ax, x, y, label, val=None, grad=None,
         color=PANEL, text_color=WHITE, r=0.38, shape='circle'):
    """Draw a computation node."""
    if shape == 'rect':
        box = FancyBboxPatch((x-0.55, y-0.28), 1.1, 0.56,
                              boxstyle='round,pad=0.05', facecolor=color,
                              edgecolor=WHITE, linewidth=1.2, zorder=5)
        ax.add_patch(box)
    else:
        circle = plt.Circle((x,y), r, color=color, zorder=5, linewidth=1.2,
                             edgecolor=WHITE)
        ax.add_patch(circle)
    ax.text(x, y + (0.06 if val is None else 0.09), label,
            ha='center', va='center', fontsize=9, fontweight='bold',
            color=text_color, zorder=6)
    if val is not None:
        ax.text(x, y-0.10, str(val), ha='center', va='center',
                fontsize=8, color=G, zorder=6)
    if grad is not None:
        ax.text(x, y-0.25, f'∂={grad}', ha='center', va='center',
                fontsize=7.5, color=R, zorder=6,
                bbox=dict(boxstyle='round,pad=0.1',facecolor='#3b0a0a',alpha=0.8))

def arrow(ax, x1,y1, x2,y2, color=EDGE, lw=1.5, label=None, label_color=WHITE):
    ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0.0'))
    if label:
        mx,my = (x1+x2)/2,(y1+y2)/2
        ax.text(mx+0.05, my+0.05, label, fontsize=7.5, color=label_color,
                ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.1',facecolor=D,alpha=0.7))

# ─────────────────────────────────────────────────────────────────────────────
# 1. SIMPLE EXAMPLE: f = (x+y)*z  with gate-level forward/backward
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor(D)

for ax_idx, (ax, mode) in enumerate(zip(axes, ['forward', 'backward'])):
    ax.set_facecolor(D); ax.set_xlim(-0.5, 6.5); ax.set_ylim(-0.5, 3.5); ax.axis('off')
    # x=3, y=2, z=-4  =>  q=x+y=5, f=q*z=-20
    x_v,y_v,z_v = 3,2,-4
    q_v = x_v+y_v   # 5
    f_v = q_v*z_v   # -20

    # ∂f/∂f=1, ∂f/∂q=z=-4, ∂f/∂z=q=5, ∂f/∂x=∂f/∂q·∂q/∂x=-4·1=-4, ∂f/∂y=-4
    df_dq=z_v; df_dz=q_v; df_dx=z_v*1; df_dy=z_v*1

    # Node positions
    NX={'x':(0.5,2.5),'y':(0.5,1.0),'z':(0.5,0.0),
        'add':(2.5,1.75),'mul':(4.5,0.9),'f':(6.0,0.9)}

    # Draw nodes
    vals = {'x':x_v,'y':y_v,'z':z_v,'add':q_v,'mul':f_v,'f':f_v}
    grads = {'x':df_dx,'y':df_dy,'z':df_dz,'add':df_dq,'mul':1,'f':1}
    labels = {'x':'x','y':'y','z':'z','add':'+','mul':'×','f':'f'}
    colors = {'x':B,'y':B,'z':O,'add':'#1e3a5f','mul':'#3b1f5f','f':G}

    for name,(nx,ny) in NX.items():
        v = vals[name] if mode=='forward' else None
        g = grads[name] if mode=='backward' else None
        node(ax, nx, ny, labels[name], val=v, grad=g,
             color=colors[name], r=0.32)

    # Edges (forward)
    arrow(ax, 0.82,2.5, 2.18,1.95, color=G if mode=='forward' else EDGE)
    arrow(ax, 0.82,1.0, 2.18,1.55, color=G if mode=='forward' else EDGE)
    arrow(ax, 2.82,1.75, 4.18,1.1, color=G if mode=='forward' else EDGE)
    arrow(ax, 0.82,0.0, 4.18,0.7,  color=G if mode=='forward' else EDGE)
    arrow(ax, 4.82,0.9, 5.68,0.9,  color=G if mode=='forward' else EDGE)

    if mode == 'backward':
        # Backward gradient arrows (red, reversed)
        arrow(ax, 5.68,1.2, 4.82,1.2, color=R, lw=2,
              label=f'∂f/∂f=1', label_color=R)
        arrow(ax, 4.18,1.2, 3.32,2.0, color=R, lw=2,
              label=f'∂={df_dq}', label_color=R)
        arrow(ax, 4.18,0.7, 0.82,0.2, color=R, lw=2,
              label=f'∂={df_dz}', label_color=R)
        arrow(ax, 2.18,2.1, 0.82,2.7, color=R, lw=2,
              label=f'∂={df_dx}', label_color=R)
        arrow(ax, 2.18,1.6, 0.82,1.1, color=R, lw=2,
              label=f'∂={df_dy}', label_color=R)

    title = ('Forward Pass: tính giá trị từ inputs\n(màu xanh = giá trị tại node)'
             if mode=='forward'
             else 'Backward Pass: chain rule ngược\n(màu đỏ = gradient ∂f/∂node)')
    ax.set_title(title, color=WHITE, fontsize=10.5, fontweight='bold', pad=8)

    # Legend
    if mode=='forward':
        ax.text(0.5,-0.4,f'x={x_v}, y={y_v}, z={z_v}\nq=x+y={q_v}, f=q×z={f_v}',
                fontsize=8.5, color=G,
                bbox=dict(boxstyle='round',facecolor=PANEL,alpha=0.8))
    else:
        ax.text(0.5,-0.45,
                f'∂f/∂z=q={df_dz}  (mul gate: grad=other input)\n'
                f'∂f/∂x=∂f/∂q·1={df_dx}  (add gate: grad passes through)',
                fontsize=8, color=R,
                bbox=dict(boxstyle='round',facecolor='#2d0a0a',alpha=0.8))

fig.suptitle('Computational Graph: f = (x+y)×z  với  x=3, y=2, z=−4',
             color=WHITE, fontsize=13, fontweight='bold')
plt.tight_layout()
save('compgraph_simple.png')

# ─────────────────────────────────────────────────────────────────────────────
# 2. GATE GRADIENT RULES — visual table
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.patch.set_facecolor(D)

gate_data = [
    ('ADD Gate  f=x+y', 'Gradient distributor',
     '∂f/∂x = 1\n∂f/∂y = 1',
     'Gradient passes through UNCHANGED\nto both inputs',
     B, [('x',1.0,2.5,'∂=1.0'), ('y',1.0,1.0,'∂=1.0'),
         ('+',3.0,1.75,'∂=?'), ('f',5.0,1.75,'∂=1.0')],
     [(0.35,2.5,2.65,2.0),(0.35,1.0,2.65,1.5),(3.35,1.75,4.65,1.75)]),
    ('MUL Gate  f=xy', 'Gradient swapper',
     '∂f/∂x = y\n∂f/∂y = x',
     'Each input gets gradient scaled by\nthe OTHER input\'s value',
     O, [('x=3',1.0,2.5,'∂=y=2'), ('y=2',1.0,1.0,'∂=x=3'),
         ('×',3.0,1.75,'∂=?'), ('f',5.0,1.75,'∂=1.0')],
     [(0.35,2.5,2.65,2.0),(0.35,1.0,2.65,1.5),(3.35,1.75,4.65,1.75)]),
    ('MAX Gate  f=max(x,y)', 'Gradient router',
     '∂f/∂x = 1 if x>y else 0\n∂f/∂y = 1 if y>x else 0',
     'Gradient flows ONLY to\nthe winning (max) input',
     PU, [('x=5',1.0,2.5,'∂=1.0'), ('y=2',1.0,1.0,'∂=0.0'),
          ('max',3.0,1.75,'∂=?'), ('f',5.0,1.75,'∂=1.0')],
     [(0.35,2.5,2.65,2.0),(0.35,1.0,2.65,1.5),(3.35,1.75,4.65,1.75)]),
    ('SIGMOID Gate  f=σ(x)', 'Smooth squasher',
     "∂f/∂x = σ(x)·(1−σ(x))\n       = f·(1−f)  ≤ 0.25",
     'Max gradient = 0.25 (at x=0)\n→ Source of vanishing gradient!',
     R, [('x=0',1.0,1.75,'∂=0.25'), ('σ',3.0,1.75,'∂=?'), ('f=0.5',5.0,1.75,'∂=1.0')],
     [(0.35,1.75,2.65,1.75),(3.35,1.75,4.65,1.75)]),
]

for ax, (title, subtitle, formula, note, col, nodes_d, edges_d) in zip(axes.flat, gate_data):
    ax.set_facecolor(D); ax.set_xlim(0,6); ax.set_ylim(0.2,3.5); ax.axis('off')

    # Draw nodes
    for (lbl,nx,ny,grad) in nodes_d:
        nc = col if lbl not in ('f','f=0.5') else G
        node(ax, nx, ny, lbl, grad=grad, color=nc, r=0.3)

    # Draw edges forward
    for (x1,y1,x2,y2) in edges_d:
        arrow(ax,x1,y1,x2,y2, color=G, lw=1.5)
    # backward arrows (red) - reverse edges
    for (x1,y1,x2,y2) in reversed(edges_d):
        arrow(ax,x2,y2,x1,y1, color=R, lw=1.8)

    ax.set_title(f'{title}\n{subtitle}', color=col, fontsize=10, fontweight='bold')
    ax.text(3.0, 0.55, formula, ha='center', va='center', fontsize=9,
            color=Y, fontfamily='monospace',
            bbox=dict(boxstyle='round',facecolor=PANEL,alpha=0.85, edgecolor=col))
    ax.text(3.0, 0.25, note, ha='center', va='center', fontsize=8,
            color='#94a3b8', style='italic')

fig.suptitle('Gradient Propagation Rules for Basic Gates\n'
             '(xanh = forward, đỏ = backward gradient)',
             color=WHITE, fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0,0,1,0.93])
save('compgraph_gates.png')

# ─────────────────────────────────────────────────────────────────────────────
# 3. SINGLE NEURON: L = BCE(σ(w₁x₁+w₂x₂+b), y)  — full computation graph
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor(D)

# Values: x1=2,x2=3, w1=0.5,w2=-1, b=1, y=1
x1,x2,w1,w2,b,y_t = 2.0, 3.0, 0.5, -1.0, 1.0, 1.0
t1=w1*x1   # 1.0
t2=w2*x2   # -3.0
z_=t1+t2+b # -1.0
a_=1/(1+np.exp(-z_))  # σ(-1) ≈ 0.269
L_=-y_t*np.log(a_)-(1-y_t)*np.log(1-a_)  # BCE

# Gradients (backward)
dL_da  = -y_t/a_ + (1-y_t)/(1-a_)        # ≈ -3.72
da_dz  = a_*(1-a_)                         # ≈ 0.197
dL_dz  = dL_da * da_dz                     # ≈ a_-y = -0.731 (shortcut)
dL_dz_shortcut = a_-y_t                    # ≈ -0.731 (CE shortcut)
dL_dt1 = dL_dz*1
dL_dt2 = dL_dz*1
dL_db  = dL_dz*1
dL_dw1 = dL_dz*x1
dL_dw2 = dL_dz*x2

node_pos = {
    'x1': (0.5, 5.5), 'w1': (0.5, 4.2),
    'x2': (0.5, 2.8), 'w2': (0.5, 1.5),
    'b':  (0.5, 0.2),
    't1': (2.2, 4.85), 't2': (2.2, 2.15),
    '+':  (3.8, 2.8),
    'σ':  (5.4, 2.8),
    'BCE':(7.0, 2.8),
    'L':  (8.5, 2.8),
    'y':  (7.0, 1.2),
}
node_vals = {
    'x1':f'{x1}','w1':f'{w1}','x2':f'{x2}','w2':f'{w2}','b':f'{b}',
    't1':f'{t1:.1f}','t2':f'{t2:.1f}','+':f'{z_:.1f}',
    'σ':f'{a_:.3f}','BCE':f'{L_:.3f}','L':f'{L_:.3f}','y':f'{y_t}'
}
node_grads = {
    'x1':f'—','w1':f'{dL_dw1:.3f}','x2':f'—','w2':f'{dL_dw2:.3f}','b':f'{dL_db:.3f}',
    't1':f'{dL_dt1:.3f}','t2':f'{dL_dt2:.3f}','+':f'{dL_dz:.3f}',
    'σ':f'{dL_da:.3f}','BCE':f'1.0','L':f'1.0','y':f'—'
}
node_colors = {
    'x1':B,'w1':O,'x2':B,'w2':O,'b':PU,
    't1':'#1e3a5f','t2':'#1e3a5f','+':'#1e3a5f',
    'σ':'#3b1f5f','BCE':'#1c3a2a','L':G,'y':'#4a1942'
}
edges = [
    ('x1','t1'),('w1','t1'),('x2','t2'),('w2','t2'),
    ('t1','+'),('+','σ'),('+','σ'),('b','+'),
    ('t2','+'),('+','σ'),('σ','BCE'),('BCE','L'),('y','BCE')
]

for ax_idx,(ax,mode) in enumerate(zip(axes,['forward','backward'])):
    ax.set_facecolor(D); ax.set_xlim(-0.2,9.5); ax.set_ylim(-0.3,6.5); ax.axis('off')

    # edges first
    edge_pairs = [
        ('x1','t1'),('w1','t1'),('x2','t2'),('w2','t2'),
        ('t1','+'),('+','σ'),('b','+'),('t2','+'),
        ('σ','BCE'),('BCE','L'),('y','BCE')
    ]
    for (a,b_) in edge_pairs:
        p1=node_pos[a]; p2=node_pos[b_]
        col = G if mode=='forward' else EDGE
        arrow(ax, p1[0]+0.35, p1[1], p2[0]-0.35, p2[1], color=col, lw=1.2)

    if mode=='backward':
        bw_edges = [
            ('L','BCE'),('BCE','σ'),('σ','+'),
            ('+','t1'),('+','t2'),('+',(0.5,0.2)),
            ('t1','w1'),('t2','w2')
        ]
        for edge in bw_edges:
            if isinstance(edge[1],str):
                p1=node_pos[edge[1]]; p2=node_pos[edge[0]]
            else:
                p1=edge[1]; p2=node_pos[edge[0]]
            arrow(ax,p1[0]+0.35,p1[1],p2[0]+0.35,p2[1],color=R,lw=1.8)

    # draw nodes
    for name,(nx,ny) in node_pos.items():
        v = node_vals.get(name) if mode=='forward' else None
        g = node_grads.get(name) if mode=='backward' else None
        if g == '—': g = None
        node(ax, nx, ny, name,
             val=v, grad=g,
             color=node_colors[name], r=0.32)

    title = 'Forward Pass' if mode=='forward' else 'Backward Pass (Chain Rule)'
    ax.set_title(title, color=WHITE, fontsize=11, fontweight='bold')

    if mode=='backward':
        shortcut_text = f'CE+Sigmoid shortcut:\n∂L/∂z = σ(z)−y = {a_:.3f}−{y_t} = {a_-y_t:.3f}'
        ax.text(3.8, 0.0, shortcut_text, ha='center', fontsize=8.5,
                color=Y, bbox=dict(boxstyle='round',facecolor='#3b2a00',alpha=0.85))

fig.suptitle('Computational Graph: Single Neuron\n'
             'L = BCE(σ(w₁x₁+w₂x₂+b), y)  [x₁=2, x₂=3, w₁=0.5, w₂=−1, b=1, y=1]',
             color=WHITE, fontsize=12, fontweight='bold')
plt.tight_layout()
save('compgraph_neuron.png')

# ─────────────────────────────────────────────────────────────────────────────
# 4. 2-LAYER MLP — step-by-step with actual numbers
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 9))
fig.patch.set_facecolor(D)
ax = fig.add_subplot(111)
ax.set_facecolor(D); ax.set_xlim(-0.5,15); ax.set_ylim(-1.5,8.5); ax.axis('off')

# Network: x=(1,) → W1(2×1)+b1 → ReLU → W2(1×2)+b2 → sigmoid → CE
# x=1.0, W1=[[0.5],[-1.0]], b1=[0.3,-0.2]
# Layer 1: z1=W1x+b1=[0.5+0.3, -1+(-0.2)]=[0.8,-1.2]
# a1=ReLU(z1)=[0.8,0] (second neuron dies)
# W2=[0.6,-0.4], b2=0.1
# z2 = W2·a1+b2 = 0.6*0.8+(-0.4)*0+0.1 = 0.58
# a2=σ(0.58)≈0.641
# y=1 → L=-log(0.641)≈0.444

x_in=1.0
z1_1=0.5*x_in+0.3;  z1_2=-1.0*x_in-0.2   # 0.8, -1.2
a1_1=max(0,z1_1);    a1_2=max(0,z1_2)       # 0.8, 0.0
z2_=0.6*a1_1+(-0.4)*a1_2+0.1               # 0.58
a2_=1/(1+np.exp(-z2_))                      # ≈0.641
y_true=1.0
L_val=-np.log(a2_)                           # ≈0.444

# Backward
dL_da2 = -1/a2_                             # ≈-1.560
da2_dz2 = a2_*(1-a2_)                       # ≈0.230
dL_dz2 = a2_-y_true                         # ≈-0.359 (CE shortcut)
dL_da1_1 = dL_dz2*0.6                       # ≈-0.215
dL_da1_2 = dL_dz2*(-0.4)                    # ≈0.144
# ReLU: gradient=1 if z>0 else 0
dL_dz1_1 = dL_da1_1*1.0   # z1_1=0.8>0     # ≈-0.215
dL_dz1_2 = dL_da1_2*0.0   # z1_2=-1.2<0 DEAD # 0
dL_dw11 = dL_dz1_1*x_in                     # ≈-0.215
dL_dw12 = dL_dz1_2*x_in                     # 0

# Layout: columns at x=0,2,4.5,6.5,9,11,13,14.5
# Rows for visual separation

# Title areas
ax.text(7.5,8.2,'2-Layer MLP — Tính forward và backward với số cụ thể',
        ha='center',fontsize=12,fontweight='bold',color=WHITE)
ax.text(7.5,7.8,f'x={x_in} | W1=[[0.5],[-1.0]] | b1=[0.3,−0.2] | W2=[0.6,−0.4] | b2=0.1 | y={y_true}',
        ha='center',fontsize=8.5,color='#94a3b8')

# ── FORWARD PASS row ──────────────────────────────────────────────────────
ax.text(-0.3,6.8,'FORWARD',fontsize=9,fontweight='bold',color=G,rotation=90,va='center')

boxes_fwd = [
    (1.0, 6.5, f'x\n={x_in}', B),
    (3.2, 7.2, f'z1₁=W11·x+b1₁\n=0.5×1+0.3={z1_1:.1f}', '#1e3a5f'),
    (3.2, 5.8, f'z1₂=W12·x+b1₂\n=−1×1−0.2={z1_2:.1f}', '#1e3a5f'),
    (6.0, 7.2, f'a1₁=ReLU({z1_1})\n={a1_1:.1f}  ✓ active', G),
    (6.0, 5.8, f'a1₂=ReLU({z1_2:.1f})\n={a1_2:.1f}  ✗ DEAD', R),
    (9.0, 6.5, f'z2=0.6×{a1_1}+(−0.4)×{a1_2}+0.1\n={z2_:.2f}', '#1e3a5f'),
    (11.8,6.5, f'a2=σ({z2_:.2f})\n={a2_:.3f}', PU),
    (13.8,6.5, f'L=−log({a2_:.3f})\n={L_val:.3f}', Y),
]

for (bx,by,txt,col) in boxes_fwd:
    box=FancyBboxPatch((bx-0.9,by-0.45),1.8,0.9,
                        boxstyle='round,pad=0.05',facecolor=col,
                        edgecolor=WHITE,linewidth=0.8,alpha=0.85,zorder=5)
    ax.add_patch(box)
    ax.text(bx,by,txt,ha='center',va='center',fontsize=7.5,color=WHITE,
            fontweight='bold',zorder=6,linespacing=1.4)

# arrows forward
fwd_arrows=[
    (1.9,6.5,2.3,7.2),(1.9,6.5,2.3,5.8),
    (4.1,7.2,5.1,7.2),(4.1,5.8,5.1,5.8),
    (6.9,7.2,7.7,6.7),(6.9,5.8,7.7,6.3),
    (10.1,6.5,10.9,6.5),(12.7,6.5,12.9,6.5),
]
for (x1f,y1f,x2f,y2f) in fwd_arrows:
    arrow(ax,x1f,y1f,x2f,y2f,color=G,lw=1.5)

# ── BACKWARD PASS row ──────────────────────────────────────────────────────
ax.text(-0.3,3.0,'BACKWARD',fontsize=9,fontweight='bold',color=R,rotation=90,va='center')

boxes_bwd = [
    (13.8,3.5, f'∂L/∂L=1', '#2d1f00'),
    (11.8,3.5, f'∂L/∂a2={dL_da2:.3f}\n(CE shortcut:\na2−y={dL_dz2:.3f})', '#2d1f00'),
    (9.0, 3.5, f'∂L/∂z2=a2−y\n={dL_dz2:.3f}\n(sigmoid+CE)', '#3b0a0a'),
    (6.0, 4.2, f'∂L/∂a1₁=∂z2·W21\n={dL_dz2:.3f}×0.6\n={dL_da1_1:.3f}', '#2d1f00'),
    (6.0, 2.8, f'∂L/∂a1₂=∂z2·W22\n={dL_dz2:.3f}×(−0.4)\n={dL_da1_2:.3f}', '#2d1f00'),
    (3.2, 4.2, f'∂L/∂z1₁=∂a1₁·1\n(ReLU,z>0)\n={dL_dz1_1:.3f}', '#3b0a0a'),
    (3.2, 2.8, f'∂L/∂z1₂=∂a1₂·0\n(ReLU,z<0 DEAD)\n={dL_dz1_2:.1f} ← 0!', R),
    (1.0, 4.2, f'∂L/∂W11={dL_dw11:.3f}\n=∂z1₁·x={dL_dz1_1:.3f}×1', O),
    (1.0, 2.8, f'∂L/∂W12={dL_dw12:.1f}\n=∂z1₂·x=0×1', '#475569'),
]

for (bx,by,txt,col) in boxes_bwd:
    box=FancyBboxPatch((bx-0.9,by-0.55),1.8,1.1,
                        boxstyle='round,pad=0.05',facecolor=col,
                        edgecolor=R,linewidth=0.8,alpha=0.85,zorder=5)
    ax.add_patch(box)
    ax.text(bx,by,txt,ha='center',va='center',fontsize=7.2,color=WHITE,
            fontweight='bold',zorder=6,linespacing=1.4)

bwd_arrows=[
    (12.9,3.5,12.7,3.5),(10.9,3.5,9.9,3.5),
    (8.1,3.7,6.9,4.1),(8.1,3.3,6.9,2.9),
    (5.1,4.2,4.1,4.2),(5.1,2.8,4.1,2.8),
    (2.3,4.2,1.9,4.2),(2.3,2.8,1.9,2.8),
]
for (x1f,y1f,x2f,y2f) in bwd_arrows:
    arrow(ax,x1f,y1f,x2f,y2f,color=R,lw=1.5)

# annotation: weight update
ax.text(7.5,1.2,
    f'Weight update: W11 ← W11 − η·∂L/∂W11 = 0.5 − 0.01×({dL_dw11:.3f}) = {0.5-0.01*dL_dw11:.4f}\n'
    f'W12 giữ nguyên (∂=0) — đây là Dying ReLU! neuron 2 không bao giờ được cập nhật.',
    ha='center',fontsize=9,color=Y,
    bbox=dict(boxstyle='round',facecolor='#3b2a00',alpha=0.85,edgecolor=Y))

# divider
ax.axhline(5.0,color=EDGE,lw=0.8,linestyle=':',alpha=0.5)
ax.text(14.5,5.05,'─── Layer 2 ───',fontsize=7,color=EDGE)
ax.text(14.5,6.85,'Forward →',fontsize=8,color=G)
ax.text(14.5,3.3,'← Backward',fontsize=8,color=R)

plt.tight_layout()
save('compgraph_2layer.png')

print('\nAll computational graph figures done!')
