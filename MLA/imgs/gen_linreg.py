import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'figure.dpi':160,'savefig.dpi':160,'savefig.bbox':'tight',
    'figure.facecolor':'#0f172a','axes.facecolor':'#1e293b',
    'axes.edgecolor':'#475569','axes.labelcolor':'#e2e8f0',
    'xtick.color':'#94a3b8','ytick.color':'#94a3b8',
    'text.color':'#e2e8f0','grid.color':'#334155','grid.alpha':0.5,
    'font.family':'DejaVu Sans','font.size':11,
})
D='#0f172a';P='#1e293b';G='#4ade80';B='#60a5fa';Y='#fbbf24';R='#f87171';T='#2dd4bf';O='#fb923c'

def save(n): plt.savefig(f'/Users/lap13954/Documents/q/MLA/imgs/{n}',facecolor=D); plt.close(); print(f'  saved {n}')

np.random.seed(42)

# ── 1. Scatter + regression line + residuals ──────────────────────────────
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5)); fig.patch.set_facecolor(D)
x=np.linspace(1,10,40); y=2.3*x+5+np.random.randn(40)*3
w,b=np.polyfit(x,1,y) if False else (np.polyfit(x,y,1))
w,b=w,b; yhat=w*x+b; res=y-yhat

ax1.set_facecolor(P)
ax1.scatter(x,y,color=B,s=50,alpha=0.85,edgecolors='white',lw=0.4,zorder=5)
ax1.plot(x,yhat,color=G,lw=2.5,label=f'ŷ = {w:.2f}x + {b:.2f}')
for xi,yi,yhi in zip(x,y,yhat):
    ax1.plot([xi,xi],[yi,yhi],color=R,lw=0.8,alpha=0.5)
ax1.set_xlabel('x (feature)'); ax1.set_ylabel('y (target)')
ax1.set_title('Hồi quy tuyến tính — Đường fit & Residuals',color='#e2e8f0',fontsize=11,fontweight='bold')
ax1.legend(fontsize=10,framealpha=0.3)
ax1.text(0.05,0.92,f'MSE = {np.mean(res**2):.2f}\nR² = {1-np.var(res)/np.var(y):.3f}',
         transform=ax1.transAxes,color='#e2e8f0',fontsize=9,
         bbox=dict(boxstyle='round',facecolor=D,alpha=0.7))
ax1.grid(True)

ax2.set_facecolor(P)
ax2.scatter(yhat,res,color=O,s=50,alpha=0.85,edgecolors='white',lw=0.4)
ax2.axhline(0,color=G,lw=1.8,linestyle='--')
ax2.set_xlabel('Fitted values ŷ'); ax2.set_ylabel('Residuals e = y − ŷ')
ax2.set_title('Residual Plot\n(Homoscedasticity check)',color='#e2e8f0',fontsize=11,fontweight='bold')
ax2.text(0.05,0.92,'Tốt: phân tán đều quanh 0\nXấu: hình phễu → heteroscedastic',
         transform=ax2.transAxes,color='#94a3b8',fontsize=8.5,
         bbox=dict(boxstyle='round',facecolor=D,alpha=0.7))
ax2.grid(True)
fig.suptitle('Linear Regression: Fit & Residual Diagnostics',color='#e2e8f0',fontsize=13,fontweight='bold')
plt.tight_layout(); save('linreg_fit.png')

# ── 2. Polynomial degree: underfitting / good / overfitting ──────────────
np.random.seed(7)
x_raw=np.linspace(0,1,25); y_raw=np.sin(2*np.pi*x_raw)+np.random.randn(25)*0.3
xp=np.linspace(0,1,300)
fig,axes=plt.subplots(1,3,figsize=(13,4.5)); fig.patch.set_facecolor(D)
for ax,(deg,lbl,col,note) in zip(axes,[
    (1,'degree=1\nUnderfitting (High Bias)',B,'Train MSE: high\nTest MSE: high'),
    (3,'degree=3\nGood Fit',G,'Train MSE: low\nTest MSE: low'),
    (15,'degree=15\nOverfitting (High Variance)',R,'Train MSE: ≈0\nTest MSE: very high'),
]):
    ax.set_facecolor(P)
    c=np.polyfit(x_raw,y_raw,deg)
    yfit=np.polyval(c,xp)
    ax.scatter(x_raw,y_raw,color=Y,s=40,edgecolors='white',lw=0.4,zorder=5,label='Data')
    ax.plot(xp,np.sin(2*np.pi*xp),color='#64748b',lw=1.5,linestyle=':',label='True fn')
    ax.plot(xp,np.clip(yfit,-2.5,2.5),color=col,lw=2.2,label=f'deg={deg}')
    ax.set_ylim(-2.5,2.5); ax.set_title(lbl,color='#e2e8f0',fontsize=10.5,fontweight='bold')
    ax.legend(fontsize=8,framealpha=0.3,loc='upper right')
    ax.text(0.5,-0.22,note,transform=ax.transAxes,ha='center',color='#94a3b8',fontsize=8.5,style='italic')
    ax.grid(True)
fig.suptitle('Polynomial Regression: Underfitting → Good Fit → Overfitting',color='#e2e8f0',fontsize=13,fontweight='bold')
plt.tight_layout(rect=[0,0.06,1,1]); save('linreg_poly.png')

# ── 3. Ridge vs Lasso — coefficient paths ────────────────────────────────
np.random.seed(3)
n,d=80,6; X=np.random.randn(n,d); X[:,2]=X[:,0]+np.random.randn(n)*0.1  # collinear
true_w=np.array([3,-2,1.5,0,0,0.8]); y=X@true_w+np.random.randn(n)*0.5

def ridge_coef(X,y,lam):
    return np.linalg.solve(X.T@X+lam*np.eye(X.shape[1]),X.T@y)

def lasso_cd(X,y,lam,iters=500):
    w=np.zeros(X.shape[1]); r=y.copy()
    for _ in range(iters):
        for j in range(X.shape[1]):
            r+=X[:,j]*w[j]; rho=X[:,j]@r
            n_=X[:,j]@X[:,j]
            w[j]=np.sign(rho)*max(abs(rho)-lam,0)/n_; r-=X[:,j]*w[j]
    return w

lambdas=np.logspace(-2,3,60)
ridge_paths=np.array([ridge_coef(X,y,l) for l in lambdas])
lasso_paths=np.array([lasso_cd(X,y,l*0.5) for l in lambdas])

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5)); fig.patch.set_facecolor(D)
colors_w=[G,B,O,Y,T,R]
lbls=[f'w{i+1}'+(' (zero-true)'if true_w[i]==0 else '') for i in range(d)]

for ax,paths,title in [(ax1,ridge_paths,'Ridge (L2) — Shrink toward 0, never exact 0'),
                        (ax2,lasso_paths,'Lasso (L1) — Sparse: drives to exact 0')]:
    ax.set_facecolor(P)
    for i,(c,lb) in enumerate(zip(colors_w,lbls)):
        ax.semilogx(lambdas,paths[:,i],color=c,lw=2,label=lb)
    ax.axhline(0,color='#475569',lw=0.8)
    ax.set_xlabel('λ (regularization strength)'); ax.set_ylabel('Coefficient value')
    ax.set_title(title,color='#e2e8f0',fontsize=10.5,fontweight='bold')
    ax.legend(fontsize=8,framealpha=0.3,loc='upper right',ncol=2)
    ax.grid(True,which='both')
    ax.text(0.02,0.08,'← Low λ\n(overfit)',transform=ax.transAxes,color='#94a3b8',fontsize=8)
    ax.text(0.80,0.08,'High λ →\n(underfit)',transform=ax.transAxes,color='#94a3b8',fontsize=8)

fig.suptitle('Regularization Coefficient Paths: Ridge vs Lasso',color='#e2e8f0',fontsize=13,fontweight='bold')
plt.tight_layout(); save('linreg_regularization.png')

# ── 4. Normal Equation geometry — projection ─────────────────────────────
fig,ax=plt.subplots(figsize=(8,6)); ax.set_facecolor(P); fig.patch.set_facecolor(D)
ax.set_xlim(-0.3,3.5); ax.set_ylim(-0.3,3.5); ax.set_aspect('equal'); ax.axis('off')

# Column space plane (simplified as line in 2D)
ax.fill([0,3,3,0],[0,0,1.5,0],color='#1e3a5f',alpha=0.5,zorder=1)
ax.text(2.5,0.3,'Col(X)\n(column space)',color=B,fontsize=9)

# y vector
ax.annotate('',xy=(1.5,2.8),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=Y,lw=2.5))
ax.text(1.55,2.85,'y (target vector)',color=Y,fontsize=9)

# ŷ = Xw* projection
ax.annotate('',xy=(2.0,1.0),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=G,lw=2.5))
ax.text(2.05,1.05,'ŷ = Xw*\n(projection)',color=G,fontsize=9)

# residual e = y - ŷ
ax.annotate('',xy=(1.5,2.8),xytext=(2.0,1.0),
            arrowprops=dict(arrowstyle='->',color=R,lw=2,linestyle='dashed'))
ax.text(1.85,1.95,'e = y−ŷ\n⊥ Col(X)',color=R,fontsize=9)

# right angle mark
ax.plot([1.93,1.85,1.77],[1.07,1.15,1.07],color=R,lw=1.2)

ax.set_title('Normal Equation: Projection onto Column Space\nw* = (XᵀX)⁻¹Xᵀy  minimizes ‖y−Xw‖²',
             color='#e2e8f0',fontsize=11,fontweight='bold',pad=15)
ax.text(0.5,-0.05,'Residual e ⊥ Col(X)  ⟺  Xᵀe = 0  ⟺  Xᵀ(y−Xw)=0  ⟺  w* = (XᵀX)⁻¹Xᵀy',
        transform=ax.transAxes,ha='center',color='#94a3b8',fontsize=9)
plt.tight_layout(); save('linreg_projection.png')

print('\nAll linreg figures done!')
