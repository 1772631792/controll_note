"""用全局 Python 重建难点专题配图；数值例题均为自编，非真题数据。

运行：D:\\Python310\\python.exe scripts/generate_difficult_concepts.py
输出：assets/difficult/*.png。使用 Agg 后端，不打开交互窗口。
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch
from scipy.linalg import expm
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'difficult'
OUT.mkdir(parents=True, exist_ok=True)
FONT = Path('C:/Windows/Fonts/msyh.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = [font_manager.FontProperties(fname=FONT).get_name(), 'DejaVu Sans']
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
                     'axes.unicode_minus': False, 'figure.facecolor': '#ffffff',
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': '#ffffff', 'mathtext.fontset': 'dejavusans'})
BLUE, RED, GREEN, GOLD = '#2367a0', '#bd453f', '#228678', '#b77b20'


def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=160, bbox_inches='tight', pad_inches=.22)
    plt.close(fig)


def axes_style(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.grid(alpha=.18)


def arrow(ax, a, b, color=BLUE, rad=0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=16,
                                lw=2, color=color, connectionstyle=f'arc3,rad={rad}'))


def mason():
    fig, ax = plt.subplots(figsize=(10, 4.3), layout='constrained')
    points = [(0, 0), (2.6, 0), (5.2, 0)]
    for (x, y), name in zip(points, ['r', 'x', 'y']):
        ax.add_patch(Circle((x, y), .14, color=BLUE))
        ax.text(x, y-.35, f'${name}$', ha='center', fontsize=17)
    arrow(ax, (.18,0),(2.4,0)); arrow(ax,(2.8,0),(5,0))
    ax.text(1.3,.12,'1',ha='center'); ax.text(3.9,.12,'$g$',ha='center',fontsize=17)
    for x, gain in [(2.6,'a'),(5.2,'b')]:
        arrow(ax,(x+.10,.15),(x-.10,.15),RED,rad=3)
        ax.text(x,.85,f'${gain}$',ha='center',color=RED,fontsize=17)
    ax.text(2.6,-.95,'两个回路分别只经过 x、y，没有共同节点',ha='center')
    ax.text(2.6,-1.4,r'$\Delta=1-a-b+ab=(1-a)(1-b),\qquad y/r=g/\Delta$',
            ha='center',fontsize=15)
    ax.set(xlim=(-.5,5.8),ylim=(-1.65,1.25)); ax.axis('off')
    ax.set_title('梅森公式：为什么不接触回路的乘积必须保留')
    save(fig,'00-mason')


def routh():
    fig, axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,K in zip(axs,[2,6,8]):
        roots=np.roots([1,3,2,K])
        ax.axvspan(-4,0,color=GREEN,alpha=.07); ax.axvline(0,color='#777',lw=1)
        ax.axhline(0,color='#777',lw=1)
        ax.scatter(roots.real,roots.imag,marker='x',s=85,color=BLUE,lw=2)
        ax.set(xlim=(-4,.8),ylim=(-2,2),title=f'K={K}：'+{2:'稳定',6:'临界',8:'不稳定'}[K])
        axes_style(ax,'实部','虚部')
    save(fig,'01-routh')


def nyquist():
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    w=np.tan(np.linspace(-np.pi/2+.0001,np.pi/2-.0001,2400))
    for ax,K in zip(axs,[.5,2]):
        z=K/(-1+1j*w); ax.plot(z.real,z.imag,color=BLUE,lw=2)
        for idx in [650,1650]: arrow(ax,(z[idx].real,z[idx].imag),(z[idx+35].real,z[idx+35].imag))
        ax.scatter([-1],[0],marker='x',s=75,color=RED,zorder=5)
        ax.annotate('−1 临界点',(-1,0),xytext=(-1.5,-.5),arrowprops={'arrowstyle':'->','color':RED})
        ax.axhline(0,color='#aaa',lw=.8); ax.axvline(0,color='#aaa',lw=.8)
        ax.set(xlim=(-2.3,.4),ylim=(-1.3,1.3),aspect='equal',title=f'K={K}：'+('闭环不稳定' if K<1 else '闭环稳定'))
        axes_style(ax,'Re L(jω)','Im L(jω)')
    fig.suptitle('完整频率方向：ω 从 −∞ 到 +∞；开环有 1 个右半平面极点')
    save(fig,'02-nyquist')


def root_locus():
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    ax.plot([-2,0],[0,0],lw=2,color=BLUE); ax.plot([-1,-1],[-3,3],lw=2,color=BLUE)
    ax.scatter([-2,0],[0,0],marker='x',s=100,color=RED,lw=2,label='开环极点')
    ax.scatter([-1],[0],s=45,color=GOLD,label='分离点 K=1')
    for a,b in [((-1.8,0),(-1.4,0)),((-.2,0),(-.6,0)),((-1,.8),(-1,1.5)),((-1,-.8),(-1,-1.5))]: arrow(ax,a,b)
    ax.axvline(0,color='#888',lw=.8); ax.set(xlim=(-2.7,.7),ylim=(-3.3,3.3),title='L(s)=K/[s(s+2)]：箭头表示 K 增大')
    axes_style(ax,'实部','虚部'); ax.legend(loc='upper right')
    save(fig,'03-root-locus')


def errors():
    t=np.linspace(0,8,500); fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.plot(t,1/3+2/3*np.exp(-3*t),color=GOLD,label='0 型 L=2/(s+1)：单位阶跃误差')
    ax.plot(t,.5*(1-np.exp(-2*t)),color=BLUE,ls='--',label='I 型 L=2/s：单位斜坡误差')
    ax.plot(t,t/2-.25+.25*np.exp(-2*t),color=RED,ls='-.',label='I 型 L=2/s：t²/2 输入误差')
    axes_style(ax,'时间 t / s','误差 e(t)'); ax.set_title('型别比较的是长期跟踪能力'); ax.legend()
    save(fig,'04-error')


def lead():
    w=np.logspace(-2,2,700); C=(1+1j*w)/(1+.25j*w)
    fig,axs=plt.subplots(2,1,figsize=(9,6),sharex=True,layout='constrained')
    axs[0].semilogx(w,20*np.log10(abs(C)),color=BLUE)
    axs[1].semilogx(w,np.angle(C,deg=True),color=GREEN)
    for ax in axs:
        ax.axvline(2,color=RED,ls='--',label='最大相位频率 2 rad/s'); ax.grid(alpha=.2)
    axs[0].scatter([2],[20*np.log10(2)],color=RED); axs[1].scatter([2],[np.degrees(np.arcsin(.6))],color=RED)
    axs[0].set(title='超前网络：T=1 s，α=0.25',ylabel='幅值 / dB'); axs[0].legend()
    axs[1].set(xlabel='角频率 ω / (rad/s)',ylabel='相位 / °')
    save(fig,'05-lead')


def exponential():
    t=np.linspace(0,6,500); x1=t*np.exp(-t); x2=np.exp(-t)
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    axs[0].plot(t,x1,color=BLUE,label='x₁=t exp(−t)'); axs[0].plot(t,x2,color=RED,ls='--',label='x₂=exp(−t)')
    axs[0].legend(); axes_style(axs[0],'时间 t / s','状态')
    axs[1].plot(x1,x2,color=BLUE); arrow(axs[1],(x1[50],x2[50]),(x1[70],x2[70]))
    axs[1].scatter([0],[1],color=RED,label='初态 (0,1)'); axs[1].legend()
    axes_style(axs[1],'x₁','x₂'); fig.suptitle('稳定的 Jordan 耦合也会产生先升后降')
    save(fig,'06-exponential')


def controllability():
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    u,v=np.meshgrid(np.linspace(-1,1,13),np.linspace(-1,1,13))
    ax.scatter(u,u+v,s=10,alpha=.3,color=BLUE,label='x₂=ABu₀+Bu₁，|u₀|、|u₁|≤1')
    arrow(ax,(0,0),(0,1),RED); arrow(ax,(0,0),(1,1),GREEN)
    ax.text(.05,1.05,'B=(0,1)',color=RED); ax.text(1.03,1,'AB=(1,1)',color=GREEN)
    ax.set(xlim=(-1.5,1.8),ylim=(-2.4,2.4),title='两个输入时刻提供两个独立方向')
    axes_style(ax,'第一状态分量','第二状态分量'); ax.legend(loc='lower left',fontsize=10)
    save(fig,'07-controllability')


def hidden():
    t=np.linspace(0,3,400); fig,axs=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    axs[0].plot(t,np.exp(-t),color=BLUE,label='y=x₁=exp(−t)')
    axs[1].plot(t,np.exp(2*t),color=RED,label='隐藏状态 x₂=exp(2t)')
    for ax in axs: axes_style(ax,'时间 t / s','状态幅值'); ax.legend()
    fig.suptitle('同一完整模型，零输入、初态 (1,1)：输出稳定不等于内部稳定')
    save(fig,'08-hidden-mode')


def observer():
    A=np.array([[0.,1.],[-2.,-3.]]); B=np.array([0.,1.]); C=np.array([1.,0.]); K=np.array([6.,3.]); L=np.array([8.,4.])
    Ac=A-np.outer(B,K); Ae=A-np.outer(L,C)
    M=np.block([[Ac,np.outer(B,K)],[np.zeros((2,2)),Ae]])
    t=np.linspace(0,5,500)
    sol=solve_ivp(lambda t,z:M@z+np.r_[8*B,[0.,0.]],(0,5),[0,0,1,-1],t_eval=t,rtol=1e-10,atol=1e-12)
    ref=solve_ivp(lambda t,z:Ac@z+8*B,(0,5),[0,0],t_eval=t,rtol=1e-10,atol=1e-12)
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    axs[0].plot(t,ref.y[0],color=BLUE,label='真状态反馈'); axs[0].plot(t,sol.y[0],color=RED,ls='--',label='观测器反馈')
    axs[0].axhline(1,color='#888',ls=':'); axs[0].legend(); axes_style(axs[0],'时间 t / s','输出')
    axs[1].plot(t,np.linalg.norm(sol.y[2:],axis=0),color=GREEN); axes_style(axs[1],'时间 t / s','估计误差二范数')
    fig.suptitle('分离原理保证联合极点；不保证不同初始估计的瞬态相同')
    save(fig,'09-observer')


def lyapunov():
    A=np.array([[-1.,2.],[0.,-2.]]); P=np.array([[.5,1/3],[1/3,7/12]])
    assert np.allclose(A.T@P+P@A,-np.eye(2))
    xx,yy=np.meshgrid(np.linspace(-3.2,3.2,200),np.linspace(-2.7,2.7,200))
    V=P[0,0]*xx**2+2*P[0,1]*xx*yy+P[1,1]*yy**2
    fig,axs=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    cs=axs[0].contour(xx,yy,V,levels=[.2,.7,1.5,3],colors='#b0b9c1'); axs[0].clabel(cs,fontsize=9)
    t=np.linspace(0,5,400)
    for init,color in [([2,1],BLUE),([-2,1],RED),([1,-2],GREEN)]:
        X=np.array([expm(A*ti)@init for ti in t]); values=np.einsum('ni,ij,nj->n',X,P,X)
        assert np.all(np.diff(values)<1e-10)
        axs[0].plot(X[:,0],X[:,1],color=color); arrow(axs[0],X[20],X[35],color)
        axs[1].plot(t,values,color=color,label=f'初态 {tuple(init)}')
    axes_style(axs[0],'x₁','x₂'); axes_style(axs[1],'时间 t / s','V(x)=xᵀPx'); axs[1].legend()
    fig.suptitle('轨迹不断穿过更小的 Lyapunov 椭圆')
    save(fig,'10-lyapunov')


def lqr():
    t=np.linspace(0,5,400); fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for q,color,style in [(1,BLUE,'-'),(4,GREEN,'--'),(16,RED,'-.')]:
        K=np.sqrt(q); x=np.exp(-K*t)
        axs[0].plot(t,x,color=color,ls=style,label=f'q={q}，K={K:g}')
        axs[1].plot(t,-K*x,color=color,ls=style,label=f'q={q}')
    axes_style(axs[0],'时间 t / s','状态 x'); axes_style(axs[1],'时间 t / s','控制 u')
    axs[0].legend(); axs[1].legend(); fig.suptitle('积分对象 LQR：输入权重 r=1、初态 x(0)=1')
    save(fig,'11-lqr')


def zoh():
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    th=np.linspace(0,2*np.pi,400); axs[0].plot(np.cos(th),np.sin(th),color='#888',ls='--',label='单位圆')
    eig=np.array([-2+0j,-1+2j,-1-2j]); points=np.exp(eig*1.2)
    axs[0].scatter(points.real,points.imag,color=BLUE,s=45,label='z=exp(sT)，T=1.2 s')
    axs[0].set(aspect='equal',xlim=(-1.2,1.2),ylim=(-1.2,1.2)); axs[0].legend(fontsize=10,loc='lower left')
    axes_style(axs[0],'Re z','Im z')
    k=np.arange(9); axs[1].plot(k,np.exp(-2*1.2*k),'o-',color=BLUE,label='精确：极点 exp(−2.4)')
    axs[1].plot(k,(-1.4)**k,'s--',color=RED,label='欧拉：极点 −1.4'); axs[1].legend(fontsize=10)
    axes_style(axs[1],'采样序号 k','零输入状态 x[k]'); fig.suptitle('精确采样与数值近似要区分')
    save(fig,'12-zoh')


if __name__ == '__main__':
    for build in [mason,routh,nyquist,root_locus,errors,lead,exponential,controllability,hidden,observer,lyapunov,lqr,zoh]:
        build()
    print(f'Generated 13 PNG figures: {OUT}')
