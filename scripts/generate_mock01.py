"""全局 Python 自编模拟卷：生成图、独立核验解析式与数值结果。"""
from pathlib import Path
import json
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.optimize import brentq, linear_sum_assignment
from verify_formula_conventions import winding, half_crossings, arrow

OUT=Path(__file__).resolve().parents[1]/'assets'/'mock01'
OUT.mkdir(parents=True,exist_ok=True)
BLUE,RED='#2367a0','#bd453f'
REPORT={}
def record(n,**kw): REPORT[f'q{n:02d}']={'passed':True,**kw}
def roots(a): return [[float(z.real),float(z.imag)] for z in np.roots(a)]
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.25)
    plt.close(fig)
def base(title,size=(10,4)):
    fig,ax=plt.subplots(figsize=size); ax.set_title(title,pad=15); return fig,ax
def wire(ax,a,b): ax.plot([a[0],b[0]],[a[1],b[1]],color=BLUE,lw=2)
def resistor(ax,x,y,label):
    xx=np.linspace(x,x+1.3,10); yy=np.r_[0,np.tile([.12,-.12],4),0]+y
    ax.plot(xx,yy,color=BLUE,lw=2); ax.text(x+.65,y+.32,label,ha='center')
def cap(ax,x,y,label):
    wire(ax,(x,1.5),(x,y+.15));wire(ax,(x,y-.15),(x,-1))
    for h in [y-.15,y+.15]: wire(ax,(x-.3,h),(x+.3,h))
    ax.text(x+.4,y,label,va='center')

def verify():
    s,t=sp.symbols('s t'); K=sp.symbols('K',positive=True)
    A=sp.Matrix([[-2,1],[1,-1]]);B=sp.Matrix([1,0]);C=sp.Matrix([[0,1]])
    X=(s*sp.eye(2)-A).inv()*B
    assert sp.simplify((s+2)*X[0]-X[1]-1)==0 and sp.simplify((s+1)*X[1]-X[0])==0
    assert sp.cancel((C*X)[0]-1/(s*s+3*s+1))==0
    record(1,transfer='1/(s^2+3s+1)',poles=roots([1,3,1]))
    y=sp.Rational(2,3)+sp.exp(-t)/2-sp.exp(-3*t)/6
    assert sp.simplify(sp.diff(y,t,2)+4*sp.diff(y,t)+3*y-2)==0
    assert y.subs(t,0)==1 and sp.diff(y,t).subs(t,0)==0
    record(2,initial_y=1,initial_dy=0,final_y=2/3)
    y3=1-sp.exp(-2*t)*(sp.cos(2*sp.sqrt(3)*t)+sp.sin(2*sp.sqrt(3)*t)/sp.sqrt(3))
    tp=sp.pi/(2*sp.sqrt(3)); assert sp.simplify(sp.diff(y3,t).subs(t,tp))==0
    assert sp.simplify(y3.subs(t,tp)-1-sp.exp(-sp.pi/sp.sqrt(3)))==0
    record(3,tp=float(tp),peak=float(y3.subs(t,tp)),zeta=.5,omega_n=4)
    sb=(-8+2*sp.sqrt(7))/3; kb=(112*sp.sqrt(7)-160)/27
    assert sp.simplify(-sb*(sb+2)*(sb+6)-kb)==0
    assert sp.expand((s+8)*(s*s+12)-(s**3+8*s*s+12*s+96))==0
    record(4,breakaway=float(sb),breakaway_gain=float(kb),stable_K=[0,96],boundary_roots=roots([1,8,12,96]))
    D=s**3+5*s*s+6*s+4*K
    E=(s*(s+2)*(s+3)/D)*(2/s**2)-4*(s+2)/D/(4*s)
    assert sp.limit(s*E.subs(K,3),s,0)==sp.Rational(5,6)
    record(5,error=5/6,stable_K=[0,7.5],closed_poles_K3=roots([1,5,6,12]))
    alpha=.2;Ta=np.sqrt(5)/8
    def loop(w): return 32/(1j*w*(1j*w+4))*(1+Ta*1j*w)/(1+alpha*Ta*1j*w)
    wc=brentq(lambda w: abs(loop(w))-1,1,30);pm=180+np.angle(loop(wc),deg=True)
    poly=[alpha*Ta,1+4*alpha*Ta,4+32*Ta,32]
    assert abs(wc-8)<1e-9 and pm>65 and np.all(np.roots(poly).real<0)
    record(6,alpha=alpha,Ta=Ta,crossover=wc,phase_margin_deg=pm,closed_poles=roots(poly))
    trials=[]
    for k,expected in [(.5,-1),(2,1)]:
        for eps in [.01,.003,.001]:
            R,d=winding([k,k],[1,-1,0],eps);p,m=half_crossings([k,k],[1,-1,0],eps)
            Z=int(sum(np.roots([1,k-1,k]).real>0))
            assert abs(R-expected)<1e-7 and 2*(p-m)==expected and Z==1-expected
            trials.append(dict(K=k,epsilon=eps,R=R,N_plus=p,N_minus=m,Z=Z,min_abs_F=d))
    record(7,trials=trials,boundary_roots_K1=roots([1,0,1]))
    assert sp.exp(-2*sp.log(2)/2)==sp.Rational(1,2)
    assert sp.integrate(sp.exp(-2*t),(t,0,sp.log(2)/2))==sp.Rational(1,4)
    assert np.max(abs(np.roots([1,-.5,.75])))<1 and np.min(abs(np.roots([1,-.5,1.25])))>1
    record(8,Ad=.5,Bd=.25,stable_K_delayed=[0,4],stable_K_no_delay=[0,6],K5_delayed_poles=roots([1,-.5,1.25]))
    Ao=sp.Matrix([[0,-6],[1,-5]]);Bo=sp.Matrix([5,2]);Co=sp.Matrix([[0,1]])
    T=sp.Matrix([[3,2],[1,1]]);Ad=sp.diag(-2,-3);Bd=sp.ones(2,1);Cd=sp.ones(1,2)
    assert Ao*T==T*Ad and T*Bd==Bo and Co*T==Cd
    assert sp.cancel((Co*(s*sp.eye(2)-Ao).inv()*Bo)[0]-(2*s+5)/(s*s+5*s+6))==0
    assert Bo.row_join(Ao*Bo).det()==-1 and Co.col_join(Co*Ao).det()==-1
    record(9,similarity_matrix=[[3,2],[1,1]],controllability_det=-1,observability_det=-1)
    A=sp.diag(-1,-2,1);B=sp.Matrix([1,1,0]);eta=sp.symbols('eta');C=sp.Matrix([[1,0,eta]])
    assert B.row_join(A*B).row_join(A*A*B).rank()==2
    O=C.col_join(C*A).col_join(C*A*A)
    assert O.rank()==2 and O.subs(eta,0).rank()==1
    assert sp.cancel((C*(s*sp.eye(3)-A).inv()*B)[0]-1/(s+1))==0
    record(10,controllability_rank=2,observability_rank_eta_nonzero=2,observability_rank_eta_zero=1,stabilizable=False,detectable_iff='eta != 0',transfer='1/(s+1)')

def figures():
    fig,ax=base('第 1 题：未隔离的两级 RC 电路',(10,4))
    ax.add_patch(plt.Circle((0,.25),.4,fill=False,color=BLUE,lw=2))
    ax.text(0,.38,'+',ha='center');ax.text(0,.03,'−',ha='center');ax.text(-.65,.25,'输入 ui',ha='right')
    wire(ax,(0,.65),(0,1.5));wire(ax,(0,1.5),(1,1.5));resistor(ax,1,1.5,'R1 = 10 kΩ')
    wire(ax,(2.3,1.5),(4,1.5));resistor(ax,4,1.5,'R2 = 10 kΩ');wire(ax,(5.3,1.5),(7,1.5))
    cap(ax,3,.25,'C1 = 100 μF');cap(ax,6,.25,'C2 = 100 μF')
    wire(ax,(0,-.15),(0,-1));wire(ax,(0,-1),(7,-1))
    for yy,ww in [(-1.1,.3),(-1.2,.2),(-1.3,.1)]:wire(ax,(.7-ww,yy),(.7+ww,yy))
    wire(ax,(.7,-1),(.7,-1.1))
    ax.plot([3,6],[1.5,1.5],'o',color=BLUE);ax.text(3,1.75,'x1',ha='center');ax.text(6,1.75,'x2',ha='center')
    ax.text(7.1,.95,'输出 uo\n（开路测量）',va='center');ax.plot([7,7],[1.5,-1],'o',mfc='white',mec=BLUE)
    ax.set(xlim=(-1.5,8.5),ylim=(-1.7,2.3));ax.axis('off');save(fig,'q01-circuit')
    fig,ax=base('第 3 题：输入为幅值 2 的阶跃')
    t=np.linspace(0,3.5,1200);y=1-np.exp(-2*t)*(np.cos(2*np.sqrt(3)*t)+np.sin(2*np.sqrt(3)*t)/np.sqrt(3))
    tp=np.pi/(2*np.sqrt(3));peak=1+np.exp(-np.pi/np.sqrt(3))
    ax.plot(t,y,color=BLUE);ax.axhline(1,color='gray',ls='--',label='稳态值 1')
    ax.plot(tp,peak,'o',color=RED);ax.annotate(f'峰值约 {peak:.4f}\n峰值时间约 {tp:.4f} s',(tp,peak),xytext=(1.5,1.25),arrowprops=dict(arrowstyle='->'))
    ax.set(xlabel='t / s',ylabel='输出 y(t)',ylim=(0,1.5));ax.grid(alpha=.2);ax.legend();save(fig,'q03-step-question')
    fig,ax=base('第 5 题：扰动进入位置与误差信号',(11,3))
    from matplotlib.patches import Rectangle
    for x,label in [(2,'C(s) = K / (s + 2)'),(6,'P(s) = 4 / [s(s + 3)]')]:
        ax.add_patch(Rectangle((x,1),2,1,fill=False,color=BLUE,lw=2));ax.text(x+1,1.5,label,ha='center',va='center',fontsize=11)
    for x in [1,5]:ax.add_patch(plt.Circle((x,1.5),.2,fill=False,color=BLUE,lw=2))
    for a,b in [((0,1.5),(.8,1.5)),((1.2,1.5),(2,1.5)),((4,1.5),(4.8,1.5)),((5.2,1.5),(6,1.5)),((8,1.5),(9.3,1.5)),((5,2.6),(5,1.7)),((1,.2),(1,1.3))]:arrow(ax,a,b)
    wire(ax,(8.7,1.5),(8.7,.2));wire(ax,(8.7,.2),(1,.2))
    for x,y,tx in [(0,1.8,'r'),(1.5,1.8,'e'),(9.2,1.8,'y'),(5.25,2.4,'n'),(.7,1.7,'+'),(1.2,.95,'−'),(4.6,1.8,'+'),(5.2,1.9,'+')]:ax.text(x,y,tx)
    ax.set(xlim=(-.2,9.8),ylim=(-.1,2.9));ax.axis('off');save(fig,'q05-disturbance')
    w=np.geomspace(.1,100,1000);db=np.where(w<=4,20*np.log10(8/w),20*np.log10(32/w**2))
    fig,ax=base('第 6 题：开环幅频渐近线（题设）')
    ax.semilogx(w,db,color=BLUE);ax.scatter([1,4],[20*np.log10(8),20*np.log10(2)],color=RED)
    ax.annotate('ω = 1：20 log₁₀ 8 dB',(1,20*np.log10(8)),xytext=(.15,5),arrowprops=dict(arrowstyle='->'))
    ax.axvline(4,color='gray',ls='--');ax.text(5,15,'转折频率 4 rad/s');ax.text(.15,32,'−20 dB/dec');ax.text(15,-20,'−40 dB/dec')
    ax.set(xlabel='ω / (rad/s)',ylabel='幅值 / dB');ax.grid(which='both',alpha=.2);save(fig,'q06-bode-question')
    fig,ax=base('第 4 题：正增益根轨迹（箭头表示 K 增大）',(9,6))
    gains=np.r_[np.linspace(0,8,1200),np.linspace(8.1,200,1800)];branches=[];prev=np.array([-6,-2,0],complex)
    for k in gains:
        rr=np.roots([1,8,12,k]);ii,jj=linear_sum_assignment(abs(prev[:,None]-rr[None,:]));prev=rr[jj];branches.append(prev.copy())
    b=np.array(branches)
    for j in range(3):
        ax.plot(b[:,j].real,b[:,j].imag,color=BLUE)
        arrow(ax,(b[1600,j].real,b[1600,j].imag),(b[1700,j].real,b[1700,j].imag))
    for ang in [60,180,300]:
        r=np.linspace(0,10,100);ax.plot(-8/3+r*np.cos(np.deg2rad(ang)),r*np.sin(np.deg2rad(ang)),ls='--',color='gray',alpha=.6)
    ax.scatter([-6,-2,0],[0,0,0],marker='x',s=90,color=RED,label='开环极点')
    ax.scatter([0,0],[np.sqrt(12),-np.sqrt(12)],color=RED,label='K = 96')
    ax.annotate('分离点约 −0.9028',(-.9028,0),xytext=(-4,1.4),arrowprops=dict(arrowstyle='->'))
    ax.axvline(0,color='black',lw=.7);ax.axhline(0,color='black',lw=.7);ax.set(xlim=(-10,3),ylim=(-6,6),xlabel='Re s',ylabel='Im s');ax.legend();ax.grid(alpha=.2);save(fig,'a04-root-locus')
    Ta=np.sqrt(5)/8;L=32/(1j*w*(1j*w+4));Lc=L*(1+Ta*1j*w)/(1+.2*Ta*1j*w)
    fig,aa=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for values,label,color in [(L,'校正前',BLUE),(Lc,'校正后',RED)]:
        aa[0].semilogx(w,20*np.log10(abs(values)),label=label,color=color)
        aa[1].semilogx(w,np.unwrap(np.angle(values))*180/np.pi,label=label,color=color)
    aa[0].set_title('第 6 题：精确频响复核');aa[0].axhline(0,color='gray',ls='--');aa[0].set_ylabel('幅值 / dB');aa[0].legend()
    aa[1].axhline(-180,color='gray',ls='--');aa[1].scatter([8],[-111.6246336],color=RED);aa[1].text(.13,-145,'校正后：ωc = 8 rad/s，PM ≈ 68.38°')
    for ax in aa:ax.axvline(8,color='gray',ls=':');ax.grid(which='both',alpha=.2)
    aa[1].set(xlabel='ω / (rad/s)',ylabel='相位 / °');fig.tight_layout();save(fig,'a06-lead-check')
    fig,aa=plt.subplots(1,3,figsize=(14,4.5))
    w7=np.geomspace(.025,30,4000)
    for ax,k in zip(aa[:2],[.5,2]):
        z=k*(1j*w7+1)/(1j*w7*(1j*w7-1));ax.plot(z.real,z.imag,color=BLUE)
        idx=np.searchsorted(w7,.8);arrow(ax,(z[idx].real,z[idx].imag),(z[idx+170].real,z[idx+170].imag))
        ax.scatter([-1], [0],marker='x',s=80,color=RED,label='临界点 −1');ax.axvline(-2*k,color='gray',ls='--',label='零频渐近线')
        ax.axhline(0,color='gray',lw=.7);ax.set(xlim=(-2*k-.5,.5),ylim=(-2,4),xlabel='Re L',ylabel='Im L',title=f'正频率支：K = {k:g}');ax.legend(fontsize=9)
    theta=np.linspace(-np.pi/2,np.pi/2,600);z=-np.exp(-1j*theta);ax=aa[2];ax.plot(z.real,z.imag,color=BLUE)
    arrow(ax,(z[260].real,z[260].imag),(z[340].real,z[340].imag));ax.axhline(0,color='gray');ax.axvline(0,color='gray');ax.set_aspect('equal')
    ax.set(title='完整小绕避的主导映射\n归一化后：左半圆，顺时针',xlabel='Re（归一化）',ylabel='Im（归一化）',xlim=(-1.3,.3),ylim=(-1.3,1.3))
    ax.text(-1.2,-1.2,'起点：下端 → 左端 → 上端',fontsize=9);fig.tight_layout();save(fig,'a07-nyquist')
    fig,aa=plt.subplots(1,2,figsize=(10,4.5));th=np.linspace(0,2*np.pi,500)
    for ax,k in zip(aa,[3,5]):
        ax.plot(np.cos(th),np.sin(th),color='gray',ls='--',label='单位圆')
        p=np.roots([1,-.5,k/4]);ax.scatter(p.real,p.imag,color=BLUE,label='有一拍延迟',s=65);ax.scatter([.5-k/4],[0],color=RED,marker='x',label='无额外延迟',s=80)
        ax.axhline(0,color='gray',lw=.5);ax.axvline(0,color='gray',lw=.5);ax.set(title=f'第 8 题：K = {k}',xlabel='Re z',ylabel='Im z',xlim=(-1.3,1.3),ylim=(-1.3,1.3));ax.set_aspect('equal');ax.legend(fontsize=9)
    fig.tight_layout();save(fig,'a08-delay-poles')

if __name__=='__main__':
    verify();figures()
    (OUT/'verification.json').write_text(json.dumps(REPORT,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'PASS: {len(REPORT)} questions; 8 figures; {OUT}')
