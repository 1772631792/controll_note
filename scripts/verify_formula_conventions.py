"""全局 Python：判据写法、Nyquist 零频与绕数验证；不使用虚拟环境。

运行 D:\\Python310\\python.exe scripts/verify_formula_conventions.py
生成 assets/conventions 下 PNG 和 verification.json。
绕数从完整右半平面顺时针围线映射计算，原点极点向右凹陷排除。
"""
from pathlib import Path
import json
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets'/'conventions'
OUT.mkdir(parents=True,exist_ok=True)
font=Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=[font_manager.FontProperties(fname=font).get_name(),'DejaVu Sans']
plt.rcParams.update({'font.size':12,'axes.unicode_minus':False,'figure.facecolor':'white',
                     'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'dejavusans'})
BLUE,RED,GREEN,GOLD='#2367a0','#bd453f','#228678','#b77b20'

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.22)
    plt.close(fig)

def arrow(ax,p,q,color=BLUE):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=15,lw=1.8,color=color))

def curve_arrow(ax,z,index,step=20,color=BLUE):
    arrow(ax,(z[index].real,z[index].imag),(z[index+step].real,z[index+step].imag),color)

def contour(eps=1e-3,radius=1e3):
    # -jR -> -j eps -> small right CCW semicircle -> +jR -> large right CW semicircle.
    negative=-1j*np.geomspace(radius,eps,9000)
    indent=eps*np.exp(1j*np.linspace(-np.pi/2,np.pi/2,9001))
    positive=1j*np.geomspace(eps,radius,9000)
    outer=radius*np.exp(1j*np.linspace(np.pi/2,-np.pi/2,9001))
    return np.r_[negative,indent[1:],positive[1:],outer[1:]]

def winding(num,den,eps):
    s=contour(eps)
    F=1+np.polyval(num,s)/np.polyval(den,s)
    assert np.min(abs(F))>1e-5, '围线过于接近临界点，不能可靠计数'
    darg=np.angle(F[1:]/F[:-1])
    assert np.max(abs(darg))<.2, '采样不够密'
    ccw=float(darg.sum()/(2*np.pi))
    return ccw, float(np.min(abs(F)))

def half_crossings(num,den,eps):
    """直接数扩展正频率半边的射线穿越；包含上半小绕避和有限端点半次。"""
    radius=1e3
    arc=eps*np.exp(1j*np.linspace(0,np.pi/2,12001))
    axis=1j*np.geomspace(eps,radius,12000)
    outer=radius*np.exp(1j*np.linspace(np.pi/2,0,12001))
    s=np.r_[arc,axis[1:],outer[1:]]
    z=np.polyval(num,s)/np.polyval(den,s)
    nz=np.flatnonzero(abs(z.imag)>1e-10)
    plus=minus=0.
    for i,j in zip(nz[:-1],nz[1:]):
        y0,y1=z[i].imag,z[j].imag
        if y0*y1<0:
            x=z[i].real+(z[j].real-z[i].real)*(-y0)/(y1-y0)
            if x<-1:
                if y0>0:plus+=1
                else:minus+=1
    if abs(z[0].imag)<1e-10 and z[0].real<-1:
        if z[nz[0]].imag<0:plus+=.5
        else:minus+=.5
    if abs(z[-1].imag)<1e-10 and z[-1].real<-1:
        if z[nz[-1]].imag>0:plus+=.5
        else:minus+=.5
    return plus,minus

def verify():
    cases=[
        ('L=2/(s-1)',[2],[1,-1],.5,0),
        ('L=0.5/(s-1)',[.5],[1,-1],0,0),
        ('L=4/[s(s+1)(s+2)]',[4],[1,3,2,0],0,0),
        ('L=12/[s(s+1)(s+2)]',[12],[1,3,2,0],0,1),
        ('L=1/[s(s+1)]',[1],[1,1,0],0,0),
        ('L=1/[s^2(s+1)]',[1],[1,1,0,0],0,1),
        ('L=(s+1)/s^2',[1,1],[1,0,0],0,0),
        ('L=1/s^2 (boundary)',[1],[1,0,0],None,None),
    ]
    result=[]
    for name,num,den,nplus,nminus in cases:
        poles=np.roots(den); closed=np.roots(np.polyadd(den,num))
        P=int(np.sum(poles.real>1e-7)); Z=int(np.sum(closed.real>1e-7))
        boundary=bool(np.any(abs(closed.real)<1e-7))
        item={'model':name,'P':P,'Z_roots':Z,'imaginary_axis_closed_poles':boundary,
              'closed_poles':[[float(x.real),float(x.imag)] for x in closed]}
        if not boundary:
            trials=[]
            for eps in [.01,.003,.001]:
                R,distance=winding(num,den,eps)
                measured_plus,measured_minus=half_crossings(num,den,eps)
                assert abs(R-round(R))<1e-7
                assert abs(P-R-Z)<1e-7
                assert (measured_plus,measured_minus)==(nplus,nminus)
                trials.append({'epsilon':eps,'R_ccw':round(R),'N_plus_measured':measured_plus,
                               'N_minus_measured':measured_minus,'min_abs_1_plus_L':distance})
            assert 2*(nplus-nminus)==trials[-1]['R_ccw']
            item.update(N_plus=nplus,N_minus=nminus,R=trials[-1]['R_ccw'],trials=trials)
        else:
            item['note']='虚轴闭环根：不套严格稳定 Nyquist 计数；即使 Z=0 也不宣布稳定。'
        result.append(item)
    # Independent symbolic equivalence checks.
    z,w,a,b=sp.symbols('z w a b',real=True)
    transformed=sp.cancel((1-w)**2*((z*z+a*z+b).subs(z,(1+w)/(1-w))))
    assert sp.expand(transformed-((1-a+b)*w*w+2*(1-b)*w+1+a+b))==0
    s=sp.symbols('s'); alpha=sp.Rational(1,4); T=sp.Integer(2); ratio=1/alpha; tau=alpha*T
    assert sp.cancel((1+T*s)/(1+alpha*T*s)-(1+ratio*tau*s)/(1+tau*s))==0
    A=sp.Matrix([[0,1],[-2,-3]]); B=sp.Matrix([0,1]); C=sp.Matrix([[1,0]]); S=sp.Matrix([[1,1],[0,1]])
    lhs=(C*S*(s*sp.eye(2)-S.inv()*A*S).inv()*S.inv()*B)[0]
    assert sp.cancel(lhs-(C*(s*sp.eye(2)-A).inv()*B)[0])==0
    p11,p12,p22=sp.symbols('p11 p12 p22');P=sp.Matrix([[p11,p12],[p12,p22]])
    K=B.T*P;Ac=A-B*K
    assert sp.simplify(Ac.T*P+P*Ac+K.T*K-(A.T*P+P*A-P*B*B.T*P))==sp.zeros(2)
    assert sp.limit(1/(s*(s+1))-1/s,s,0)==-1
    assert sp.limit(s*(1/(s*s*(s+1))-1/s**2),s,0)==-1
    (OUT/'verification.json').write_text(json.dumps({'nyquist':result,
        'symbolic_checks':['Jury via bilinear transform','lead parameter reciprocity','similarity transfer invariance',
                           'continuous CARE closed-loop equivalence','Laurent coefficients at zero']},
        ensure_ascii=False,indent=2),encoding='utf-8')
    return result

def crossing_plot():
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for ax,down in zip(axs,[True,False]):
        ax.axhline(0,color='#aaa',lw=1)
        ax.plot([-3,-1],[0,0],color=GOLD,lw=4,alpha=.5,label='计数射线 (−∞,−1)')
        ax.scatter([-1],[0],marker='x',s=70,color=RED)
        yy=np.linspace(.7,-.7,100) if down else np.linspace(-.7,.7,100)
        xx=-2+.3*yy
        ax.plot(xx,yy,lw=2,color=BLUE);arrow(ax,(xx[40],yy[40]),(xx[60],yy[60]))
        ax.set(xlim=(-3.1,-.3),ylim=(-1,1),xlabel='Re L',ylabel='Im L',
               title='上 → 下：正穿越 N₊ 加 1' if down else '下 → 上：负穿越 N₋ 加 1')
        ax.text(-2.9,-.85,'只数临界点左侧；箭头沿频率增加方向',fontsize=10)
    save(fig,'01-crossing-sign')

def half_and_bode():
    fig,axs=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    w=np.geomspace(1e-5,100,1800);L=2/(-1+1j*w)
    axs[0].plot(L.real,L.imag,color=BLUE,label='ω: 0⁺ → +∞')
    axs[0].plot(L.real,-L.imag,color=BLUE,ls=':',alpha=.3,label='负频率镜像')
    axs[0].scatter([-2,-1],[0,0],color=[GREEN,RED],s=45)
    curve_arrow(axs[0],L,1100,30)
    axs[0].annotate('起点 −2，离开到下侧：+1/2',(-2,0),(-2.4,.55),arrowprops={'arrowstyle':'->'})
    axs[0].set(xlim=(-2.6,.3),ylim=(-1.3,1),title='L=2/(s−1)：端点半次',xlabel='Re L',ylabel='Im L')
    axs[0].legend(loc='lower left',fontsize=9)
    w=np.geomspace(.12,10,2200);L=12/((1j*w)*(1+1j*w)*(2+1j*w))
    axs[1].plot(L.real,L.imag,color=BLUE); axs[1].scatter([-2,-1],[0,0],color=[GREEN,RED],s=40)
    idx=np.argmin(abs(w-np.sqrt(2)));curve_arrow(axs[1],L,idx-25,50)
    axs[1].annotate('ω=√2：负穿越一次',(-2,0),(-3.4,1.1),arrowprops={'arrowstyle':'->'})
    axs[1].set(xlim=(-4.5,.5),ylim=(-3,2),title='L=12/[s(s+1)(s+2)]',xlabel='Re L',ylabel='Im L')
    phase=-90-np.degrees(np.arctan(w))-np.degrees(np.arctan(w/2))
    axs[2].semilogx(w,phase,color=BLUE);axs[2].axhline(-180,color=GOLD,ls='--')
    axs[2].scatter([np.sqrt(2)],[-180],color=GREEN)
    axs[2].annotate('相位向下穿越 −180°\n此处幅值 > 1，故计负穿越',(.9,-145),fontsize=10)
    axs[2].set(title='同一个穿越在 Bode 图上',xlabel='ω / (rad/s)',ylabel='展开相位 / °')
    for ax in axs:ax.grid(alpha=.16)
    save(fig,'02-half-crossing-bode')

def indent_plot():
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    theta=np.linspace(-np.pi/2,np.pi/2,1000);s=np.exp(1j*theta)
    axs[0].plot(s.real,s.imag,color=GREEN)
    curve_arrow(axs[0],s,420,110,GREEN)
    axs[0].scatter([0],[0],marker='x',s=65,color=RED)
    axs[0].text(.12,-.12,'原点极点在围线外',fontsize=10)
    axs[0].set(title='s 平面：小右半圆逆时针 180°',xlim=(-.3,1.4),ylim=(-1.3,1.3),xlabel='Re(s/ε)',ylabel='Im(s/ε)')
    for ax,nu in zip(axs[1:],[1,2]):
        image=np.exp(-1j*nu*theta)
        ax.plot(image.real,image.imag,color=BLUE)
        for k in ([420] if nu==1 else [170,670]):curve_arrow(ax,image,k,70)
        ax.scatter([image[0].real],[image[0].imag],color=GREEN,s=45)
        ax.set(title=f'ν={nu}：像弧顺时针 {nu*180}°',xlim=(-1.3,1.3),ylim=(-1.3,1.3),
               xlabel='归一化像弧实部',ylabel='归一化像弧虚部')
    for ax in axs:ax.set_aspect('equal');ax.axhline(0,color='#ccc',lw=.7);ax.axvline(0,color='#ccc',lw=.7)
    fig.suptitle('示意比例：真实像弧半径约为 |g₀|/ε^ν，ε→0 时趋于无穷')
    save(fig,'03-origin-indent')

def low_frequency():
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    w=np.geomspace(.07,6,2000)
    L=1/((1j*w)*(1+1j*w));axs[0].plot(L.real,L.imag,color=BLUE)
    axs[0].axvline(-1,color=RED,ls='--',label='实部渐近线 = −1')
    curve_arrow(axs[0],L,500,60);axs[0].set(xlim=(-1.3,.2),ylim=(-15,1),title='单积分：不能只记相角 −90°')
    axs[0].legend(fontsize=10)
    for ax,lead in zip(axs[1:],[False,True]):
        L=(1+1j*w)/(1j*w)**2 if lead else 1/((1j*w)**2*(1+1j*w))
        ax.plot(L.real,L.imag,color=GREEN if lead else BLUE)
        idx=np.argmin(abs(w-.8));curve_arrow(ax,L,idx,50,GREEN if lead else BLUE)
        ax.scatter([-1],[0],marker='x',s=50,color=RED);ax.axhline(0,color='#888',lw=.8)
        ax.set(xlim=(-18,1),ylim=(-6,6),title='双积分+超前零点：从下侧出发' if lead else '双积分+惯性极点：从上侧出发')
    for ax in axs:ax.set_xlabel('Re L(jω)');ax.set_ylabel('Im L(jω)');ax.grid(alpha=.16)
    fig.suptitle('零频相角相同不代表轨迹位于同一侧；图示为正频率支路的局部窗口')
    save(fig,'04-low-frequency-side')

def jury_lead():
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    axs[0].fill([-2,2,0],[1,1,-1],color=GREEN,alpha=.18)
    axs[0].plot([-2,2,0,-2],[1,1,-1,1],color=GREEN,ls='--')
    axs[0].scatter([-.8],[.3],color=BLUE,s=50,label='a₁=−0.8，a₀=0.3：稳定')
    axs[0].set(xlim=(-2.5,2.5),ylim=(-1.4,1.4),xlabel='a₁',ylabel='a₀',title='二阶 Jury：两套条件给出同一开三角形')
    axs[0].legend(fontsize=9,loc='upper center');axs[0].grid(alpha=.16)
    w=np.geomspace(.01,100,500);s=1j*w
    form1=(1+2*s)/(1+.5*s);form2=(1+4*.5*s)/(1+.5*s)
    assert np.allclose(form1,form2)
    axs[1].semilogx(w,20*np.log10(abs(form1)),color=BLUE,label='α=1/4，T=2')
    axs[1].semilogx(w[::22],20*np.log10(abs(form2[::22])),marker='o',ls='none',ms=4,color=RED,label='a=4，τ=1/2')
    axs[1].set(xlabel='ω / (rad/s)',ylabel='幅值 / dB',title='超前网络：参数取倒数，曲线完全重合')
    axs[1].legend();axs[1].grid(alpha=.16)
    save(fig,'05-jury-lead-equivalence')

if __name__=='__main__':
    results=verify()
    for fn in [crossing_plot,half_and_bode,indent_plot,low_frequency,jury_lead]:fn()
    print(f'Validated {len(results)} Nyquist cases, 3 epsilon values per non-boundary case, measured half-crossings; generated 5 PNGs.')
