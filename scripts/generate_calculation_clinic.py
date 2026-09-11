"""全局 Python：计算易错专题；符号推导和独立数值检查。"""
from pathlib import Path
import json
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.optimize import brentq
from verify_formula_conventions import BLUE,RED

OUT=Path(__file__).resolve().parents[1]/'assets'/'calculation'
OUT.mkdir(parents=True,exist_ok=True)
s,z,t,K,beta=sp.symbols('s z t K beta',real=True)
report={}
def check(name,ok,**data):
    assert ok,name
    report[name]={'passed':True,**data}
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)

F=(s+3)/((s+1)**2*(s+2));decomp=-1/(s+1)+2/(s+1)**2+1/(s+2)
f=-sp.exp(-t)+2*t*sp.exp(-t)+sp.exp(-2*t)
check('01_repeated_pole',sp.cancel(F-decomp)==0 and f.subs(t,0)==0 and sp.diff(f,t).subs(t,0)==1)
check('01_selftest',sp.cancel(1/((s+1)**2*(s+2))-(-1/(s+1)+1/(s+1)**2+1/(s+2)))==0)
A=sp.Matrix([[0,1],[-2,-3]])
Phi=(2*sp.exp(-t)-sp.exp(-2*t))*sp.eye(2)+(sp.exp(-t)-sp.exp(-2*t))*A
check('02_exponential',Phi.subs(t,0)==sp.eye(2) and sp.simplify(sp.diff(Phi,t)-A*Phi)==sp.zeros(2))
AJ=sp.Matrix([[-1,2],[0,-1]]);PhiJ=sp.exp(-t)*sp.Matrix([[1,2*t],[0,1]])
check('02_repeated_eigenvalue',sp.simplify(sp.diff(PhiJ,t)-AJ*PhiJ)==sp.zeros(2) and np.allclose(np.array(PhiJ.subs(t,.7)).astype(float),expm(np.array(AJ).astype(float)*.7)))
poly=s**3+6*s*s+11*s+K
check('03_shift',sp.expand(poly.subs(s,z-1))==z**3+3*z*z+2*z+K-6,strict_interval=[6,12])
for gain,expected in [(5,False),(9,True),(13,False)]:
    pp=np.roots([1,6,11,gain]);check(f'03_roots_{gain}',bool(np.all(pp.real<-1))==expected)
check('03_boundaries',sp.expand((z+3)*(z*z+2))==sp.expand(poly.subs({s:z-1,K:12})) and poly.subs({s:-1,K:6})==0)
w=sp.symbols('w',positive=True)
den=sp.expand((sp.I*w)*(sp.I*w+1)*(sp.I*w+2))
check('04_frequency_algebra',sp.simplify(den-(-3*w*w+sp.I*w*(2-w*w)))==0 and sp.simplify(den.subs(w,sp.sqrt(2)))==-6)
wc=brentq(lambda omega:omega**2*(omega**2+1)*(omega**2+4)-9,.01,10)
loop=3/(1j*wc*(1j*wc+1)*(1j*wc+2))
check('04_frequency_numeric',abs(abs(loop)-1)<1e-10,crossover_K3=wc,gain_margin_K3=2,gain_margin_db=20*np.log10(2),phase_margin_K3=180+float(np.angle(loop,deg=True)))
for a1,gain,wn,p3 in [(8,sp.Rational(224,27),sp.Rational(4,3),sp.Rational(14,3)),(10,sp.Rational(325,27),sp.Rational(5,3),sp.Rational(13,3))]:
    check(f'05_damping_{a1}',sp.expand((s+p3)*(s*s+wn*s+wn*wn))==s**3+6*s*s+a1*s+gain,roots=[[float(v.real),float(v.imag)] for v in np.roots([1,6,a1,float(gain)])])
y0=1-(1+t)*sp.exp(-t);y1=1-(1+2*t)*sp.exp(-t)
check('06_zero_response',sp.cancel(sp.laplace_transform(y1,t,s,noconds=True)-(1-s)/(s*(s+1)**2))==0 and sp.diff(y1,t).subs(t,0)==-1 and sp.diff(y1,t).subs(t,sp.Rational(1,2))==0,minimum=float(y1.subs(t,sp.Rational(1,2))))
cross=np.log(1000)/1.9
check('06_residue_crossover',abs(.001*np.exp(-.1*cross)-np.exp(-2*cross))<1e-12,crossover=cross)
Adbl=sp.Matrix([[0,1],[0,0]]);B=sp.Matrix([0,1]);T=sp.symbols('T',positive=True)
Ad=sp.eye(2)+Adbl*T;Bd=sp.Matrix([T*T/2,T])
M=Adbl.row_join(B).col_join(sp.zeros(1,3));EM=expm(np.array(M).astype(float)*.5)
check('07_singular_zoh',Adbl.det()==0 and np.allclose(EM[:2,:2],np.array(Ad.subs(T,.5)).astype(float)) and np.allclose(EM[:2,2],np.array(Bd.subs(T,.5)).astype(float).ravel()),Ad_half=str(Ad.subs(T,sp.Rational(1,2))),Bd_half=str(Bd.subs(T,sp.Rational(1,2))))
Ap=sp.diag(-1,-2);Bp=sp.Matrix([1,beta]);k1,k2=sp.symbols('k1 k2')
pk=sp.expand((s*sp.eye(2)-Ap+Bp*sp.Matrix([[k1,k2]])).det())
check('08_parametric_rank',Bp.row_join(Ap*Bp).det()==-beta)
check('08_parametric_feedback',sp.simplify(pk.subs({k1:6,k2:-2/beta})-(s+3)*(s+4))==0,weak_beta=.001,gain=[6,-2000])
check('08_degenerate_selftest',sp.expand(pk.subs({beta:0,k1:2}))==(s*s+5*s+6))
Af=sp.Matrix([[1,1],[0,1]])
for name,x0,u0,u1 in [('main',sp.Matrix([1,0]),-1,1),('selftest',sp.Matrix([0,1]),-2,1)]:
    x1=Af*x0+B*u0;x2=Af*x1+B*u1
    check('09_finite_steps_'+name,x2==sp.zeros(2,1),u=[u0,u1],intermediate=list(map(int,x1)))

tt=np.linspace(0,8,800);v0=1-(1+tt)*np.exp(-tt);v1=1-(1+2*tt)*np.exp(-tt)
fig,ax=plt.subplots(figsize=(10,4.5));ax.plot(tt,v0,color=BLUE,label='无有限零点');ax.plot(tt,v1,'--',color=RED,label='右半平面零点 +1')
ax.scatter([.5],[1-2*np.exp(-.5)],color=RED);ax.annotate('最低点：t = 0.5 s',(.5,1-2*np.exp(-.5)),xytext=(2,-.15),arrowprops=dict(arrowstyle='->'))
ax.axhline(0,color='gray',lw=.7);ax.axhline(1,color='gray',ls=':');ax.set(xlabel='t / s',ylabel='单位阶跃输出',title='相同稳定极点与直流增益，零点改变响应');ax.legend();ax.grid(alpha=.2);save(fig,'01-zero-response')
tt=np.linspace(0,12,800);slow=.001*np.exp(-.1*tt);fast=np.exp(-2*tt)
fig,ax=plt.subplots(figsize=(10,4.5));ax.semilogy(tt,slow,color=RED,ls='--',label='慢项：0.001 exp(−0.1t)');ax.semilogy(tt,fast,color=BLUE,label='快项：exp(−2t)');ax.semilogy(tt,slow+fast,color='#228678',ls=':',label='总脉冲响应')
ax.axvline(cross,color='gray',ls=':');ax.text(cross+.3,.02,f'两项相等：约 {cross:.3f} s');ax.set(xlabel='t / s',ylabel='幅值（对数刻度）',title='极点决定衰减率，留数决定该模态的贡献',ylim=(1e-6,2));ax.legend();ax.grid(which='both',alpha=.2);save(fig,'02-residues')
bb=np.geomspace(.001,1,300);fig,ax=plt.subplots(figsize=(10,4.5));ax.loglog(bb,2/bb,color=RED,label='第二反馈系数的绝对值：2 / β');ax.scatter([.001],[2000],color=RED);ax.text(.003,1300,'β = 0.001 时，|k₂| = 2000')
ax.set(xlabel='β（固定状态坐标，β > 0）',ylabel='|k₂|',title='目标极点固定为 −3、−4：弱输入作用需要大反馈系数');ax.grid(which='both',alpha=.2);ax.legend();save(fig,'03-weak-control')
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: {len(report)} checks, 3 figures; {OUT}')
