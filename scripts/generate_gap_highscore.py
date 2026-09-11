"""全局 Python：第 14、15 份的符号核验、独立积分与配图。"""
from pathlib import Path
import json
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp,quad
from verify_formula_conventions import BLUE,RED,arrow

ROOT=Path(__file__).resolve().parents[1]
GAP=ROOT/'assets'/'gaps';HIGH=ROOT/'assets'/'highscore';OUT=ROOT/'assets'/'gaps_highscore'
for p in [GAP,HIGH,OUT]:p.mkdir(parents=True,exist_ok=True)
report={}
def check(name,ok,**data):
    assert bool(ok),name
    report[name]={'passed':True,**data}
def save(fig,folder,name):
    fig.savefig(folder/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.2);plt.close(fig)
def integrate(fun,span,x0,times):
    sol=solve_ivp(fun,span,x0,t_eval=times,rtol=1e-10,atol=1e-12)
    assert sol.success,sol.message
    return sol.y

s,z,t,tau,k,a,b,th,v=sp.symbols('s z t tau k a b th v',real=True)
x=sp.symbols('x',real=True)
sol=1/sp.sqrt(1+2*t)
check('14_1_scalar',sp.simplify(sp.diff(sol,t)+sol**3)==0 and sp.diff(-x**3,x).subs(x,0)==0 and sp.expand(x*(-x**3))==-x**4)
energy=v*v/2+1-sp.cos(th)
check('14_2_energy',sp.simplify(sp.diff(energy,th)*v+sp.diff(energy,v)*(-sp.sin(th)-v))==-v*v)
check('14_2_linearization',sp.Matrix([[0,1],[-1,-1]]).det()==1 and sp.Matrix([[0,1],[1,-1]]).det()==-1)
amplitude=2/(3*np.pi);w0=np.sqrt(2)
Gj=lambda w:1/(1j*w*(1j*w+1)*(1j*w+2))
N=lambda amp:4/(np.pi*amp)
harmonic=abs(Gj(3*w0)/Gj(w0))/3
check('14_3_harmonic_balance',abs(1+Gj(w0)*N(amplitude))<1e-12 and abs(quad(lambda q:abs(np.sin(q)),-np.pi,np.pi)[0]/np.pi-4/np.pi)<1e-12,amplitude=amplitude,omega=w0,relative_third_harmonic=harmonic)
Phi=sp.exp(-(t*t-tau*tau));xs=1-sp.exp(-t*t)
check('14_4_scalar_transition',sp.simplify(sp.diff(Phi,t)+2*t*Phi)==0 and Phi.subs(t,tau)==1 and sp.simplify(sp.diff(xs,t)+2*t*xs-2*t)==0)
A1=sp.Matrix([[0,1],[0,0]]);A2=A1.T;product=(sp.eye(2)+A2)*(sp.eye(2)+A1)
check('14_4_noncommuting',product==sp.Matrix([[1,1],[1,2]]) and A1*A2!=A2*A1 and not np.allclose(np.array(product).astype(float),expm(np.array(A1+A2).astype(float))),transition=str(product))
At=sp.Matrix([[-1,sp.exp(3*t)],[0,-2]]);xt=sp.Matrix([(sp.exp(t)-sp.exp(-t))/2,sp.exp(-2*t)])
check('14_5_frozen_eigenvalue_counterexample',sp.simplify(sp.diff(xt,t)-At*xt)==sp.zeros(2,1) and xt.subs(t,0)==sp.Matrix([0,1]))
leadpoly=s*(s+2)*(s+8)+16*(s+3);sd=-2+2j
check('14_6_lead',sp.expand(leadpoly-(s+6)*(s*s+4*s+8))==0 and abs(16*(sd+3)/(sd*(sd+2)*(sd+8))+1)<1e-12,gain=16,compensator_pole=-8,closed_poles=[[-6,0],[-2,2],[-2,-2]],Kv=3)
D=s**3+2*s*s+k*s+k*a;shift=sp.expand(D.subs(s,z-sp.Rational(1,2)))
check('15_1_shift',sp.expand(shift-(z**3+z*z/2+(k-sp.Rational(5,4))*z+k*a-k/2+sp.Rational(3,8)))==0)
pp=np.roots([1,2,12,9]);error=sp.limit(s*(s*s*(s+2)/D)/s**3,s,0)
check('15_1_feasible',np.all(pp.real<-.5) and sp.simplify(error-2/(k*a))==0,example={'k':12,'a':.75,'error':2/9,'poles':[[float(q.real),float(q.imag)] for q in pp]})
check('15_1_boundary',sp.expand(shift.subs({k:9,a:sp.Rational(8,9)})-(z+sp.Rational(1,2))*(z*z+sp.Rational(31,4)))==0)
A=sp.Matrix([[0,1],[-2,-3]]);B=sp.Matrix([0,1]);C=sp.Matrix([[1,0]]);K=sp.Matrix([[10,4]]);L=sp.Matrix([8,4]);AK=A-B*K;AO=A-L*C
ef=AO.inv()*L*b;xf=-AK.inv()*B*(12+(K*ef)[0])
check('15_2_bias',ef==sp.Matrix([-sp.Rational(14,15),sp.Rational(8,15)])*b and sp.simplify((C*xf)[0]-(1-sp.Rational(3,5)*b))==0 and set(AK.eigenvals())=={-3,-4} and set(AO.eigenvals())=={-5,-6},bias_multiplier=-.6)
Ab=sp.diag(A,0);Cb=sp.Matrix([[1,0,1]]);Ob=Cb.col_join(Cb*Ab).col_join(Cb*Ab*Ab)
check('15_2_augmented_observability',Ob.det()==2)
Bt=sp.Matrix([1,t]);W=sp.integrate(Bt*Bt.T,(t,0,1));uf=(Bt.T*W.inv()*sp.Matrix([0,1]))[0]
check('15_3_gramian',W.det()==sp.Rational(1,12) and uf==12*t-6 and sp.integrate(Bt*uf,(t,0,1))==sp.Matrix([0,1]) and sp.integrate(uf*uf,(t,0,1))==12,W=str(W),minimum_energy=12)
AF=sp.Matrix([[1-k,-3*k],[-1,-2]]);BF=sp.Matrix([k,1]);CF=sp.Matrix([[1,0]])
check('15_4_hidden_mode',sp.expand((s*sp.eye(2)-AF).det()-(s-1)*(s+2+k))==0 and sp.cancel((CF*(s*sp.eye(2)-AF).inv()*BF)[0]-k/(s+2+k))==0)
y=sp.Rational(3,4)*sp.exp(t)+sp.exp(-3*t)/4
check('15_4_initial_response',sp.simplify((CF*(AF.subs(k,1)*t).exp()*sp.Matrix([1,0]))[0]-y)==0)

# Independent ODE calculations, then pictures.
fig,ax=plt.subplots(figsize=(10,6))
angles=np.linspace(-3.3,3.3,27);vel=np.linspace(-2.8,2.8,23);TH,V=np.meshgrid(angles,vel);U=V;VV=-np.sin(TH)-V;length=np.sqrt(U*U+VV*VV)+1e-12
ax.quiver(TH,V,U/length,VV/length,color='#b4bdc7',alpha=.8)
grid=np.linspace(-np.pi,np.pi,400);bound=np.sqrt(np.maximum(0,2*(1+np.cos(grid))))
ax.plot(grid,bound,'k--',label='E = 2 边界');ax.plot(grid,-bound,'k--')
for i,x0 in enumerate([[2,0],[-2,.3],[0,1.5]]):
    tt=np.linspace(0,15,700);xx=integrate(lambda t,q:[q[1],-np.sin(q[0])-q[1]],(0,15),x0,tt)
    en=.5*xx[1]**2+1-np.cos(xx[0]);check(f'14_2_trajectory_{i}',np.max(np.diff(en))<1e-7 and np.linalg.norm(xx[:,-1])<.003)
    ax.plot(xx[0],xx[1],color=[BLUE,RED,'#228678'][i],ls=['-','-.',':'][i],label=f'初态 {x0}');ax.scatter([x0[0]],[x0[1]],s=25)
ax.scatter([0],[0],color='black',s=35);ax.scatter([-np.pi,np.pi],[0,0],marker='x',color=RED,s=70)
ax.set(xlabel='θ / rad',ylabel='v / (rad/s)',title='阻尼摆：相轨迹、方向场与能量边界',xlim=(-3.4,3.4),ylim=(-2.8,2.8));ax.legend(fontsize=10,loc='upper right');save(fig,GAP,'01-pendulum')
fig,aa=plt.subplots(1,2,figsize=(11,4));tt=np.linspace(0,3,600)
num=integrate(lambda t,q:[-2*t*q[0]+2*t],(0,3),[0],tt)[0]
check('14_4_numeric_response',np.max(abs(num-(1-np.exp(-tt**2))))<1e-8)
aa[0].plot(tt,num,color=BLUE,label='x = 1 − exp(−t²)');aa[0].set(title='时变受迫响应',xlabel='t / s',ylabel='x');aa[0].legend()
num2=integrate(lambda t,q:[-q[0]+np.exp(3*t)*q[1],-2*q[1]],(0,3),[0,1],tt)
check('14_5_numeric_counterexample',np.max(abs(num2[0]-(np.exp(tt)-np.exp(-tt))/2))<1e-6)
aa[1].plot(tt,num2[0],color=RED,label='第一状态：增长');aa[1].plot(tt,num2[1],'--',color=BLUE,label='第二状态：衰减');aa[1].set(title='瞬时特征值负，实际解发散',xlabel='t / s',ylabel='状态');aa[1].legend()
for ax in aa:ax.grid(alpha=.2)
fig.tight_layout();save(fig,GAP,'02-timevarying')
fig,aa=plt.subplots(1,2,figsize=(12,4.5))
for q,lab,color,mark in [(0,'原极点 0',BLUE,'x'),(-2,'原极点 −2',BLUE,'x'),(-3,'补偿零点 −3','#228678','o'),(-8,'补偿极点 −8',RED,'x')]:
    aa[0].plot([q,-2],[0,2],ls='--',color=color,alpha=.7);aa[0].scatter([q],[0],color=color,marker=mark,s=70,label=lab)
aa[0].scatter([-2],[2],color='black',label='目标 −2 + j2');aa[0].set(title='角度条件：零点贡献减极点贡献',xlabel='Re s',ylabel='Im s',ylim=(-.5,3));aa[0].legend(fontsize=9)
aa[1].axvspan(-9,0,color='#e7f2e9');aa[1].scatter([-6,-2,-2],[0,2,-2],marker='x',color=RED,s=90);aa[1].set(title='回代完整特征式：三个闭环极点',xlabel='Re s',ylabel='Im s',xlim=(-9,1),ylim=(-3,3))
for ax in aa:ax.axhline(0,color='gray',lw=.6);ax.axvline(0,color='gray',lw=.6);ax.grid(alpha=.2)
fig.tight_layout();save(fig,GAP,'03-lead-design')
An=np.array(A).astype(float);Bn=np.array(B).astype(float).ravel();Cn=np.array(C).astype(float).ravel();Kn=np.array(K).astype(float).ravel();Ln=np.array(L).astype(float).ravel()
def dynamics(bias):
    def fun(t,q):
        state=q[:2];est=q[2:];u=-Kn@est+12
        return np.r_[An@state+Bn*u,An@est+Bn*u+Ln*(Cn@state+bias-Cn@est)]
    return fun
t1=np.linspace(0,3,300);y1=integrate(dynamics(0),(0,3),np.zeros(4),t1)
t2=np.linspace(3,10,650);y2=integrate(dynamics(.2),(3,10),y1[:,-1],t2)
check('15_2_numeric_bias',abs(y2[0,-1]-.88)<1e-7,final_output=float(y2[0,-1]))
fig,ax=plt.subplots(figsize=(10,4));ax.plot(np.r_[t1,t2],np.r_[y1[0],y2[0]],color=BLUE,label='真实输出');ax.axhline(1,color='gray',ls='--',label='参考 1');ax.axhline(.88,color=RED,ls=':',label='预测终值 0.88');ax.axvline(3,color='gray',ls=':');ax.set(xlabel='t / s',ylabel='y',title='三秒时加入测量偏置 0.2，原控制器保持不变');ax.legend();ax.grid(alpha=.2);save(fig,HIGH,'01-bias')
tt=np.linspace(0,1,300);numeric=integrate(lambda t,q:[-6+12*t,t*(-6+12*t)],(0,1),[0,0],tt)
check('15_3_numeric_terminal',np.allclose(numeric[:,-1],[0,1],atol=1e-9))
fig,aa=plt.subplots(1,2,figsize=(11,4));aa[0].plot(tt,-6+12*tt,color=BLUE);aa[0].set(title='最小能量控制',xlabel='t / s',ylabel='u')
aa[1].plot(tt,numeric[0],color=BLUE,label='第一状态');aa[1].plot(tt,numeric[1],'--',color=RED,label='第二状态');aa[1].set(title='终态：第一状态 0，第二状态 1',xlabel='t / s',ylabel='状态');aa[1].legend()
for ax in aa:ax.grid(alpha=.2)
fig.tight_layout();save(fig,HIGH,'02-minimum-energy')
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: {len(report)} checks; 5 figures; {OUT}')
