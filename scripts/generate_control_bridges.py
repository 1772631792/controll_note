"""全局 Python：线控与自控贯通专题的符号核验与计算配图。"""
from pathlib import Path
import json
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy.linalg import expm
from verify_formula_conventions import BLUE, RED, arrow

OUT=Path(__file__).resolve().parents[1]/'assets'/'bridges'
OUT.mkdir(parents=True,exist_ok=True)
s,t,k=sp.symbols('s t k',real=True)
A=sp.Matrix([[0,1],[-2,-3]]);B=sp.Matrix([0,1]);C=sp.Matrix([[1,0]])
An=np.array(A).astype(float);Bn=np.array(B).astype(float);Cn=np.array(C).astype(float)
report={}
def check(name,condition,**details):
    assert condition,name
    report[name]={'passed':True,**details}
def tf(A,B,C):return sp.cancel((C*(s*sp.eye(A.rows)-A).inv()*B)[0])
def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=170,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)
def steady(A,b):return np.linalg.solve(-A,b)

G=1/(s*s+3*s+2)
check('01_transfer_initial',tf(A,B,C)==G and sp.cancel((C*(s*sp.eye(2)-A).inv()*sp.Matrix([1,0]))[0]-(2/(s+1)-1/(s+2)))==0)
y=2*sp.exp(-t)-sp.exp(-2*t)
check('01_ode',sp.simplify(sp.diff(y,t,2)+3*sp.diff(y,t)+2*y)==0 and y.subs(t,0)==1 and sp.diff(y,t).subs(t,0)==0)
Acl=A-B*k*C;poly=Acl.charpoly(s).as_expr()
# charpoly may normalize the symbol assumptions; use determinant for symbolic comparison.
check('02_determinant_identity',sp.cancel((s*sp.eye(2)-Acl).det()-(s*sp.eye(2)-A).det()*(1+k*G))==0)
K=sp.Matrix([[10,4]]);AK=A-B*K;L=sp.Matrix([8,4]);AO=A-L*C
check('02_feedback',set(AK.eigenvals())=={-3,-4},gain=[10,4])
P=sp.Matrix([[sp.Rational(5,4),sp.Rational(1,4)],[sp.Rational(1,4),sp.Rational(1,4)]])
x0=sp.Matrix([1,-sp.Rational(1,10)])
check('03_lyapunov',A.T*P+P*A==-sp.eye(2) and P.det()>0 and P[0,0]>0 and (x0.T*(A.T+A)*x0)[0]==sp.Rational(7,50),P=str(P))
Ah=sp.diag(A,1);Bh=sp.Matrix([0,1,0]);Ch=sp.Matrix([[1,0,0]])
check('04_hidden_mode',tf(Ah,Bh,Ch)==G and (sp.eye(3)-Ah).row_join(Bh).rank()==2 and (sp.eye(3)-Ah).col_join(Ch).rank()==2)
check('05_prefilter',-(C*AK.inv()*B)[0]==sp.Rational(1,12),N=12,unit_disturbance_offset=1/12)
KI=sp.Matrix([[9,3]]);AI=(A-B*KI).row_join(B*6).col_join((-C).row_join(sp.zeros(1,1)))
br=sp.Matrix([0,0,1]);bd=sp.Matrix([0,1,0]);ci=sp.Matrix([[1,0,0]])
check('05_integral',set(AI.eigenvals())=={-1,-2,-3} and -(ci*AI.inv()*br)[0]==1 and (ci*AI.inv()*bd)[0]==0 and A.row_join(B).col_join(C.row_join(sp.zeros(1,1))).det()!=0,gain=[9,3],integral_gain=6)
Aug=AK.row_join(B*K).col_join(sp.zeros(2,2).row_join(AO))
check('06_observer',set(AO.eigenvals())=={-5,-6} and set(Aug.eigenvals())=={-3,-4,-5,-6},observer_gain=[8,4])
check('07_sampling',abs(3/np.e-2)<1 and abs(np.exp(-3)-(3/np.e-2))>.5 and abs(3*np.exp(-np.log(3))-2+1)<1e-12,continuous_sample=float(np.exp(-3)),digital_sample=3/np.e-2,boundary_T=float(np.log(3)))
D=s*s+3*s+9
check('08_channels',tf(A-B*7*C,B*7,C)==7/D and tf(A-B*7*C,B,C)==1/D and tf(A-B*7*C,-B*7,C)==-7/D)
check('09_exercises',(s*sp.eye(2)-(A-B*sp.Matrix([[8,4]]))).det()==s*s+7*s+10 and -(C*(A-B*sp.Matrix([[8,4]])).inv()*B)[0]==sp.Rational(1,10))

fig,ax=plt.subplots(figsize=(9,5))
kk=np.linspace(0,20,1500);rr=np.array([np.roots([1,3,2+v]) for v in kk])
ax.axvspan(-6,0,color='#e7f2e9',label='连续稳定区域')
for j in range(2):ax.plot(rr[:,j].real,rr[:,j].imag,color=BLUE,lw=2,label='标量输出反馈根轨迹' if j==0 else None)
ax.scatter([-1,-2],[0,0],marker='o',s=55,facecolors='white',edgecolors=BLUE,label='开环极点')
ax.scatter([-3,-4],[0,0],marker='x',s=110,color=RED,label='状态反馈目标：−3、−4')
arrow(ax,(-1.5,2),(-1.5,3));arrow(ax,(-1.5,-2),(-1.5,-3))
ax.text(-1.3,3.5,'箭头：k 增大',fontsize=10)
ax.axvline(0,color='black',lw=.8);ax.axhline(0,color='gray',lw=.6)
ax.set(xlim=(-5,.5),ylim=(-5,5),xlabel='Re s',ylabel='Im s',title='同一对象：标量输出增益（0 至 20）与向量状态反馈');ax.legend(loc='upper left',fontsize=10);ax.grid(alpha=.2);save(fig,'01-feedback-poles')

fig,aa=plt.subplots(1,2,figsize=(11,4));Pn=np.array(P).astype(float);xn=np.array([1,-.1])
for ax,end,kind in zip(aa,[.15,4],['euclidean','energy']):
    tt=np.linspace(0,end,700);xx=np.array([expm(An*v)@xn for v in tt])
    yy=np.sum(xx*xx,axis=1) if kind=='euclidean' else np.einsum('bi,ij,bj->b',xx,Pn,xx)
    ax.plot(tt,yy,color=BLUE if kind=='euclidean' else RED,ls='-' if kind=='euclidean' else '--')
    ax.set(xlabel='t / s',ylabel='平方长度' if kind=='euclidean' else 'V',title='欧氏平方长度：初期可增长' if kind=='euclidean' else 'Lyapunov 函数：严格下降');ax.grid(alpha=.2)
fig.tight_layout();save(fig,'02-energy')

fig,ax=plt.subplots(figsize=(10,4.5));tt=np.linspace(0,15,900)
for M,vr,vd,c,label,color,style in [(np.array(AK).astype(float),np.array([0,12.]),np.array([0,1.]),np.array([1,0]),'状态反馈＋前馈',BLUE,'-'),(np.array(AI).astype(float),np.array([0,0,1.]),np.array([0,1.,0]),np.array([1,0,0]),'状态反馈＋积分',RED,'--')]:
    xr=steady(M,vr);xd=steady(M,vd);yy=[]
    for time in tt:
        x=xr-expm(M*time)@xr
        if time>=5:x=x+xd-expm(M*(time-5))@xd
        yy.append(c@x)
    ax.plot(tt,yy,color=color,ls=style,label=label)
    check('05_numeric_'+('prefilter' if len(c)==2 else 'integral'),abs(c@(xr+xd)-(1+1/12 if len(c)==2 else 1))<1e-10)
ax.axhline(1,color='gray',ls=':',label='单位参考');ax.axvline(5,color='gray',ls='--');ax.text(5.2,.35,'此时加入单位输入扰动')
ax.set(xlabel='t / s',ylabel='输出 y',title='名义跟踪与常值抗扰：前馈和积分的区别');ax.legend(loc='lower right');ax.grid(alpha=.2);save(fig,'03-tracking')

fig,ax=plt.subplots(figsize=(10,4));steps=np.arange(13)
ax.plot(steps,np.exp(-3)**steps,'o-',color=BLUE,label='连续反馈后采样：exp(−3)')
ax.plot(steps,(3/np.e-2)**steps,'s--',color=RED,label='采样更新并保持反馈：3/e − 2')
ax.axhline(0,color='gray',lw=.7);ax.set(xlabel='采样序号 k（T = 1 s）',ylabel='采样状态',title='初态为一：两种反馈实现的采样轨迹');ax.legend();ax.grid(alpha=.2);save(fig,'04-sampling')
(OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'PASS: {len(report)} checks, 4 figures; {OUT}')
