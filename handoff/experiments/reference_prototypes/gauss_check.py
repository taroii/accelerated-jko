import numpy as np
from scipy.linalg import sqrtm, eigh
from scipy.optimize import minimize
np.set_printoptions(precision=4, suppress=True)
d=2
def sym(M): return (M+M.T)/2
def psqrt(M):
    w,V=eigh(sym(M)); return (V*np.sqrt(np.maximum(w,0)))@V.T
def ot(S0,S1):   # OT map matrix N(0,S0)->N(0,S1)
    r=psqrt(S0); ri=np.linalg.inv(r); return sym(ri@psqrt(r@S1@r)@ri)
def W2sq(S0,S1):
    r=psqrt(S0); return np.trace(S0)+np.trace(S1)-2*np.trace(psqrt(r@S1@r))
def KL(S):       # KL(N(0,S)||N(0,I))
    return 0.5*(np.trace(S)-d-np.log(np.linalg.det(S)))
def jko(Sy,g):   # exact JKO step for KL(.|N(0,I)) from N(0,Sy)
    w,V=eigh(sym(Sy)); sy=np.sqrt(w)
    u=(sy+np.sqrt(sy**2+4*g*(1+g)))/(2*(1+g)); return (V*u**2)@V.T

# --- 0. the 3-SPD counterexample from the search
A=np.array([[3,-3],[-3,4.]]);B=np.array([[1,1],[1,2.]]);C=np.array([[1,-1],[-1,4.]])
print("SPD eigs A,B,C:",np.linalg.eigvalsh(A),np.linalg.eigvalsh(B),np.linalg.eigvalsh(C))
print("eig C(AB)^-1:",np.linalg.eigvals(C@np.linalg.inv(A@B)))

# --- 1. ORIGINAL algorithm, exact, centered Gaussians, q=N(0,I), rho*=N(0,I), G*=0
def original(S0,g,T):
    Sz=S0.copy(); Sx=S0.copy(); B0=9*W2sq(S0,np.eye(d))/(2*g)
    worst=-1; lyap_viol=0
    for t in range(T):
        a=3/(t+3)
        X=ot(Sz,Sx); Y=(1-a)*X+a*np.eye(d); Sy=Y@Sz@Y.T
        Sx1=jko(Sy,g); Tt=ot(Sy,Sx1); X1=Tt@Y
        Z=np.eye(d)+(X1-Y)/a; Sz1=Z@Sz@Z.T
        # one-step lemma check
        lhs=a*a/(2*g)*W2sq(Sz1,np.eye(d))+KL(Sx1); rhs=a*a/(2*g)*W2sq(Sz,np.eye(d))+(1-a)*KL(Sx)
        if lhs>rhs*(1+1e-9)+1e-14: lyap_viol+=1
        Sz,Sx=Sz1,Sx1
        ratio=KL(Sx)/(B0/((t+3)**2)); worst=max(worst,ratio)
    return worst,lyap_viol

rng=np.random.default_rng(0)
worst_all=0; nviol=0; found=None
for trial in range(400):
    M=rng.normal(size=(d,d)); S0=M@M.T+rng.uniform(0.01,0.5)*np.eye(d)
    S0*=np.exp(rng.uniform(-3,2))
    g=np.exp(rng.uniform(-3,2))
    w,lv=original(S0,g,60)
    nviol+=lv>0
    if w>worst_all: worst_all=w; found=(S0,g,lv)
print(f"ORIGINAL: max of gap/bound over 400 random 2D instances = {worst_all:.4f}; instances with a one-step-lemma violation: {nviol}/400")

# --- 2. REPAIR: fixed base nu=x0, linearized prox over symmetric PSD maps (Gaussian case)
def lin_prox(Yv,S0,g,Sinit):
    L0=np.linalg.cholesky(Sinit+1e-9*np.eye(d))
    def f(p):
        L=np.array([[p[0],0],[p[1],p[2]]]); S=L@L.T
        Sx=S@S0@S; D=S-Yv
        return KL(Sx)+np.trace(D@S0@D.T)/(2*g)
    p0=np.array([L0[0,0],L0[1,0],L0[1,1]])
    r=minimize(f,p0,method="BFGS",options={"gtol":1e-12,"maxiter":5000})
    L=np.array([[r.x[0],0],[r.x[1],r.x[2]]]); return L@L.T
def repaired(S0,g,T):
    Sstar=ot(S0,np.eye(d)); B0=9*np.trace((np.eye(d)-Sstar)@S0@(np.eye(d)-Sstar))/(2*g)  # = 9 W2^2(x0,rho*)/(2g)
    Xs=np.eye(d); Zs=np.eye(d); worst=-1
    for t in range(T):
        a=3/(t+3); Yv=(1-a)*Xs+a*Zs
        X1=lin_prox(Yv,S0,g,Xs if t else np.eye(d))
        Zs=Zs+(X1-Yv)/a; Xs=X1
        worst=max(worst,KL(Xs@S0@Xs)/(B0/(t+3)**2))
    return worst
w2=0
for trial in range(40):
    M=rng.normal(size=(d,d)); S0=M@M.T+rng.uniform(0.05,0.5)*np.eye(d); S0*=np.exp(rng.uniform(-2,1.5)); g=np.exp(rng.uniform(-2,1.5))
    w2=max(w2,repaired(S0,g,40))
print(f"REPAIR (fixed base): max gap/bound over 40 instances = {w2:.4f}")
