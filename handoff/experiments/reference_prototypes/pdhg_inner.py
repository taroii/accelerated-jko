"""Prototype: one restricted proximal step  min_psi  -∫log det(I+∇²ψ) + ∫V(x+∇ψ) + (1/2γ)∫|x+∇ψ-Y|²
on the periodic torus [0,1)^2 (uniform ν), V(u)=½(u-μ)ᵀQ⁻¹(u-μ) convex on R².
PDHG with dual blocks: Λ (entropy, pointwise, single channel) and m_j (quadratic, dyadic Fourier bands, per-band steps)."""
import numpy as np, sys, time
n=int(sys.argv[1]); gamma=float(sys.argv[2]); amp=float(sys.argv[3]); nbands=int(sys.argv[4]); iters=int(sys.argv[5]); seed=int(sys.argv[6]) if len(sys.argv)>6 else 0
rng=np.random.default_rng(seed)
h=1.0/n; xs=(np.arange(n)+0.5)*h; X1,X2=np.meshgrid(xs,xs,indexing='ij'); Xg=np.stack([X1,X2],-1)
k=2*np.pi*np.fft.fftfreq(n,d=h); K1,K2=np.meshgrid(k,k,indexing='ij'); Kabs2=K1**2+K2**2
fft=np.fft.fft2; ifft=lambda a: np.real(np.fft.ifft2(a))
def grad(psi): P=fft(psi); return np.stack([ifft(1j*K1*P),ifft(1j*K2*P)],-1)
def gradT(m): return ifft(-1j*K1*fft(m[...,0])-1j*K2*fft(m[...,1]))      # adjoint of grad (= -div)
def hess(psi): P=fft(psi); return np.stack([ifft(-K1*K1*P),np.sqrt(2)*ifft(-K1*K2*P),ifft(-K2*K2*P)],-1)   # (ψxx, √2ψxy, ψyy)
def hessT(L): return ifft(-K1*K1*fft(L[...,0])-np.sqrt(2)*K1*K2*fft(L[...,1])-K2*K2*fft(L[...,2]))
# target and Y
mu=np.array([0.5,0.5]); Q=np.diag([0.02,0.05]); Qi=np.linalg.inv(Q)
Yf=Xg+0.08*grad(np.sin(2*np.pi*X1)*np.cos(4*np.pi*X2)+0.5*np.cos(2*np.pi*X2))+0.03*rng.normal(size=Xg.shape)*0  # given field Y
B=Qi+np.eye(2)/gamma; Binv=np.linalg.inv(B)
ustar=np.einsum('ij,...j->...i',Binv,(Qi@mu)[None,None,:]+Yf/gamma); a=ustar-Xg            # f2(m)=½(m-a)ᵀB(m-a)+c
# dyadic bands on |k|: band j = {2^-(j+1) kmax < |k| <= 2^-j kmax}, last band includes k=0
kmax=np.sqrt(Kabs2.max()); bands=[]
for j in range(nbands):
    lo=0 if j==nbands-1 else kmax*2.0**-(j+1); hi=kmax*2.0**-j
    bands.append(((Kabs2>lo**2)&(Kabs2<=hi**2)) if j<nbands-1 else (Kabs2<=hi**2))
assert sum(b.astype(int) for b in bands).min()==1 and sum(b.astype(int) for b in bands).max()==1
def proj(m,b): return np.stack([ifft(fft(m[...,c])*b) for c in range(2)],-1)
# step sizes: dual σ_Λ=σ0 for entropy; σ_j = σ0*amp^j for band j (j=0 top octave); τ from exact spectrum
sig0=float(sys.argv[7]) if len(sys.argv)>7 else 1.0; ratio=float(sys.argv[9]) if len(sys.argv)>9 else 1.0; sigb=[sig0*ratio*amp**j for j in range(nbands)]
spec=sig0*Kabs2**2+sum(s*b*Kabs2 for s,b in zip(sigb,bands))
PRECOND=(len(sys.argv)>8 and sys.argv[8]=='fourier')
if PRECOND:
    Tk=0.99/np.where(spec>0,spec,spec[spec>0].min()); tau=None   # primal Fourier preconditioner
else:
    tau=0.99/spec.max()
# proxes
def prox_entropy_dual(L,s):   # prox_{s h*}(L) = L - s prox_{h/s}(L/s), h(H)=-log det(I+H), H sym 2x2 via (a,√2 b,c)
    M=L/s; A=M[...,0]; Bm=M[...,1]/np.sqrt(2); C=M[...,2]
    tr=A+C; det=A*C-Bm*Bm; disc=np.sqrt(np.maximum(tr*tr/4-det,0)); e1=tr/2+disc; e2=tr/2-disc
    t=1.0/s
    f=lambda e: ((e+1)+np.sqrt((e+1)**2+4*t))/2-1
    p1=f(e1); p2=f(e2)
    # eigenvectors of [[A,B],[B,C]]
    th=0.5*np.arctan2(2*Bm,A-C); c=np.cos(th); sn=np.sin(th)
    Pa=p1*c*c+p2*sn*sn; Pb=(p1-p2)*c*sn; Pc=p1*sn*sn+p2*c*c
    P=np.stack([Pa,np.sqrt(2)*Pb,Pc],-1)
    return L-s*P
def prox_quad_dual(v,s,b):    # band j: prox_{s f_j*}(v) = (I+sB⁻¹)⁻¹(v - s P_j a)
    Mi=np.linalg.inv(np.eye(2)+s*Binv); return np.einsum('ij,...j->...i',Mi,v-s*proj(a,b))
def objective(psi):
    H=hess(psi); A=1+H[...,0]; Bm=H[...,1]/np.sqrt(2); C=1+H[...,2]; det=A*C-Bm*Bm
    if det.min()<=0 or (A+C).min()<=0: return np.inf
    u=Xg+grad(psi); d=u-mu
    return np.mean(-np.log(det)+0.5*np.einsum('...i,ij,...j->...',d,Qi,d)+0.5*np.sum((u-Yf)**2,-1)/gamma)
# run PDHG
psi=np.zeros((n,n)); psib=psi.copy(); Lam=np.zeros((n,n,3)); ms=[np.zeros((n,n,2)) for _ in bands]
hist=[]; t0=time.time()
for it in range(iters):
    Hb=hess(psib); Gb=grad(psib)
    Lam=prox_entropy_dual(Lam+sig0*Hb,sig0)
    for j,(b,s) in enumerate(zip(bands,sigb)): ms[j]=prox_quad_dual(ms[j]+s*proj(Gb,b),s,b)
    mtot=sum(ms); g=hessT(Lam)+gradT(mtot)
    psin=psi-(ifft(Tk*fft(g)) if PRECOND else tau*g); psin-=psin.mean()
    psib=2*psin-psi; psi=psin
    if it%10==0 or it==iters-1: hist.append((it,objective(psi)))
np.save(f'/tmp/claude-0/inner/psi_n{n}_g{gamma}_amp{amp}_b{nbands}.npy',psi)
print(f"n={n} gamma={gamma} bands={nbands} amp={amp}: tau={tau}, sigma_bands={[f'{s:.2g}' for s in sigb]}, {time.time()-t0:.1f}s")
ref=5.2993728099
its=[i for i,o in hist]; gaps=[o-ref for i,o in hist]
def first(th): return next((i for i,g in zip(its,gaps) if g<th), None)
print(f"   iterations to gap<1e-2: {first(1e-2)}, <1e-4: {first(1e-4)}, <1e-6: {first(1e-6)}, <1e-8: {first(1e-8)};  final gap {gaps[-1]:.2e}")
