import numpy as np
from scipy.fft import dctn, idctn
n=12; I=np.eye(n)
# 1-D operators on a cell-centered grid with Neumann conditions: D = forward difference (n-1 x n), L = D^T D (Neumann Laplacian, n x n)
D=np.zeros((n-1,n)); 
for i in range(n-1): D[i,i]=-1; D[i,i+1]=1
L=D.T@D
# 2-D: gradient blocks Dx = D ⊗ I, Dy = I ⊗ D ; Hessian blocks Dxx = L ⊗ I (sym second diff), Dyy = I ⊗ L, Dxy = D ⊗ D ((n-1)^2 x n^2)
Dx=np.kron(D,I); Dy=np.kron(I,D); Dxx=np.kron(L,I); Dyy=np.kron(I,L); Dxy=np.kron(D,D)
Kgrad=np.vstack([Dx,Dy]); Khess=np.vstack([Dxx,np.sqrt(2)*Dxy,Dyy])
def is_dct_diag(A):   # is A diagonalized by DCT-II on the n x n grid?
    V=np.zeros((n*n,n*n))
    for p in range(n):
        for q in range(n):
            e=np.zeros((n,n)); e[p,q]=1; V[:,p*n+q]=idctn(e,norm='ortho').ravel()
    M=V.T@A@V; off=np.abs(M-np.diag(np.diag(M))).max(); return off, np.diag(M)
for name,K in [('grad^T grad',Kgrad.T@Kgrad),('hess^T hess',Khess.T@Khess),('sum',Kgrad.T@Kgrad+3*Khess.T@Khess)]:
    off,dg=is_dct_diag(K); print(f"{name:12s}: max off-diagonal in DCT-II basis = {off:.2e}")
# predicted multipliers: m(k)=2 sin(pi k/(2n)) per axis; grad^T grad -> m(k)^2+m(l)^2 ; hess^T hess -> (m(k)^2+m(l)^2)^2
m=2*np.sin(np.pi*np.arange(n)/(2*n)); MK,ML=np.meshgrid(m,m,indexing='ij')
pred_g=(MK**2+ML**2).ravel(); pred_h=((MK**2+ML**2)**2).ravel()
off,dg=is_dct_diag(Kgrad.T@Kgrad); print("grad: max |diag - predicted| =",np.abs(np.sort(dg)-np.sort(pred_g)).max())
off,dg=is_dct_diag(Khess.T@Khess); print("hess: max |diag - predicted| =",np.abs(np.sort(dg)-np.sort(pred_h)).max())
