import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2
from scipy.integrate import odeint, solve_ivp
from scipy.sparse import spdiags
from scipy.linalg import kron




m = 8    # N value in x and y directions
n = m * m  # total size of matrix

e0 = np.zeros((n, 1))  # vector of zeros
e1 = np.ones((n, 1))   # vector of ones
e2 = np.copy(e1)    # copy the one vector
e4 = np.copy(e0)    # copy the zero vector

for j in range(1, m+1):
    e2[m*j-1] = 0  # overwrite every m^th value with zero
    e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
e3 = np.zeros_like(e2)
e3[1:n] = e2[0:n-1]
e3[0] = e2[n-1]

e5 = np.zeros_like(e4)
e5[1:n] = e4[0:n-1]
e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [e1.flatten(), e1.flatten(), e5.flatten(), 
             e2.flatten(), -4 * e1.flatten(), e3.flatten(), 
             e4.flatten(), e1.flatten(), e1.flatten()]
offsets = [-(n-m), -m, -m+1, -1, 0, 1, m-1, m, (n-m)]

A = spdiags(diagonals, offsets, n, n).toarray()
A = A / ((20/8)**2)

def sech(x):
    return 1 / np.cosh(x)

def tanh(x):
    return np.sinh(x) / np.cosh(x)


beta=1
D1 = 0.1
D2 = 0.1
m=1
tspan = np.arange(0, 4.5, 0.5)

Lx, Ly = 20, 20
nx, ny = 64, 64
N = nx * ny

x2 = np.linspace(-Lx/2, Lx/2, nx + 1)
x = x2[:nx]
y2 = np.linspace(-Ly/2, Ly/2, ny + 1)
y = y2[:ny]
X, Y = np.meshgrid(x, y)
u=tanh(np.sqrt(X**2+Y**2)) * np.cos(m*np.angle(X+1j*Y) - (np.sqrt(X**2+Y**2))) 
v=tanh(np.sqrt(X**2+Y**2)) * np.sin(m*np.angle(X+1j*Y) - (np.sqrt(X**2+Y**2)))

kx = (2 * np.pi / Lx) * np.concatenate((np.arange(0, nx/2), np.arange(-nx/2, 0)))
kx[0] = 1e-6
ky = (2 * np.pi / Ly) * np.concatenate((np.arange(0, ny/2), np.arange(-ny/2, 0)))
ky[0] = 1e-6
KX, KY = np.meshgrid(kx, ky)
K = KX**2 + KY**2

def uv_rhs(t, uv, beta, nx, ny, D1, D2):
    u, v = np.split(uv, 2, axis=0)
    ut = u.reshape(nx, ny)
    vt = v.reshape(nx, ny)
    u = ifft2(ut)
    v = ifft2(vt)
    A2 = u**2 + v**2
    u_rhs = fft2((1-A2)*u - (-beta*A2)*v) - D1*K*ut
    v_rhs = fft2((-beta*A2)*u + (1-A2)*v) - D2*K*vt
    return np.hstack([u_rhs.flatten(), v_rhs.flatten()])
   

uv0 = np.hstack([fft2(u).flatten(), fft2(v).flatten()])
sol = solve_ivp(uv_rhs, [tspan[0], tspan[-1]], uv0, method='RK45', t_eval=tspan, args=(beta, nx, ny, D1, D2))
A1 = sol.y

print(A1.shape)
print(A1)

A1_final = np.zeros((9, 8192))
for j, t in enumerate(tspan):
    u, v = A1.T[j,:N].reshape((nx, ny)), A1.T[j, N:].reshape((nx, ny))
    u = np.real(ifft2(u))
    v = np.real(ifft2(v))
    uv_curr = np.concatenate((u.flatten(), v.flatten()))
    A1_final[j,:] = uv_curr
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, u, shading='interp')
    plt.title(f'Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()





# PART 2

def cheb(N):
	if N==0: 
		D = 0.; x = 1.
	else:
		n = np.arange(0,N+1)
		x = np.cos(np.pi*n/N).reshape(N+1,1) 
		c = (np.hstack(( [2.], np.ones(N-1), [2.]))*(-1)**n).reshape(N+1,1)
		X = np.tile(x,(1,N+1))
		dX = X - X.T
		D = np.dot(c,1./c.T)/(dX+np.eye(N+1))
		D -= np.diag(np.sum(D.T,axis=0))
	return D, x.reshape(N+1)

N=30
D, x = cheb(N)
x = x * 10
D[N, :] = 0
D[0, :] = 0
#print(x.shape)
Dsquared = np.dot(D, D) / 100
y = x

I = np.eye(len(Dsquared))
L = kron(I, Dsquared) + kron(Dsquared, I)  # 2D Laplacian
print(L.shape)
X, Y = np.meshgrid(x, y)

u=tanh(np.sqrt(X**2+Y**2)) * np.cos(m*np.angle(X+1j*Y) - (np.sqrt(X**2+Y**2))) 
v=tanh(np.sqrt(X**2+Y**2)) * np.sin(m*np.angle(X+1j*Y) - (np.sqrt(X**2+Y**2)))

def cheb_rhs(t, uv, u, v, beta, D1, D2, L):
    u = uv[0:(N+1)**2]
    v = uv[(N+1)**2:]
    # u, v = np.split(uv, 2, axis=0)
    A2 = u**2 + v**2
    u_rhs = D1 * (np.dot(L, u)) + ((1-A2)*u - (-beta*A2)*v)
    v_rhs = D2 * (np.dot(L, v)) + ((-beta*A2)*u + (1-A2)*v) 
    return np.hstack([u_rhs.flatten(), v_rhs.flatten()])

uv0 = np.hstack([u.flatten(), v.flatten()])
sol = solve_ivp(cheb_rhs, [tspan[0], tspan[-1]], uv0, method='RK45', t_eval=tspan, args=(u, v, beta, D1, D2, L))
A2 = sol.y

print(A2.shape)
print(A2)

for j, t in enumerate(tspan):
    u, v = A2.T[j,:961].reshape((N+1, N+1)), A2.T[j, 961:].reshape((N+1, N+1))
    uv_curr = np.hstack([u, v])
    #A1_final[j, :] = uv_curr
    plt.subplot(3, 3, j + 1)
    plt.pcolor(x, y, u, shading='interp')
    plt.title(f'Time: {t}')
    plt.colorbar()
plt.tight_layout()
plt.show()