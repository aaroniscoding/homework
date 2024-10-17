
import numpy as np
from scipy.integrate import odeint
from scipy.integrate import RK45
import matplotlib.pyplot as plt

def shoot2(y, x, n0, epsilon):
    return [y[1], (x**2 - epsilon) * y[0]]

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm', 'k']  # eigenfunc colors
L=4; n0 =0.5; xp = [-L, L] 
xspan =  np.linspace(xp[0], xp[1], 81)

A1 = np.zeros((len(xspan), 5))
A2 = []

beta_start = n0  # beginning value of beta
for modes in range(1, 6):  # begin mode loop
    beta = beta_start  # initial value of eigenvalue beta
    dbeta = n0/100  # default step size in beta
    for _ in range(1000):  # begin convergence loop for beta
        A = 1; y0 = [A, np.sqrt(L**2 - n0)]
        y = odeint(shoot2, y0, xspan, args=(n0, beta)) 
        # y = RK45(shoot2, xp[0], y0, xp[1], args=(beta,)) 

        if abs(y[-1, 1] + np.sqrt(L**2 - beta)*y[-1, 0]) < tol:
        #if abs(y[-1, 0] - 0) < tol:  # check for convergence
            A2.append(beta)
            print(beta)  # write out eigenvalue
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * (y[-1, 1] + np.sqrt(L**2 - beta)*y[-1, 0]) > 0:
            beta += dbeta
        else:
            beta -= dbeta / 2
            dbeta /= 2

    beta_start = beta + 0.01  # after finding eigenvalue, pick new start
    norm = np.trapezoid(y[:, 0] * y[:, 0], xspan)  # calculate the normalization
    eigfunc_norm = y[:, 0] / np.sqrt(norm)
    A1[:, modes-1] = np.abs(eigfunc_norm)
    plt.plot(xspan, y[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes

print(A2)
plt.show()  # end mode loop


 