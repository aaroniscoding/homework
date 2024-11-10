import numpy as np
from scipy.sparse.linalg import eigs
from scipy.integrate import simpson
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from scipy.integrate import RK45
import matplotlib.pyplot as plt


# PART A
def shoot2(x, phi, n0, epsilon):
    return [phi[1], (x**2 - epsilon) * phi[0]]

tol = 1e-4  # define a tolerance level 
col = ['r', 'b', 'g', 'c', 'm']  # eigenfunc colors
L=4; n0 =0.5; xp = [-L, L] 
xspan =  np.arange(-L, L+0.1, 0.1)

A1 = np.zeros((len(xspan), 5))
A2 = []

beta_start = n0  # beginning value of beta
for modes in range(1, 6):  # begin mode loop
    beta = beta_start  # initial value of eigenvalue beta
    dbeta = n0/100  # default step size in beta
    for _ in range(1000):  # begin convergence loop for beta
        A = 1; y0 = [A, np.sqrt(L**2 - beta)]
        # y = odeint(shoot2, y0, xspan, args=(n0, beta)) 
        sol = solve_ivp(shoot2, [xspan[0], xspan[-1]], y0, t_eval= xspan, args=(n0, beta)) 
        # y = RK45(shoot2, xp[0], y0, xp[1], args=(beta,)) 
        y_sol = sol.y.T
        if abs(y_sol[-1, 1] + np.sqrt(L**2 - beta)*y_sol[-1, 0]) < tol:
        #if abs(y[-1, 0] - 0) < tol:  # check for convergence
            A2.append(beta)
            break  # get out of convergence loop

        if (-1) ** (modes + 1) * (y_sol[-1, 1] + np.sqrt(L**2 - beta)*y_sol[-1, 0]) > 0:
            beta += dbeta
        else:
            beta -= dbeta / 2
            dbeta /= 2

    beta_start = beta + 0.01  # after finding eigenvalue, pick new start
    norm = np.trapezoid(y_sol[:, 0] * y_sol[:, 0], xspan)  # calculate the normalization
    eigfunc_norm = y_sol[:, 0] / np.sqrt(norm)
    A1[:, modes-1] = np.abs(eigfunc_norm)
    #plt.plot(xspan, y_sol[:, 0] / np.sqrt(norm), col[modes - 1])  # plot modes

#plt.show()  # end mode loop



# PART B
col = ['r', 'b', 'g', 'c', 'm', 'k']
L=4
xspan = np.arange(-L, L+0.1, 0.1)
N = len(xspan)
dx = 0.1

A4 = []

A = np.zeros((N-2, N-2))

for j in range(N-2):
    A[j, j] = -2 - (dx**2*xspan[j+1]**2)
    if j < N-3:
        A[j, j+1] = 1
        A[j+1, j] = 1

A[0, 0] +=4 / 3 
A[0, 1] += -1 / 3 
A[-1, -1] +=4 / 3 
A[-1, -2] += -1 / 3 
A = A / (dx**2) 

eigvals, eigfunc = eigs(-A, k=5, which = 'SM')
A4 = eigvals
#print(A4)

temp = eigfunc.real
leftbound = []
rightbound = []
for i in range(0, 5):
    phi1 = 4/3*temp[0, i] - 1/3*temp[1, i]
    phi2 = 4/3*temp[-1, i] - 1/3*temp[-2, i]
    leftbound = np.append(leftbound, phi1)
    rightbound = np.append(rightbound, phi2)

leftbound = np.array(leftbound)
rightbound = np.array(rightbound)

V = np.vstack([leftbound, temp, rightbound])

V_new = np.zeros((N, 5))

for i in range(5):
    norm = np.trapezoid(V[:, i] * V[:, i], xspan)  # calculate the normalization
    eigfunc_norm = V[:, i] / np.sqrt(norm)
    V_new[:, i] = np.abs(eigfunc_norm)
    #plt.plot(xspan, abs(eigfunc_norm), col[i])

A3 = V_new


# PART C
def hw3_rhs(x, y, gamma, epsilon):
    return [y[1], (gamma*np.abs(y[0])**2 + x**2 - epsilon)*y[0]]

L=2
xspan = np.arange(-L, L+0.1, 0.1)
N = len(xspan)
col = ['r', 'b', 'g', 'c', 'm', 'k']
tol = 1e-6

A6 = np.zeros(2)
A5 = np.zeros((N, 2)) 

A8 = np.zeros(2)
A7 = np.zeros((N, 2))

epsilon_start = 0.1
A_start = 0.01

for currGamma in [0.05, -0.05]: 
    A_curr = A_start
    #epsilon_curr = 0.1
    epsilon_start = 0.1
    dE = 0.2
    dA= 0.01  
    for mode in range(0, 2): #gets first 2 eigenvalues and eigenfunctions 
        dA= 0.01   
        #dA = 0.01
        for currA in range(100):
            epsilon_curr = epsilon_start
            dE = 0.2
            for _ in range(100):
                y0 = [A_curr, np.sqrt(L**2 - epsilon_curr)*A_curr]
                #sol = solve_ivp(lambda xspan, y : hw3_rhs(xspan, y, currGamma, epsilon_curr), [xspan[0], xspan[-1]], y0, t_eval = xspan)
                sol = solve_ivp(hw3_rhs, [xspan[0], xspan[-1]], y0, t_eval= xspan, args=(currGamma, epsilon_curr))
                y_sol = sol.y.T
                x_sol = sol.t

                #bc = y_sol[-1, 1] + np.sqrt(L**2 - epsilon_curr)*y_sol[-1, 0] #ASK WHY THIS IS!!!!
                bc = y_sol[-1, 1] + np.sqrt(L**2 - epsilon_curr)*y_sol[-1, 0]
                if abs(bc) < tol:
                    break
                if (-1) ** (mode) * bc > 0:
                    epsilon_curr += dE
                else:
                    epsilon_curr = epsilon_curr - (dE / 2)
                    dE = dE/2
            area = np.trapezoid(y_sol[:, 0]**2, x_sol)

            if abs(area - 1) < tol:
                epsilon_start = epsilon_curr + 0.2
                break
            if area < 1:
                A_curr += dA
            else:
                A_curr -= dA/2
                dA /= 2
        if currGamma > 0:
            A6[mode] = epsilon_curr 
            #norm = simpson(y_sol[:, 0]**2, x=x_sol)
            #A5[:, mode] = (abs(y_sol[:, 0]) / np.sqrt(norm))
            A5[:, mode] = abs(y_sol[:, 0])            
            plt.plot(xspan, A5[:, mode], col[mode], label=f"n = {mode}") # plot abs value
        else:
            A7[:, mode] = abs(y_sol[:, 0])
            A8[mode] = epsilon_curr
            #plt.plot(xspan, A7[:, mode], col[mode], label=f"n = {mode}") # plot abs value
plt.show()

#norm = simpson(y_sol[:, 0]**2, x=x_sol)
#A5[:, mode] = (abs(y_sol[:, 0]) / np.sqrt(norm))


# PART D
def hw1_rhs_a(x, y, E):
    return [y[1], (x**2 - E)*y[0]]

L = 2
E=1
xspan = (-L, L)
y0 = [1, np.sqrt(L**2 - 1)]
tols = [10**-4, 10**-5, 10**-6, 10**-7, 10**-8, 10**-9, 10**-10]
dt45 = []; dt23 = []; dtradau = []; dtBDF = []

for tol in tols:
    options = {'rtol': tol, 'atol': tol}
    sol45 = solve_ivp(hw1_rhs_a, xspan, y0, method = 'RK45', args=(E,), **options)
    dt45.append(np.mean(np.diff(sol45.t)))
    sol23 = solve_ivp(hw1_rhs_a, xspan, y0, method = 'RK23', args=(E,), **options)
    dt23.append(np.mean(np.diff(sol23.t)))
    solradau = solve_ivp(hw1_rhs_a, xspan, y0, method = 'Radau', args=(E,), **options)
    dtradau.append(np.mean(np.diff(solradau.t)))
    solBDF = solve_ivp(hw1_rhs_a, xspan, y0, method = 'BDF', args=(E,), **options)
    dtBDF.append(np.mean(np.diff(solBDF.t)))

A9 = []
fit45 = np.polyfit(np.log(dt45), np.log(tols), 1)
A9.append(fit45[0])
fit23 = np.polyfit(np.log(dt23), np.log(tols), 1)
A9.append(fit23[0])
fitradau = np.polyfit(np.log(dtradau), np.log(tols), 1)
A9.append(fitradau[0])
fitBDF = np.polyfit(np.log(dtBDF), np.log(tols), 1)
A9.append(fitBDF[0])


# PART E
parta_eigvals = A2
parta_eigvecs = A1

partb_eigvals = A4
partb_eigvecs = A3

true_eigvals = [1, 3, 5, 7, 9]
L=4
xspan = np.arange(-L, L+0.1, 0.1)

hermite_funcs = [(np.pi)**(-1/4)*np.e**((-1/2) * xspan**2), 
                 np.sqrt(2)*np.pi**(-1/4)*xspan*np.e**((-1/2) * xspan**2),
                 (np.sqrt(2)*np.pi**(1/4))**(-1)*(2*xspan**2 - 1)*np.e**((-1/2) * xspan**2),
                 (np.sqrt(3)*np.pi**(1/4))**(-1)*(2*xspan**3 - 3*xspan)*np.e**((-1/2) * xspan**2),
                 (2*np.sqrt(6)*np.pi**(1/4))**(-1)*(4*xspan**4 - 12*xspan**2 + 3)*np.e**((-1/2) * xspan**2)]

hermite_eigfuncs = np.zeros((len(xspan), 5))

for i in range(0, 5):
    hermite_eigfuncs[:, i]= hermite_funcs[i].T

eigvec_error_parta = np.zeros(5)
eigvec_error_partb = np.zeros(5)
eigval_error_parta = np.zeros(5)
eigval_error_partb =np.zeros(5)

for i in range(0, 5):
    eigval_error_parta[i] = (abs(A2[i] - true_eigvals[i]) / true_eigvals[i])*100
    eigval_error_partb[i] = (abs(A4[i] - true_eigvals[i]) / true_eigvals[i])*100

    eigvec_error_parta[i] = simpson((abs(A1[:, i]) - abs(hermite_eigfuncs[:, i]))**2, x = xspan)
    eigvec_error_partb[i] = simpson((abs(A3[:, i]) - abs(hermite_eigfuncs[:, i]))**2, x = xspan)

A10 = eigvec_error_parta
A11 = eigval_error_parta
A12 = eigvec_error_partb
A13 = eigval_error_partb


print(A2)
