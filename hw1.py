import numpy as np
import math

#Question 1
A3 = []

# Method 1: Newton Raphson 
x_curr = -1.6
A1 = []

for i in range(0, 99):
    fx = x_curr*np.sin(3*x_curr) - np.exp(x_curr)
    #print(fx)
    #print(x_curr)
    A1.append(x_curr)

    if (abs(fx) < 1e-6):
        print(f"Iterations: {i+1}")
        x_next = x_curr - ((x_curr*np.sin(3*x_curr) - np.exp(x_curr))/(np.sin(3*x_curr) + 3*x_curr*np.cos(3*x_curr) - np.exp(x_curr)))
        A1.append(x_next)
        A3.append(i+1)
        break
    else:
        x_next = x_curr - ((x_curr*np.sin(3*x_curr) - np.exp(x_curr))/(np.sin(3*x_curr) + 3*x_curr*np.cos(3*x_curr) - np.exp(x_curr)))
        x_curr = x_next
print(A1)
print(len(A1))
np.save('A1.npy', A1)

#Method 2: Bisection
xl = -0.7
xr = -0.4
A2 = []

for i in range(0, 99):
    xc = (xl + xr)/2
    #print(xc)
    A2.append(xc)
    fc = xc*np.sin(3*xc) - np.exp(xc)

    if (fc > 0):
        xl = xc
    else:
        xr = xc
    if (abs(fc) < 10**(-6)):
        #print(xc)
        print(f"Iterations: {i+1}")
        A3.append(i+1)
        #print(fc)
        break   
np.save('A2.npy', A2)

np.save('A3.npy', A3)
print(A3)

# Question 2
A = np.array([[1, 2], [-1, 1]])
B = np.array([[2, 0], [0, 2]])
C = np.array([[2, 0, -3], [0, 0, -1]])
D = np.array([[1, 2], [2, 3], [-1, 0]])
x = np.array([[1], [0]])
y = np.array([[0], [1]])
z = np.array([[1], [2], [-1]])

A4 = A + B
print(A4)
A5 = (3*x - 4*y).flatten()
print(A5)
A6 = (np.dot(A, x)).flatten()
print(A6)
A7 = (np.dot(B, x-y)).flatten()
print(A7)
A8 = (np.dot(D, x)).flatten()
print(A8)   
A9 = (np.dot(D, y) + z).flatten()
print(A9)
A10 = np.dot(A, B)
print(A10)
A11 = np.dot(B, C)
print(A11)
A12 = np.dot(C, D)
print(A12)

np.save('A4.npy', A4)
np.save('A5.npy', A5)
np.save('A6.npy', A6)
np.save('A7.npy', A7)
np.save('A8.npy', A8)
np.save('A9.npy', A9)
np.save('A10.npy', A10)
np.save('A11.npy', A11)
np.save('A12.npy', A12)
