import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import spdiags

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

A1 = spdiags(diagonals, offsets, n, n).toarray()
A1 = A1 / ((20/8)**2)
# Plot matrix structure
plt.figure(5)
plt.spy(A1)
plt.title('Matrix Structure')
plt.show()


# Matrix B

# Grid size (8x8) = 64 elements in total
m = 8
n = m * m  # 64 elements

# Create vectors for the diagonals
main_diag = np.zeros(n)  # Main diagonal: 0 (no self-contribution)
upper_diag = np.ones(n)  # Upper diagonal: +1 (i, j+1)
lower_diag = np.ones(n)  # Lower diagonal: -1 (i, j-1)
upper2 = np.copy(upper_diag)
lower2 = np.copy(lower_diag)

for j in range(1, m+1):
    main_diag[m*j-1] = 0  # overwrite every m^th value with zero
    #e4[m*j-1] = 1  # overwirte every m^th value with one

# Shift to correct positions
# e3 = np.zeros_like(e2)
# e3[1:n] = e2[0:n-1]
# e3[0] = e2[n-1]

# e5 = np.zeros_like(e4)
# e5[1:n] = e4[0:n-1]
# e5[0] = e4[n-1]

# Place diagonal elements
diagonals = [lower2, -1*lower_diag, main_diag, upper_diag, -1*upper2]
offsets = [-(n-m), -m, 0, m, (n-m)]

A2 = spdiags(diagonals, offsets, n, n).toarray()
A2 = A2 / (40/8)
# Plot matrix structure
plt.figure(5)
plt.spy(A2)
plt.title('Matrix Structure')
plt.show()


#Matrix C

# Grid size (8x8) = 64 elements in total
m = 8
n = m * m  # 64 elements

# Create vectors for the diagonals
main_diag = np.zeros(n)  # Main diagonal: 0 (no self-contribution)
upper_diag = np.ones(n)  # Upper diagonal: +1 (i, j+1)
lower_diag = np.ones(n)  # Lower diagonal: -1 (i, j-1)
upper2 = np.zeros(n)
lower2 = np.copy(main_diag)

for j in range(1, m+1):
    lower_diag[m*j-1] = 0  # overwrite every m^th value with zero
    #lower_diag[m*j-1] = 0
    upper2[m*j-1] = 1
    # lower2[m*j-1] = -1
    #e4[m*j-1] = 1  # overwirte every m^th value with one

#Shift to correct positions
lower2 = np.zeros_like(upper2)
lower2[1:n] = upper2[0:n-1]
lower2[0] = upper2[n-1]

upper_diag = np.zeros_like(lower_diag)
upper_diag[1:n] = lower_diag[0:n-1]
upper_diag[0] = lower_diag[n-1]

# Place diagonal elements
diagonals = [lower2, -1*lower_diag, main_diag, upper_diag, -1*upper2]
offsets = [-m+1, -1, 0, 1, m-1]

A3 = spdiags(diagonals, offsets, n, n).toarray()
A3 = A3 / (40/8)
# Plot matrix structure
plt.figure(5)
plt.spy(A3)
plt.title('Matrix Structure')
plt.show()










