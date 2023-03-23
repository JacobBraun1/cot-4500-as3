import numpy as np

#question 1
#this question works to create a eueler function and then insert
#the needed numbers from the problem
def f(t, y):
    return t - y**2
def euler(function, y, t0, tf, n):
    solution = [(t0, y)]
    h = (tf - t0) / n
    for i in range(n):
        y = y + h * function(t0, y)
        t0 = t0 + h
        solution.append((t0, y))
    return solution

#output 1
solution = euler(f, 1, 0, 2, 10)
t,y=solution[-1]
print(y)
print()


#question 2
#this function uses a runge function to find the answer and then 
#inputs the needed numbers 
def runge(f, y0, t0, tf, n):
    solution = [(t0, y0)]
    j = (tf - t0) / n
    for i in range(n):
        l1 = j * f(t0, y0)
        l2 = j * f(t0 + j/2, y0 + l1/2)
        l3 = j * f(t0 + j/2, y0 + l2/2)
        l4 = j * f(t0 + j, y0 + l3)
        y0 = y0 + (l1 + 2*l2 + 2*l3 + l4) / 6
        t0 = t0 + j
        solution.append((t0, y0))
    return solution

#output 2
solution = runge(f, 1, 0, 2, 10)
t,y=solution[-1]
print(y)
print()

#question 3
A = np.array([[2, -1, 1, 6], [1, 3, 1, 0], [-1, 5, 4, -3]])

for i in range(A.shape[0]):
    pivot_row = i
    while A[pivot_row, i] == 0:
        pivot_row += 1
    
    A[[i, pivot_row]] = A[[pivot_row, i]]
    
    for j in range(i+1, A.shape[0]):
        factor = A[j, i] / A[i, i]
        A[j, i:] = A[j, i:] - factor * A[i, i:]

x = np.zeros(A.shape[0])
for i in range(A.shape[0]-1, -1, -1):
    x[i] = (A[i, -1] - np.dot(A[i, i:-1], x[i:])) / A[i, i]

#output 3
ans=[]
print("[",end =" ")
for i in range(len(x)):
    print(int(x[i]),end =" ")
print("]",end="\n")
print()

#question 4
A = np.array([[1.0, 1.0, 0.0, 3.0], [2.0, 1.0, -1.0, 1.0], [3.0, -1.0, -1.0, 2.0], [-1.0, 2.0, 3.0, -1.0]])
L = np.eye(A.shape[0])
U = np.zeros_like(A)

for i in range(A.shape[0]):
    for j in range(i, A.shape[0]):
        U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])
    
    for j in range(i+1, A.shape[0]):
        L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]


#output 4
determinite_A = np.prod(np.diag(U))-0.00000000001
print(determinite_A)
print()
print(L)
print()
print(U)
print()

#question 5
A = np.array([[9, 0, 5, 2, 1], [3, 9, 1, 2, 1], [0, 1, 7, 2, 3],[4, 2, 3, 12, 2],[3, 2, 4, 0, 8]])
diagnol_abs = np.abs(np.diag(A))
diagnol_abs_2 = np.abs(A) - np.diag(diagnol_abs)
diagnol = np.all(diagnol_abs >= np.sum(diagnol_abs_2, axis=1))

#output 5
print("True") if diagnol else print("False")
print()

#question 6
A = np.array([[2, 2, 1],[2, 3, 0],[1, 0, 2]])
eigvals = np.linalg.eigvals(A)

#output 6
print("True") if np.all(eigvals > 0) else print("False")