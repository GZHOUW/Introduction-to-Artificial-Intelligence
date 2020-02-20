def shape(M):
    return len(M),len(M[0])

def matxRound(M, decPts=4):
    for row in M:
        for element in row:   
            row[row.index(element)] = round(element,decPts)
    pass

def transpose(M):
    N = [*zip(*M)]
    result = []
    for i in range(len(N)):
        result.append([])
    for i in range(len(N)):
        for element in N[i]:
            result[i].append(element)
    return result

def matxMultiply(A, B):
    if len(A[0]) == len(B):
        result = [[0 for x in range(len(B[0]))] for y in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result
    else:
        raise ValueError

def augmentMatrix(A, b):
    Ab = []
    for i in range(len(A)):
        Ab.append(A[i]+b[i])
    return Ab

def swapRows(M, r1, r2):
    M[r1],M[r2] = M[r2],M[r1]
    pass

def scaleRow(M, r, scale):
    if scale == 0:
        raise ValueError
    else:
        for i in range(len(M[r])):
            M[r][i] = scale * M[r][i]
    pass


def addScaledRow(M, r1, r2, scale):
    for i in range(len(M[r2])):
        M[r1][i] = M[r1][i] + M[r2][i] * scale
    pass

from helper import *
A = generateMatrix(3,seed,singular=False)
b = np.ones(shape=(3,1),dtype=int)
Ab = augmentMatrix(A.tolist(),b.tolist())
printInMatrixFormat(Ab,padding=3,truncating=0)

A = generateMatrix(3,seed,singular=True)
b = np.ones(shape=(3,1),dtype=int)
Ab = augmentMatrix(A.tolist(),b.tolist()) 
printInMatrixFormat(Ab,padding=3,truncating=0)

""" Apply Gaussian Jordan Elimination to solve Ax = b
        A: matrix 
        b: column vector
        decPts: round to 4 decimal places
        epsilon: determine whether equal to 0, using 1.0e-16
        
    return x so that Ax = b 
    if A and b have different heights, return None
    
"""
def gj_Solve(A, b, decPts=4, epsilon=1.0e-16):
    if len(A) != len(b):
        return None
    else:
        Ab = augmentMatrix(A, b)
        for i in range(len(Ab)):
            columnAbs = [abs(row[i]) for row in Ab]    # the current column absolute value
            absMax = max(columnAbs)
            underDiagonal = []
            for a in range(i,len(b)):
                underDiagonal.append(columnAbs[a])
            if len(underDiagonal)> 1:
                columnMax = max(underDiagonal) # maximun value in current column in/under main diagonal
                diagIndex = underDiagonal.index(columnMax)
                maxIndex = i + diagIndex
            elif len(underDiagonal)== 1 and underDiagonal[0] > 1.0e-10:
                maxIndex = len(columnAbs)-1
            elif len(underDiagonal)== 1 and underDiagonal[0] < 1.0e-10:
                return None
            if absMax <= epsilon:
                return None
            else:
                swapRows(Ab, i, maxIndex)
                mainDiagonal = Ab[i][i]
                scale = 1.0/mainDiagonal
                scaleRow(Ab, i, scale)
                mainDiagonal = Ab[i][i]
                currentColumn = [row[i] for row in Ab]    # the current column
                for j in range(len(currentColumn)):
                    if j!=i and currentColumn[j]!=0:
                        scale2 = -currentColumn[j]
                        addScaledRow(Ab, j, i, scale2)
        matxRound(Ab)
        result = []
        for element in Ab:
            result.append([element[-1]])
        return result


%matplotlib notebook
from helper import *

X,Y = generatePoints2D(seed)
vs_scatter_2d(X, Y)

def calculateMSE2D(X,Y,m,b):
    n = len(X)
    z = []
    for i in range(len(X)):
        z.append((Y[i]-m1*X[i]-b1)**2)
    MSE= sum(z)/n
    return MSE

print(calculateMSE2D(X,Y,m1,b1))

def linearRegression2D(X,Y):
    X1 = []
    for element in X:
        X1.append([element,1])
    Y1 = []
    for element in Y:
        Y1.append([element])
    X_T = transpose(X1)
    left = matxMultiply(X_T,X1) #2x2
    right = matxMultiply(X_T,Y1) #2x1
    # left*h = right
    # h is 2x1
    # a = [ a11 a12
    #       a21 a22]
    # h = [h1
    #      h2]
    a11 = left[0][0]
    a12 = left[0][1]
    a21 = left[1][0]
    a22 = left[1][1]
    c1 = right[0][0]
    c2 = right[1][0]
    h2 = (a21*c1-a11*c2)/(a21*a12-a22*a11)
    h1 = (c1-a12*h2)/a11
    return(h1,h2)

m2,b2 = linearRegression2D(X,Y)
assert isinstance(m2,float),"m is not a float"
assert isinstance(b2,float),"b is not a float"
print(m2,b2)

vs_scatter_2d(X, Y, m2, b2)
print(calculateMSE2D(X,Y,m2,b2))
