import numpy as np

shape    = (64, 512)
diag_eps = 1e-8
a_dtype  = np.float16
q_dtype  = np.float16
c_dtype  = np.float32

A = np.random.normal(0.0, 1.0, shape).astype(a_dtype)
M, N = shape

def routine(A, original = True):
    R = A.astype(c_dtype)
    R += np.eye(M, N, dtype=c_dtype) * diag_eps

    Q = np.eye(M, N, dtype=c_dtype)
    if original: 
        Q = np.eye(N, dtype=c_dtype)


    vs = [None for _ in range(M)] #

    #first kernel
    for i in range(M):
        z = R[i,i:]
        v = -z
        v[0] -= np.copysign(np.linalg.norm(z), z[0])
        v /= np.linalg.norm(v)

        for m in range(M):
            R[m,i:] -= 2.0 * v * np.dot(v, R[m,i:])

        vs[i] = v #

    #second kernel
    for i in range(M): #
        v = vs[i] #

        # for n in range(N):
        #     Q[n,i:] -= 2.0 * v * np.dot(v, Q[n,i:])
        
        Q[:, i:] -= 2.0 * np.outer(Q[:, i:] @ v, v) #equal to above, checked


    if original:
        Q = Q[:,:M].T
    
    return Q.astype(q_dtype)

Q1 = routine(A)
Q2 = routine(A, original=False)
print(Q1.shape)
print(Q2.shape)
print(Q1 == Q2)
print(np.linalg.norm(Q1 - Q2))