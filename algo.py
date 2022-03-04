import torch

def original_orth(A, diag_eps):
    M, N = A.shape
    device = A.device

    R = torch.clone(A)
    R += torch.eye(M, N, device=device) * diag_eps

    Q = torch.eye(M, N, device=device)

    vs = [None for _ in range(M)] #

    #first kernel
    for i in range(M):
        z = R[i,i:]
        v = -z
        v[0] -= torch.linalg.norm(z) * z[0] / abs(z[0])
        v /= torch.linalg.norm(v)

        for m in range(M):
            R[m,i:] -= 2.0 * v * torch.dot(v, R[m,i:])

        vs[i] = v #


    #second kernel
    for i in reversed(range(M)): #
        v = vs[i][None, :] #
        Q[:, i:] -= 2.0 * (Q[:, i:] @ v.T) @ v
        # v = vs[i]
        # for r in range(M):
        #     Q[r, i:] -= 2.0 * v * np.dot(Q[r, i:], v)

    return Q

