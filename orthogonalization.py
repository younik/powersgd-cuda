import torch
import torch.utils.cpp_extension as cpp_extension

try:
    _qr_func = cpp_extension.load(
        name="qr_func",
        sources=["qr_orthogonalization.cpp", "qr_orthogonalization.cu"],
    )
except RuntimeError as e:
    _qr_func = None
    print("Failed to compile cuda op")


def pytorch(A: torch.Tensor, diag_eps: float = 0):
    """
    Orthogonalize the columns of A using PyTorch's built-in Householder QR decomposition
    Assumes A has more rows than columns
    """
    assert diag_eps == 0
    A.data = torch.linalg.qr(A).Q


def cuda_v1(A: torch.Tensor, diag_eps: float = 0):
    """
    Orthogonalize the columns of A using Omar's custom CUDA op
    Assumes A has more rows than columns.
    """
    if _qr_func is None:
        raise RuntimeError("CUDA op is not available.")

    # This algorithm was written to orthogonalize *rows* of A instead of *columns*,
    # so we transpose the matrix first
    A = A.T

    Q = torch.empty_like(A)
    _qr_func.qr_orthogonalization(A.clone(), Q, diag_eps)  # type: ignore
    A.data = Q.T


def householder_manual(A: torch.Tensor, diag_eps: float = 0):
    """
    Orthogonalize the columns of A using a manual householder implementation.
    Assumes A has more rows than columns.
    """
    m, n = A.shape
    assert n <= m
    q = torch.eye(m, n, dtype=A.dtype, device=A.device)
    r = A.add(q, alpha=diag_eps)
    for k in range(n):
        z = r[k:, k : (k + 1)]
        v = -z
        v[0] -= torch.linalg.norm(z) * torch.sign(z[0])
        v /= torch.linalg.norm(v)
        r[k:, k:].addmm_(v, v.T @ r[k:, k:], alpha=-2)
        q[k:].addmm_(v, v.T @ q[k:], alpha=-2)
    A.data = q


def torch_powersgd(A: torch.Tensor, diag_eps: float = 0):
    """
    Applies Gram-Schmidt procedure to orthogonalize a given 2D tensor.
    If epsilon is 0, this is equivalent to `torch.qr(matrix, out=(matrix, _))`,
    """
    # TODO Consider using Q = torch.orgqr(*torch.geqrf(A)) to compute the Q of the QR _much_ faster
    # and more reliably.
    # Works on FP32/64 or complex numbers (does not work for half precision)
    num_cols = A.shape[1]
    for i in range(num_cols):
        # Normalize the i'th column.
        col = A[:, i : i + 1]
        # If no epsilon is added here, division by zero may be caused by vanishing gradients.
        # This epsilon is not needed if the input matrix covers the gradients of at least one entire layer in the neural network.
        if diag_eps == 0:
            # Note that col ** 2 can underflow/overflow if we use FP16.
            # May need to consider multiplying a scaling factor and dividing it later, or using bfloat16 instead.
            try:
                col /= torch.norm(col)
            except ZeroDivisionError:
                # Recover the values from NaNs to 0s.
                col.fill_(0.0)
        else:
            col /= torch.norm(col) + diag_eps
        # Project it on the rest and remove it.
        if i + 1 < num_cols:
            rest = A[:, i + 1 :]
            rest -= torch.sum(col * rest, dim=0) * col


def torch_orgqr(A: torch.Tensor, diag_eps: float = 0):
    """
    Orthogonalize the columns of A using PyTorch's ORGQR.
    Assumes A has more rows than columns
    """
    A.data = torch.orgqr(*torch.geqrf(A))


implementations = [pytorch, cuda_v1, householder_manual, torch_powersgd, torch_orgqr]
