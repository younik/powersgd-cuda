import pytest
import torch

import orthogonalization

torch.manual_seed(0)

devices = ["cuda"] #, "cpu"] if torch.cuda.is_available() else ["cpu"]
shapes = [(100, 3), (10, 10)]
dtypes = [torch.float64, torch.float32]  # , torch.float16]

matrices = [torch.randn(shape, dtype=dtype) 
            for shape in shapes 
            for dtype in dtypes
            ]


@pytest.mark.parametrize("M", matrices)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("algo", orthogonalization.implementations)
def test_normalized_columns(M, algo, device):
    A = M.clone().to(device)
    Q = A.clone()
    algo(Q, diag_eps=0)

    column_sq_norms = Q.square().sum(axis=0)

    assert Q.shape == A.shape
    assert column_sq_norms.allclose(torch.ones([], device=device, dtype=M.dtype))


@pytest.mark.parametrize("M", matrices)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("algo", orthogonalization.implementations)
def test_orthogonal(M, algo, device):
    A = M.clone().to(device)
    Q = A.clone()
    algo(Q, diag_eps=0)

    assert torch.eye(Q.shape[1], device=device, dtype=M.dtype).allclose(
        Q.T @ Q, rtol=1e-03, atol=1e-04
    )


@pytest.mark.parametrize("M", matrices)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("algo", orthogonalization.implementations)
def test_not_worse_than_ref(M, algo, device):
    A = M.clone().to(device)
    Q, Q_ref = A.clone(), A.clone()

    algo(Q, diag_eps=0)
    orthogonalization.pytorch(Q_ref, diag_eps=0)

    distance = torch.linalg.norm(A - Q)
    ref_distance = torch.linalg.norm(A - Q_ref)

    assert distance <= ref_distance * 1.05


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("algo", orthogonalization.implementations)
def test_idempotence(algo, device):
    A = torch.eye(100, device=device)[:, :10]  # orthogonal matrix 
    #A = torch.qr(torch.randn(100, 10, device=device)).Q
    Q = A.clone()
    algo(Q, diag_eps=0)
    assert Q.allclose(A) or Q.allclose(-A)
