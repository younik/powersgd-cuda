import pytest
import torch

import orthogonalization

torch.manual_seed(0)

devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]

matrices = [
    torch.randn([100, 3]),
    torch.randn([10, 10]),
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

    assert column_sq_norms.allclose(torch.ones([], device=device))


@pytest.mark.parametrize("M", matrices)
@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("algo", orthogonalization.implementations)
def test_orthogonal(M, algo, device):
    A = M.clone().to(device)
    Q = A.clone()
    algo(Q, diag_eps=0)

    assert torch.eye(Q.shape[1], device=device).allclose(Q.T @ Q, rtol=1e-03, atol=1e-04)


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
