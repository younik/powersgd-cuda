#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void qr_orthogonalization_cuda(torch::Tensor A, torch::Tensor Q, int m, int n, float epsilon);

void qr_orthogonalization(torch::Tensor A, torch::Tensor Q, float epsilon){
    CHECK_INPUT(A);
    CHECK_INPUT(Q);

    const int m = A.size(0);
    const int n = Q.size(1);
    
    return qr_orthogonalization_cuda(A, Q, m, n, epsilon);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qr_orthogonalization", &qr_orthogonalization, "QR Orthogonalization");
}