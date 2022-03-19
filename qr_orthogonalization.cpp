#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void qrOrthogonalizationCuda(torch::Tensor A, int m, int n, float epsilon);

void qrOrthogonalization(torch::Tensor A, float epsilon){
    CHECK_INPUT(A);

    const int m = A.size(0);
    const int n = A.size(1);
    
    return qrOrthogonalizationCuda(A, m, n, epsilon);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qr_orthogonalization", &qrOrthogonalization, "QR Orthogonalization");
}