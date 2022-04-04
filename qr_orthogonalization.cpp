#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void qr_orthogonalization_cuda(torch::Tensor A, torch::Tensor Q, uint m, uint n, float epsilon);

torch::Tensor qr_orthogonalization(torch::Tensor A, float epsilon, torch::Tensor Q){
    CHECK_INPUT(A);
    TORCH_CHECK(A.size(1) <= std::numeric_limits<int32_t>::max(), "Input too big. Use torch.linalg.qr instead.");
    const uint m = A.size(0);
    const uint n = A.size(1);

    if (Q.size(0) == 0) {
      Q = torch::empty({m, n}, A.options());
    }
    CHECK_INPUT(Q);
    TORCH_CHECK(A.sizes() == Q.sizes(), "Output and input tensors must have same sizes.");
    
    qr_orthogonalization_cuda(A, Q, m, n, epsilon);
    return Q;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qr_orthogonalization",
        &qr_orthogonalization,
        py::arg("A"),
        py::arg("epsilon") = 1e-8,
        py::arg("out") = torch::empty({0})
  );
}