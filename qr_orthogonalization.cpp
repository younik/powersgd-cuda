#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void qrOrthogonalizationCuda(torch::Tensor A, torch::Tensor out, int m, int n, float epsilon);

torch::Tensor qrOrthogonalization(torch::Tensor A, float epsilon, torch::Tensor out){
    CHECK_INPUT(A);
    const int m = A.size(0);
    const int n = A.size(1);

    if (out.size(0) == 0) {
      out = torch::empty({m, n}, A.options());
    }
    CHECK_INPUT(out);
    TORCH_CHECK(A.sizes() == out.sizes(), "Output and input tensors must have same sizes.");
    
    qrOrthogonalizationCuda(A, out, m, n, epsilon);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("qr_orthogonalization",
        &qrOrthogonalization,
        py::arg("A"),
        py::arg("epsilon") = 1e-8,
        py::arg("out") = torch::empty({0})
  );
}