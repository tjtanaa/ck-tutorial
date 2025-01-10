#include <torch/extension.h>
// #include <bmm.h>
// #include "fp8gemm.h"
#include "machete_ops.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
//   m.def("linear_ab_i8_de_f32", &linear_ab_i8_de_f32);
  // m.def("linear_relu_abde_i8", &linear_relu_abde_i8);
  // m.def("linear_abde_fp8", &linear_abde_fp8);
  m.def("machete_prepack_B", &prepackB);
  m.def("machete_prepack_B_cpu", &prepackB_cpu);
  m.def("machete_prepack_BDebug", &prepackBDebug);
  m.def("machete_prepack_B_cpuDebug", &prepackB_cpuDebug);
  m.def("compare_fp8_cpuDebug", &compare_fp8_cpuDebug);
  // m.def("machete_mm", &machete_mm);
  // m.def("f8f8bf16_pshuffleb_rowwise", &f8f8bf16_pshuffleb_rowwise);
  m.def("machete_mm_out", &machete_mm_out);
  // m.def("linear_abde_i8", &linear_abde_i8);
//   m.def("bmm_abe_i8", &bmm_abe_i8);
//   m.def("bmm_ab_i8_e_f32", &bmm_ab_i8_e_f32);
}