// model2mlir cb1a3da: aten._int_mm.default on i8[3,17] x i8[17,5].
builtin.module attributes {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [3, 17], \22storage_offset\22: 0, \22stride\22: [17, 1]}, {\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [17, 5], \22storage_offset\22: 0, \22stride\22: [5, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int32\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [3, 5], \22storage_offset\22: 0, \22stride\22: [5, 1]}], \22schema_version\22: 1}", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<3x17xi8>, %1: tensor<17x5xi8>) -> tensor<3x5xi32> {
    %2 = arith.constant {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} 0 : i32
    %3 = tensor.splat %2 {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : tensor<3x5xi32>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : tensor<3x17xi8>, tensor<17x5xi8>) outs(%3 : tensor<3x5xi32>) attrs =  {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} {
    ^bb0(%5: i8, %6: i8, %7: i32):
      %8 = arith.extsi %5 : i8 to i32
      %9 = arith.extsi %6 : i8 to i32
      %10 = arith.muli %8, %9 : i32
      %11 = arith.addi %7, %10 : i32
      linalg.yield %11 : i32
    } -> tensor<3x5xi32>
    func.return %4 : tensor<3x5xi32>
  }
}
