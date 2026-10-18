builtin.module attributes {m2m.result_layout = "{\22authority\22: \22same_conversion_exported_program\22, \22inputs\22: [{\22dtype\22: \22int8\22, \22requires_grad\22: false, \22shape\22: [5, 10], \22storage_offset\22: 0, \22stride\22: [10, 1]}], \22results\22: [{\22alias_inputs\22: [], \22alias_results\22: [], \22dtype\22: \22int32\22, \22requires_grad\22: false, \22role\22: \22user_output\22, \22shape\22: [5, 10], \22storage_offset\22: 0, \22stride\22: [10, 1]}], \22schema_version\22: 1}", prov.level = "linalg-on-tensors"} {
  func.func @forward(%0: tensor<5x10xi8>) -> tensor<5x10xi32> {
    %1 = "tensor.empty"() {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int32"} : () -> tensor<5x10xi32>
    %2 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : tensor<5x10xi8>) outs(%1 : tensor<5x10xi32>) attrs =  {prov.region_id = "dtype_cast_0", prov._pattern_hint = "dtype_cast", prov.op = "dtype_cast", prov.family = "cast", prov.aten = "aten._to_copy.default", prov.orig_dtype = "int32"} {
    ^bb0(%3: i8, %4: i32):
      %5 = arith.extsi %3 : i8 to i32
      linalg.yield %5 : i32
    } -> tensor<5x10xi32>
    func.return %2 : tensor<5x10xi32>
  }
}