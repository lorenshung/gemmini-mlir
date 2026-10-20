"builtin.module"() ({
  "func.func"() <{sym_name = "forward", function_type = (tensor<17x16xi8>, tensor<16x32xi8>) -> tensor<17x32xi32>}> ({
  ^bb0(%0: tensor<17x16xi8>, %1: tensor<16x32xi8>):
    %2 = "arith.constant"() <{value = 0 : i32}> {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : () -> i32
    %3 = "tensor.splat"(%2) {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : (i32) -> tensor<17x32xi32>
    %4 = "linalg.generic"(%0, %1, %3) <{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb1(%5: i8, %6: i8, %7: i32):
      %8 = "arith.extsi"(%5) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i8) -> i32
      %9 = "arith.extsi"(%6) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i8) -> i32
      %10 = "arith.muli"(%8, %9) <{overflowFlags = #arith.overflow<none>}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i32, i32) -> i32
      %11 = "arith.addi"(%7, %10) <{overflowFlags = #arith.overflow<none>}> {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i32, i32) -> i32
      "linalg.yield"(%11) {prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering"} : (i32) -> ()
    }) {prov.region_id = "matmul_0", prov.family = "contraction", prov._pattern_hint = "int_matmul", prov.op = "int_matmul", prov.source_node_ids = ["g:prepared:root:n2"], prov.origin_node_ids = ["g:original:root:n2", "g:quantized:root:n2"], prov.trace_role = "lowering", prov.aten = "aten._int_mm.default", prov.orig_dtype = "int32"} : (tensor<17x16xi8>, tensor<16x32xi8>, tensor<17x32xi32>) -> tensor<17x32xi32>
    "func.return"(%4) : (tensor<17x32xi32>) -> ()
  }) : () -> ()
}) {prov.weights_file = "capsule.weights.safetensors", prov.level = "linalg-on-tensors"} : () -> ()