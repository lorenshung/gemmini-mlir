builtin.module {
  func.func @forward(%a: tensor<2x2xi8>, %b: tensor<2x2xi8>, %c: tensor<2x2xi8>, %d: tensor<2x2xi8>) -> (tensor<2x2xi32>, tensor<2x2xf32>) {
    %zero = arith.constant 0 : i32
    %z0 = tensor.splat %zero : tensor<2x2xi32>
    %z1 = tensor.splat %zero : tensor<2x2xi32>
    %raw = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%a, %b : tensor<2x2xi8>, tensor<2x2xi8>) outs(%z0 : tensor<2x2xi32>) {
    ^bb_raw(%a_raw: i8, %b_raw: i8, %acc_raw: i32):
      %ai_raw = arith.extsi %a_raw : i8 to i32
      %bi_raw = arith.extsi %b_raw : i8 to i32
      %p_raw = arith.muli %ai_raw, %bi_raw : i32
      %sum_raw = arith.addi %acc_raw, %p_raw : i32
      linalg.yield %sum_raw : i32
    } -> tensor<2x2xi32>
    %readout_acc = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%c, %d : tensor<2x2xi8>, tensor<2x2xi8>) outs(%z1 : tensor<2x2xi32>) {
    ^bb_readout_acc(%a_readout_acc: i8, %b_readout_acc: i8, %acc_readout_acc: i32):
      %ai_readout_acc = arith.extsi %a_readout_acc : i8 to i32
      %bi_readout_acc = arith.extsi %b_readout_acc : i8 to i32
      %p_readout_acc = arith.muli %ai_readout_acc, %bi_readout_acc : i32
      %sum_readout_acc = arith.addi %acc_readout_acc, %p_readout_acc : i32
      linalg.yield %sum_readout_acc : i32
    } -> tensor<2x2xi32>
    %e = tensor.empty() : tensor<2x2xf32>
    %out = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%readout_acc : tensor<2x2xi32>) outs(%e : tensor<2x2xf32>) attrs =  {prov.quant.numerics = "integer_readout", prov.quant.accumulator = "i32", prov.quant.arithmetic = "exact", prov.quant.scale_numerator = ["1"], prov.quant.scale_denominator = ["1"]} {
    ^bb0(%value: i32, %unused: f32):
      %f = arith.sitofp %value : i32 to f32
      linalg.yield %f : f32
    } -> tensor<2x2xf32>
    func.return %raw, %out : tensor<2x2xi32>, tensor<2x2xf32>
  }
}
