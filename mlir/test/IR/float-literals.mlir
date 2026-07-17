// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// Human-readable floating point special values and C-style hexadecimal floats,
// as accepted by the parser and produced by the printer.

// CHECK-LABEL: @infinities
func.func @infinities() {
  // CHECK: arith.constant +inf : f32
  %0 = arith.constant +inf : f32
  // CHECK: arith.constant -inf : f32
  %1 = arith.constant -inf : f32
  // CHECK: arith.constant +inf : f64
  %2 = arith.constant +inf : f64
  // CHECK: arith.constant -inf : f16
  %3 = arith.constant -inf : f16
  // CHECK: arith.constant +inf : bf16
  %4 = arith.constant +inf : bf16
  return
}

// CHECK-LABEL: @quiet_nans
func.func @quiet_nans() {
  // CHECK: arith.constant +qnan : f32
  %0 = arith.constant +qnan : f32
  // CHECK: arith.constant -qnan : f64
  %1 = arith.constant -qnan : f64
  // CHECK: arith.constant +qnan : f16
  %2 = arith.constant +qnan : f16
  // CHECK: arith.constant -qnan : bf16
  %3 = arith.constant -qnan : bf16
  return
}

// CHECK-LABEL: @nan_payloads
func.func @nan_payloads() {
  // CHECK: arith.constant +nan(0x1) : f32
  %0 = arith.constant +nan(0x1) : f32
  // CHECK: arith.constant -nan(0x3FFFFF) : f32
  %1 = arith.constant -nan(0x3FFFFF) : f32
  // CHECK: arith.constant +snan(0x1) : f64
  %2 = arith.constant +snan(0x1) : f64
  // CHECK: arith.constant -snan(0x1000000) : f64
  %3 = arith.constant -snan(0x1000000) : f64
  return
}

// CHECK-LABEL: @hex_floats
func.func @hex_floats() {
  // 0x1.8p3 == 1.5 * 2^3 == 12.0
  // CHECK: arith.constant 1.200000e+01 : f64
  %0 = arith.constant 0x1.8p3 : f64
  // CHECK: arith.constant -1.200000e+01 : f64
  %1 = arith.constant -0x1.8p3 : f64
  // CHECK: arith.constant 1.600000e+01 : f32
  %2 = arith.constant 0x1p4 : f32
  return
}

// CHECK-LABEL: @special_values_in_elements
func.func @special_values_in_elements() {
  // CHECK: dense<[+inf, -inf, +qnan]> : tensor<3xf32>
  %0 = arith.constant dense<[+inf, -inf, +qnan]> : tensor<3xf32>
  return
}
