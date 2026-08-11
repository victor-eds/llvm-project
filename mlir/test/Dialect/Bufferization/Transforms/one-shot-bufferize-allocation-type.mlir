// RUN: mlir-opt %s -test-one-shot-module-bufferize -split-input-file | FileCheck %s
// RUN: mlir-opt %s -test-one-shot-module-bufferize=pad-allocation-rows -split-input-file | FileCheck %s --check-prefix=CHECK-PAD

// The default allocation type is a static identity layout. `allocationTypeFn`
// replaces it, and `bufferization.alloc_tensor` allocates the type it reports,
// so a consumer that predicts the type through `getBufferType` agrees with the
// buffer that is finally allocated.

// CHECK-LABEL: func @alloc_tensor_allocation_type(
// CHECK: memref.alloc() {{.*}} : memref<2x3xf32>
//
// CHECK-PAD-LABEL: func @alloc_tensor_allocation_type(
// CHECK-PAD: memref.alloc() {{.*}} : memref<2x3xf32, strided<[4, 1]>>
func.func @alloc_tensor_allocation_type(%f: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %0 = bufferization.alloc_tensor() : tensor<2x3xf32>
  %1 = tensor.insert %f into %0[%c0, %c0] : tensor<2x3xf32>
  %2 = tensor.extract %1[%c0, %c0] : tensor<2x3xf32>
  return %2 : f32
}

// -----

// A consumer that derives its own type from the allocation sees the layout the
// hook reported, not an identity layout.

// CHECK-LABEL: func @extract_slice_of_allocation(
// CHECK: memref.subview {{.*}} : memref<2x3xf32> to memref<2xf32, strided<[3]>>
//
// CHECK-PAD-LABEL: func @extract_slice_of_allocation(
// CHECK-PAD: memref.subview {{.*}} : memref<2x3xf32, strided<[4, 1]>> to memref<2xf32, strided<[4]>>
func.func @extract_slice_of_allocation(%f: f32) -> tensor<2xf32> {
  %c0 = arith.constant 0 : index
  %0 = bufferization.alloc_tensor() : tensor<2x3xf32>
  %1 = tensor.insert %f into %0[%c0, %c0] : tensor<2x3xf32>
  %2 = tensor.extract_slice %1[0, 0] [2, 1] [1, 1] : tensor<2x3xf32> to tensor<2xf32>
  return %2 : tensor<2xf32>
}
