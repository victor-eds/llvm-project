// RUN: mlir-opt %s -split-input-file -canonicalize | FileCheck %s
// RUN: mlir-opt %s -split-input-file -test-single-fold | FileCheck %s --check-prefix=SINGLE
// Partial folding must reach a fixed point: the pass fails if the greedy
// driver does not converge.
// RUN: mlir-opt %s -split-input-file -canonicalize='test-convergence=true' | FileCheck %s

// Only the result whose operand is constant folds. The operation stays and
// keeps defining the other result.

// CHECK-LABEL: func @partial_fold_one_result
//  CHECK-SAME:   %[[ARG:.*]]: i32
//       CHECK:   %[[C:.*]] = "test.constant"() <{value = 42 : i32}>
//       CHECK:   %[[FOLD:.*]]:2 = test.partial_fold %[[C]], %[[ARG]]
//       CHECK:   return %[[C]], %[[FOLD]]#1

// SINGLE-LABEL: func @partial_fold_one_result
//  SINGLE-SAME:   %[[ARG:.*]]: i32
//       SINGLE:   %[[C:.*]] = "test.constant"() <{value = 42 : i32}>
//       SINGLE:   %[[FOLD:.*]]:2 = test.partial_fold %[[C]], %[[ARG]]
//       SINGLE:   return %[[C]], %[[FOLD]]#1
func.func @partial_fold_one_result(%arg0: i32) -> (i32, i32) {
  %c42 = "test.constant"() {value = 42 : i32} : () -> i32
  %0:2 = "test.partial_fold"(%c42, %arg0) : (i32, i32) -> (i32, i32)
  return %0#0, %0#1 : i32, i32
}

// -----

// Every result folds, so the operation is erased.

// CHECK-LABEL: func @complete_fold_erases_op
//   CHECK-NOT:   test.partial_fold
//       CHECK:   %[[C0:.*]] = "test.constant"() <{value = 42 : i32}>
//       CHECK:   %[[C1:.*]] = "test.constant"() <{value = 24 : i32}>
//       CHECK:   return %[[C0]], %[[C1]]
func.func @complete_fold_erases_op() -> (i32, i32) {
  %c42 = "test.constant"() {value = 42 : i32} : () -> i32
  %c24 = "test.constant"() {value = 24 : i32} : () -> i32
  %0:2 = "test.partial_fold"(%c42, %c24) : (i32, i32) -> (i32, i32)
  return %0#0, %0#1 : i32, i32
}

// -----

// No operand is constant, so the folder fails and the operation is untouched.

// CHECK-LABEL: func @no_fold
//  CHECK-SAME:   %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
//       CHECK:   %[[FOLD:.*]]:2 = test.partial_fold %[[LHS]], %[[RHS]]
//       CHECK:   return %[[FOLD]]#0, %[[FOLD]]#1
func.func @no_fold(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %0:2 = "test.partial_fold"(%arg0, %arg1) : (i32, i32) -> (i32, i32)
  return %0#0, %0#1 : i32, i32
}

// -----

// A partially folded operation whose remaining results are all unused is dead
// and gets removed.

// CHECK-LABEL: func @partial_fold_then_dead
//   CHECK-NOT:   test.partial_fold
//       CHECK:   %[[C:.*]] = "test.constant"() <{value = 42 : i32}>
//       CHECK:   return %[[C]]
func.func @partial_fold_then_dead(%arg0: i32) -> i32 {
  %c42 = "test.constant"() {value = 42 : i32} : () -> i32
  %0:2 = "test.partial_fold"(%c42, %arg0) : (i32, i32) -> (i32, i32)
  return %0#0 : i32
}

// -----

// The folder of this operation reports success on every invocation. The driver
// must apply the fold once and then reach a fixed point instead of looping.

// CHECK-LABEL: func @partial_fold_reaches_fixed_point
//  CHECK-SAME:   %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
//       CHECK:   %{{.*}}, %[[RES1:.*]] = test.partial_fold_to_value %[[LHS]], %[[RHS]]
//       CHECK:   return %[[LHS]], %[[RES1]]

// SINGLE-LABEL: func @partial_fold_reaches_fixed_point
//  SINGLE-SAME:   %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
//       SINGLE:   %{{.*}}, %[[RES1:.*]] = test.partial_fold_to_value %[[LHS]], %[[RHS]]
//       SINGLE:   return %[[LHS]], %[[RES1]]
func.func @partial_fold_reaches_fixed_point(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %0:2 = test.partial_fold_to_value %arg0, %arg1 : i32
  return %0#0, %0#1 : i32, i32
}

// -----

// A folded result must reach its users, so that they can fold in turn. Here
// `arith.addi` folds only because the first result folded to 42.

// CHECK-LABEL: func @partial_fold_enables_user_fold
//   CHECK-DAG:   %[[C43:.*]] = arith.constant 43 : i32
//   CHECK-DAG:   %[[C42:.*]] = "test.constant"() <{value = 42 : i32}>
//       CHECK:   %[[FOLD:.*]]:2 = test.partial_fold %[[C42]], %{{.*}}
//       CHECK:   return %[[C43]], %[[FOLD]]#1
func.func @partial_fold_enables_user_fold(%arg0: i32) -> (i32, i32) {
  %c42 = "test.constant"() {value = 42 : i32} : () -> i32
  %c1 = arith.constant 1 : i32
  %0:2 = "test.partial_fold"(%c42, %arg0) : (i32, i32) -> (i32, i32)
  %1 = arith.addi %0#0, %c1 : i32
  return %1, %0#1 : i32, i32
}

// -----

// A fold result that is a result of the folded operation means "not folded",
// whatever its index. Without that, the driver would swap the uses of the two
// results on every visit and never converge.

// CHECK-LABEL: func @fold_to_own_results_is_not_a_fold
//  CHECK-SAME:   %[[LHS:.*]]: i32, %[[RHS:.*]]: i32
//       CHECK:   %[[A:.*]], %[[B:.*]] = test.fold_to_own_results %[[LHS]], %[[RHS]]
//       CHECK:   return %[[A]], %[[B]]
func.func @fold_to_own_results_is_not_a_fold(%arg0: i32, %arg1: i32) -> (i32, i32) {
  %0:2 = test.fold_to_own_results %arg0, %arg1 : i32
  return %0#0, %0#1 : i32, i32
}
