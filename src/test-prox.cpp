#include <testthat.h>
#include <cmath>
#include <iostream>
#include <RcppArmadillo.h>
#include "penalties/sparse_group_slope.hpp"


context("Proximal Calculations") {

  // The format for specifying tests is similar to that of
  // testthat's R functions. Use 'test_that()' to define a
  // unit test, and use 'expect_true()' and 'expect_false()'
  // to test the desired conditions.
  test_that("prox calculation correct on examples") {
    const double expected_prox_arr[] = {
      7.97740041261, 3.65160027507, 0, 
      6.38192033009, 2.92128022005, 0, 
      4.78644024756, 2.19096016505, 0, 
      3.19096016505, 1.46064011003, 0, 
      1.59548008252, 0.73032005501, 0
    };
    arma::mat expected_prox(&expected_prox_arr[0], 3, 5);

    const double lambda_init[] = {3, 2, 1};
    arma::vec lambda(&lambda_init[0], 3);

    const double kappa_init[] = {5, 4, 3, 2, 1};
    arma::vec kappa(&kappa_init[0], 5);

    SparseGroupSlope penalty(lambda, kappa);

    const double init[] = {15, 10, 5, 12, 8, 4, 9, 6, 3, 6, 4, 2, 3, 2, 1};
    arma::mat B(&init[0], 3, 5);


    expect_true(std::abs(penalty.evaluate(B) - 433.826778819339319) < 1e-10);
    expect_true(arma::norm(penalty.prox(B) - expected_prox, "inf") < 1e-10);
  }

}
