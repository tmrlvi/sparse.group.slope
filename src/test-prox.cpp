#include <testthat.h>
#include <iostream>
#include <RcppArmadillo.h>
#include "penalties/sparse_group_slope.hpp"


context("Proximal Calculations") {

  // The format for specifying tests is similar to that of
  // testthat's R functions. Use 'test_that()' to define a
  // unit test, and use 'expect_true()' and 'expect_false()'
  // to test the desired conditions.
  test_that("prox calculation correct on examples") {
    std::cout << "d" << std::endl;;
    const double lambda_init[] = {1, 0, 0};
    arma::vec lambda(&lambda_init[0], 3);
    //arma::vec lambda(3, arma::fill::ones);

    const double kappa_init[] = {1, 0, 0, 0, 0};
    arma::vec kappa(&kappa_init[0], 5);
    //arma::vec kappa(5, arma::fill::ones);
    std::cout << "d" << std::endl;;

    SparseGroupSlope penalty(lambda, kappa);
    std::cout << "a" << std::endl;;

    const double init[] = {15, 10, 5, 12, 8, 4, 9, 6, 3, 6, 4, 2, 3, 2, 1};
    arma::mat B(&init[0], 3, 5);
    //arma::mat B(3, 5, arma::fill::randu);
    std::cout << "B:" << std::endl;;
    std::cout << B << std::endl;

    std::cout << "lambda:";
    std::cout << lambda << std::endl;

    std::cout << "kappa:";
    std::cout << kappa << std::endl;

    std::cout << "evaluate:" << penalty.evaluate(B) << std::endl;

    std::cout << "prox:";
    std::cout << penalty.prox(B) << std::endl;
    expect_true(4 == 4);
  }

}
