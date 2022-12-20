// [[Rcpp::plugins(cpp14)]]
// [[Rcpp::depends(RcppArmadillo)]]
#include <iostream>
#include <string>
#include <math.h>
#include <RcppArmadillo.h>

#include "mfista.hpp"
#include "families/multinomial.hpp"
#include "penalties/sparse_group_slope.hpp"

#if defined(SGS_DEBUG)
#define DPRINT(x) std::cout << #x << ": " << x << std::endl;
#else
#define DPRINT(x) {}
#endif

//[[Rcpp::export]]
Rcpp::List SparseGroupSLOPE(
    Rcpp::RObject X, arma::mat y, arma::vec lambda, arma::vec alpha, Rcpp::RObject BInit, bool intercept = true,
    double stepSize = 100, double eta = 0.9, double tol=1e-5, long maxIter = 1e5,
    double minStepSize=1e-10, bool accelerated = false, bool monotone = false, bool saveData=false,
    bool standardize = true
) {
  DPRINT("started. compiled at " + std::string(__TIME__));
  Rcpp::Function ncols("ncol");
  DPRINT(Rcpp::as<long>(ncols(X)));
  DPRINT(lambda.n_elem);
  if (Rcpp::as<long>(ncols(X)) != lambda.n_elem){
    throw std::invalid_argument("Lambda is different then the number of columns in X");
  }
  if (y.n_cols == 1){
    throw std::invalid_argument("Encoding Y is not yet implemented. Please pass one hot encoding");
  }
  if (y.n_cols != alpha.n_elem){
    throw std::invalid_argument("Alpha is different then the number of classes (columns in y)");
  }

  Multinomial family;
  SparseGroupSlope reg(lambda, alpha);

  const double min_val = std::min(arma::min(lambda), arma::min(alpha));
  DPRINT(min_val);

  arma::mat BInitMat;
  if(Rcpp::is<Rcpp::NumericMatrix>(BInit)){
    BInitMat = Rcpp::as<arma::mat>(BInit);
  }
  else if(Rcpp::is<Rcpp::NumericVector>(BInit)){
    BInitMat = Rcpp::as<arma::vec>(BInit);
  }

  if (X.isS4()) {
    if(X.inherits("dgCMatrix")) {
      return single_mfista<arma::mat>(
        family, reg, Rcpp::as<arma::sp_mat>(X), y, BInitMat, intercept, stepSize, eta, tol, maxIter,
        minStepSize, accelerated, monotone, saveData, min_val, standardize=standardize
      );
    }
    Rcpp::stop("unknown class of X");
  } else {
    return single_mfista<arma::mat>(
      family, reg, Rcpp::as<arma::mat>(X), y, BInitMat, intercept, stepSize, eta, tol, maxIter,
      minStepSize, accelerated, monotone, saveData, min_val, standardize=standardize
    );
  }
}

// [[Rcpp::export]]
double norm_sgs(arma::mat B, arma::vec lambda, arma::vec alpha){
  SparseGroupSlope sgs(lambda, alpha);
  return sgs.evaluate(B);
}