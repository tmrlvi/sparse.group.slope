#pragma once

#include <RcppArmadillo.h>

#if defined(SGS_DEBUG)
#define DPRINT(x) std::cout << #x << ": " << x << std::endl;
#else
#define DPRINT(x) {}
#endif

inline void standardizeMatrix(arma::mat& X, arma::rowvec& scale) {
  X.each_row() %= scale;
}

inline void standardizeMatrix(arma::sp_mat& X, arma::rowvec& scale) {
  for(arma::sp_mat::iterator it = X.begin(); it != X.end(); ++it) {  
    (*it) = (*it)/scale[it.col()];
  }
}

template<typename C, typename T, typename F, typename R>
Rcpp::List mfista(
    F const &family, R const &reg, T X, arma::mat y, arma::mat BInit, bool intercept = true,
    double stepSize = 100, double eta = 0.9, double tol=1e-5, long maxIter = 1e5,
    double minStepSize=1e-10, bool accelerated = false, bool monotone = false,
    bool saveData=false, const double min_val=1, bool standardize=true
){
  if (BInit.n_rows == 1 && BInit.n_cols == 1) {
    BInit = arma::zeros(X.n_cols, y.n_cols) + BInit(0,0);
  }
  
  double sqrt_n = std::sqrt(X.n_rows);
  arma::rowvec colScales(arma::sqrt(arma::sum(arma::square(X), 0)));
  colScales = colScales.transform( [&](double val) { return (val < arma::datum::eps) ? 0.0 : sqrt_n / val; } );
  if (standardize) {
    //X = normalise(X, 2, 0)*sqrt_n;
    standardizeMatrix(X, colScales);
  }
  DPRINT(arma::size(X));
  DPRINT(arma::size(y));
  DPRINT(arma::size(BInit));

  arma::rowvec b_prev(BInit.n_cols, arma::fill::zeros), b(BInit.n_cols, arma::fill::zeros);
  arma::rowvec b_u(BInit.n_cols, arma::fill::zeros), b_v(BInit.n_cols, arma::fill::zeros);
  C BPrev, B(BInit);
  C U, V(B);
  double tPrev, t = 1;
  long iter = 0;

  arma::mat prod(X*B);
  prod.each_row() += b;
  prod.each_col() -= arma::max(prod, 1);

  arma::mat vprod(prod);
  const T &X_t = X.t();
  //const double inv_n_sqrt = 1/sqrt(X.n_rows);
  const double inv_n_sqrt = 1./X.n_rows;

  eta = eta*std::min(std::sqrt(X.n_rows)*min_val, 1.);
  double tol_infeas = std::max(arma::datum::eps, tol*std::min(min_val,1.));
  double tol_dual = std::max(arma::datum::eps, tol);

  while(
    iter < maxIter &&
    stepSize > minStepSize &&
    ( iter % 10 != 0 || (
    (abs(family.dualgap(prod, y)*inv_n_sqrt + reg.evaluate(B)) > tol_dual ||
      //family.dualgap(prod, y)*inv_n_sqrt + reg.evaluate(B) < 0 ||
      reg.infeasibility(-inv_n_sqrt*(X_t*(family.linkinv(prod) - y))) > tol_infeas)))
  ){
    if (iter % 100 == 0) {
        Rcpp::checkUserInterrupt();
    }

    tPrev = t;

    const arma::mat& tmp = family.derivative(vprod, y)*inv_n_sqrt;
    const arma::mat& grad = X_t * tmp ;
    const arma::rowvec& b_grad = arma::sum(tmp, 0);

    U = reg.prox(V - stepSize * grad, stepSize);

    arma::mat uprod = arma::conv_to<arma::mat>::from(X*U);
    if (intercept){
      b_u = b_v - stepSize * b_grad;
      uprod.each_row() += b_u;
    }
    uprod.each_col() -= arma::max(uprod, 1);

    // Backtracking
    double primal = family.primal(vprod, y);
    unsigned int k = 0;
    while(family.primal(uprod, y)*inv_n_sqrt > primal*inv_n_sqrt +
          arma::dot(arma::vectorise(grad), arma::vectorise(U - V)) + arma::dot(b_grad, b_u - b_v) +
          1/(stepSize*2) * arma::accu(arma::square(arma::vectorise(U - V))) + 
          arma::accu(arma::square(b_u - b_v)) && stepSize > minStepSize){
      k++;
      if (k % 100 == 0) {
        Rcpp::checkUserInterrupt();
      }
      DPRINT(stepSize);
      stepSize *= eta;
      U = reg.prox(V - stepSize * grad, stepSize);
      uprod = arma::conv_to<arma::mat>::from(X*U);
      if (intercept){
        b_u = b_v - stepSize * b_grad;
        uprod.each_row() += b_u;
      }
      uprod.each_col() -= arma::max(uprod, 1);
    }

    DPRINT(stepSize);
    if (accelerated){
      BPrev = B;
      b_prev = b;
      t = (1 + sqrt(1 + 4 * tPrev*tPrev))/2;
      if (!monotone || family.primal(uprod, y)*inv_n_sqrt + reg.evaluate(U) < family.primal(prod, y)*inv_n_sqrt + reg.evaluate(B) + tol/iter){
        B = U;
        b = b_u;
      }
      V = B + (tPrev/t)*(U - B) + ((tPrev-1)/t)*(B - BPrev);
      b_v = b + (tPrev/t)*(b_u - b) + ((tPrev-1)/t)*(b - b_prev);
    } else {
      B = U;
      V = B;
      b = b_u;
      b_v = b;
    }

    iter++;
    prod = arma::conv_to<arma::mat>::from(X*B);
    prod.each_row() += b;
    prod.each_col() -= arma::max(prod, 1);
    vprod = arma::conv_to<arma::mat>::from(X*V);
    vprod.each_row() += b_v;
    vprod.each_col() -= arma::max(vprod, 1);
  }

  prod = arma::conv_to<arma::mat>::from(X*B);
  prod.each_row() += b;
  prod.each_col() -= arma::max(prod, 1);
  if (abs(family.dualgap(prod, y)*inv_n_sqrt + reg.evaluate(B)) > tol ||
      reg.infeasibility(-inv_n_sqrt*(X_t*(family.linkinv(prod) - y))) > tol_infeas){
    std::cerr << "Stopped before convergence. Dual gap: " << family.dualgap(prod, y)*inv_n_sqrt+ reg.evaluate(B)
              << ", infeasibility: " << reg.infeasibility(-inv_n_sqrt*(X_t*(family.linkinv(prod) - y)))
              << " (iterations: " << iter << ", last step size: " << stepSize << ")" << std::endl;
  }
  Rcpp::List res = Rcpp::List::create(
    Rcpp::Named("coefficients") = arma::mat(B),
    Rcpp::Named("intercepts") = b,
    Rcpp::Named("final.loss") = family.primal(prod, y)*inv_n_sqrt + reg.evaluate(B),
    Rcpp::Named("final.dualgap") = family.dualgap(prod, y)*inv_n_sqrt + reg.evaluate(B),
    Rcpp::Named("final.infeasibility") = reg.infeasibility(X_t*(family.linkinv(prod) - y)*inv_n_sqrt),
    Rcpp::Named("iterations") = iter,
    Rcpp::Named("final.step.size") = stepSize,
    Rcpp::Named("standardize") = standardize,
    Rcpp::Named("col.scale") = colScales
  );
  if (saveData){
    res["data"] = Rcpp::List::create(
      Rcpp::Named("X") = X,
      Rcpp::Named("y") = y
    );
  }
  res.attr("class") = "sgs";
  return res;
}