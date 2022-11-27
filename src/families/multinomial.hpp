#pragma once

#include <RcppArmadillo.h>
#include <cmath>

class Multinomial {
public:
  const double primal(const arma::mat &pred, const arma::mat &y) const {     
    double res = arma::accu(arma::log(
      arma::sum(arma::expm1(pred), 1) + pred.n_cols
    )) - arma::accu(pred % y);
    if (res < 0) throw std::overflow_error("negative log likelihood is smaller then 0. Might be numerical problem");
    return res;
  }

  // pred is subtracted the maximum. b is the vector of maxima
  const double dualgap(const arma::mat &pred, const arma::mat &y) const{
    arma::mat p = linkinv(pred);
    return arma::accu((p - y)%pred);
  }

  const arma::mat linkinv(const arma::mat &pred) const {
    return arma::normalise(arma::exp(pred), 1, 1);
  }

  const arma::mat derivative(const arma::mat &pred, const arma::mat &y) const{
    return linkinv(pred) - y;
  }
};