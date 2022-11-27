#pragma once
#include <RcppArmadillo.h>
#include "utils/prox.h"

class SparseGroupSlope {
private:
  const arma::vec lambda;
  const arma::vec alpha;
public:
  SparseGroupSlope(arma::vec lambda, arma::vec alpha) : lambda(lambda), alpha(alpha) {}

  double evaluate (const arma::mat& B) const {
    double res = arma::dot(arma::sort(arma::sqrt(arma::sum(arma::square(B), 1)), "descend"), lambda);
    B.each_row([this, &res](const arma::rowvec &row){res += arma::dot(arma::sort(arma::abs(row), "descend"), alpha.t());});
    return res;
  }

  /*double evaluate(const arma::sp_mat& B){
   arma::sp_mat Bcopy(B.t());
   std::vector<double> col_norms;
   std::vector<std::vector<double>> cols;

   sp_mat::const_iterator it = Bcopy.begin();
   sp_mat::const_iterator it_end = Bcopy.end();
   double col_norm = 0;
   long cur_col = 0;
   std::vector<double> cur_row;
   for(; it != it_end; ++it){
   if(it.col() != cur_col && col_norm != 0){
   col_norms.push_back(sqrt(col_norm));
   cols.push_back(cur_row);
   cur_col = it.col();
   cur_row.clear();
   col_norm = 0;
   }
   col_norm += pow(*it, 2);
   cur_row.push_back(*it);
   }
   if(col_norm != 0){
   col_norms.push_back(sqrt(col_norm));
   cols.push_back(cur_row);
   }

   if (col_norms.size() == 0){
   return 0;
   }
   double res = arma::dot(arma::sort(arma::vec(col_norms)), lambda.head(col_norms.size()));
   std::vector<std::vector<double>>::iterator col_it = cols.begin();
   for ( ; col_it != cols.end(); ++col_it){
   res += arma::dot(arma::sort(arma::vec(*col_it)), alpha.head((*col_it).size()));
   }
   return res;
  }*/


  double infeasibility(const arma::mat& B) const{
    return (arma::abs(prox(B))).max();
  }

  const arma::mat prox(const arma::mat& B, double weight=1) const {
    arma::mat Bcopy = B;
    Bcopy = Bcopy.each_row([this, weight] (arma::rowvec &row) {
      row = slope_prox(row, alpha*weight);
    });
    /*arma::mat Bcopy(size(B));
    arma::vec norms(B.n_rows);
    #pragma omp parallel for shared(Bcopy, B, norms)
    for (int i=0; i<B.n_rows; i++) {
      Bcopy.row(i) = slope_prox(B.row(i).eval(), alpha*weight);
      norms(i) = arma::norm(Bcopy.row(i));
      if (norms(i) > arma::datum::eps) {
        Bcopy.row(i) /= norms(i);
      }
    }

    const arma::vec& scale = slope_prox(norms, lambda*weight);
    return Bcopy.each_col() % scale;*/

    const arma::vec norms = arma::sqrt(arma::sum(arma::square(Bcopy), 1));
    const arma::vec& scale = slope_prox(norms, lambda*weight);
    return arma::normalise(Bcopy, 2, 1).eval().each_col() % scale;
  }

  /*arma::mat subgradient(arma::mat X){
    arma::vec norms= arma::sqrt(arma::sum(arma::square(X), 1));
    arma::uvec sorted_indices = arma::stable_sort_index(norms, "descend");
    int i = 0;
    for (auto it = sorted_indices.begin(); it!=sorted_indices.end(); ++it){
      if (norms(*it) >= 1e-12){
        norms(*it) = lambda(i)/norms(*it);
      } else {
        norms(*it) = 0;
      }
      i++;
    }
    arma::mat Xcopy(X);
    Xcopy = Xcopy.each_row([this](arma::rowvec row){
      arma::uvec sorted_indices =  arma::stable_sort_index(row, "descend");
      int i = 0;
      for (auto it = sorted_indices.begin(); it!=sorted_indices.end(); ++it){
        if (row(*it) >= 1e-12){
          row(*it) = alpha(i);
        } else {
          row(*it) = 0;
        }
        i++;
      }
    });
    return arma::diagmat(norms)*X + Xcopy;
  }*/
};
