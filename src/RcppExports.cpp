// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// SparseGroupSLOPE
Rcpp::List SparseGroupSLOPE(Rcpp::RObject X, arma::mat y, arma::vec lambda, arma::vec alpha, Rcpp::RObject BInit, bool intercept, double stepSize, double eta, double tol, long maxIter, double minStepSize, bool accelerated, bool monotone, bool saveData, bool standardize);
RcppExport SEXP _sparse_group_slope_SparseGroupSLOPE(SEXP XSEXP, SEXP ySEXP, SEXP lambdaSEXP, SEXP alphaSEXP, SEXP BInitSEXP, SEXP interceptSEXP, SEXP stepSizeSEXP, SEXP etaSEXP, SEXP tolSEXP, SEXP maxIterSEXP, SEXP minStepSizeSEXP, SEXP acceleratedSEXP, SEXP monotoneSEXP, SEXP saveDataSEXP, SEXP standardizeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Rcpp::RObject >::type X(XSEXP);
    Rcpp::traits::input_parameter< arma::mat >::type y(ySEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< Rcpp::RObject >::type BInit(BInitSEXP);
    Rcpp::traits::input_parameter< bool >::type intercept(interceptSEXP);
    Rcpp::traits::input_parameter< double >::type stepSize(stepSizeSEXP);
    Rcpp::traits::input_parameter< double >::type eta(etaSEXP);
    Rcpp::traits::input_parameter< double >::type tol(tolSEXP);
    Rcpp::traits::input_parameter< long >::type maxIter(maxIterSEXP);
    Rcpp::traits::input_parameter< double >::type minStepSize(minStepSizeSEXP);
    Rcpp::traits::input_parameter< bool >::type accelerated(acceleratedSEXP);
    Rcpp::traits::input_parameter< bool >::type monotone(monotoneSEXP);
    Rcpp::traits::input_parameter< bool >::type saveData(saveDataSEXP);
    Rcpp::traits::input_parameter< bool >::type standardize(standardizeSEXP);
    rcpp_result_gen = Rcpp::wrap(SparseGroupSLOPE(X, y, lambda, alpha, BInit, intercept, stepSize, eta, tol, maxIter, minStepSize, accelerated, monotone, saveData, standardize));
    return rcpp_result_gen;
END_RCPP
}
// norm_sgs
double norm_sgs(arma::mat B, arma::vec lambda, arma::vec alpha);
RcppExport SEXP _sparse_group_slope_norm_sgs(SEXP BSEXP, SEXP lambdaSEXP, SEXP alphaSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::mat >::type B(BSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< arma::vec >::type alpha(alphaSEXP);
    rcpp_result_gen = Rcpp::wrap(norm_sgs(B, lambda, alpha));
    return rcpp_result_gen;
END_RCPP
}

RcppExport SEXP run_testthat_tests(SEXP);

static const R_CallMethodDef CallEntries[] = {
    {"_sparse_group_slope_SparseGroupSLOPE", (DL_FUNC) &_sparse_group_slope_SparseGroupSLOPE, 15},
    {"_sparse_group_slope_norm_sgs", (DL_FUNC) &_sparse_group_slope_norm_sgs, 3},
    {"run_testthat_tests", (DL_FUNC) &run_testthat_tests, 1},
    {NULL, NULL, 0}
};

RcppExport void R_init_sparse_group_slope(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}