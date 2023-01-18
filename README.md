# Sparse group SLOPE

Implementation of [Sparse Group SLOPE for multinomial regression](https://arxiv.org/abs/2204.06264).


# Installation

```{r}
# install.packages("remotes")
remotes::install_github("tmrlvi/sparse.group.slope")
```

# Usage

```{r}
library(sparse.group.slope)
# labels are expected in one hot encoding
model <- SparseGroupSLOPE(X, y = diag(L)[y,], lambda = lambda, alpha = kappa, maxIter = 1000)
pred <- apply(predict.sgs(model, X), 1, which.max)])
```
