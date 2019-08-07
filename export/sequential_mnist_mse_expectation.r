rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)

apply.rowwise = function (X, fn) {
  return(t(apply(X, 1, fn)))
}

simulate.mse = function (epsilon, sigma.2, samples, operation, digits, extrapolation.lengths) {
  max.length = max(extrapolation.lengths)
  X = matrix(sample(digits, samples*max.length, replace=T), samples, max.length)
  X.noise = X + matrix(rnorm(samples*max.length, 0, sqrt(sigma.2)), samples, max.length)
  
  # Should a epsilon transformation be done here?
  Z = X.noise
  
  error.for.length = function (seq.length) {
    X.slice = X[,1:seq.length, drop=FALSE]
    Z.slice = Z[,1:seq.length, drop=FALSE]
    
    if (operation == 'cumsum') {
      target = apply.rowwise(X.slice, cumsum)
      pred = apply.rowwise(Z.slice, cumsum)
    } else if (operation == 'cumprod') {
      target = apply.rowwise(X.slice, cumprod)
      pred = apply.rowwise(Z.slice, cumprod)
    }

    errors = mean((pred - target)**2)
  }
  
  df = data.frame(
      operation=operation,
      extrapolation.length=extrapolation.lengths
  ) %>%
    rowwise() %>%
    mutate(
      threshold = error.for.length(extrapolation.length)
    )
  return(df)
}

mse = rbind(
 #simulate.mse(NA, 0.25, 100000, 'cumsum', 0:9, c(1, 10, 100, 1000)), #0.25
 simulate.mse(NA, 0.125, 100000, 'cumprod', 1:9, seq(1,9)) # 0.25
)
print(mse)
write.csv(mse, file="../results/sequential_mnist_mse_expectation.csv", row.names=F)

