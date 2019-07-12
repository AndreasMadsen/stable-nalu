rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)

rowProds = function (X) {
  return(apply(X, 1, prod))
}

simulate.mse = function (epsilon, sigma, samples, operation, digits, extrapolation.lengths) {
  max.length = max(extrapolation.lengths)
  X = matrix(sample(digits, samples*max.length, replace=T), samples, max.length)
  X.noise = X + matrix(rnorm(samples*max.length, 0, sigma), samples, max.length)
  
  # Add an initialization column (either 0 or 1),  this is learned and can also have an error
  if (operation == 'cumsum') {
    Z = cbind(0 + epsilon, X.noise)
  } else if (operation == 'cumprod') {
    Z = cbind(1 - epsilon, X.noise)
  }
  
  # Run Z though the weight dependent transformation 
  Z = (1 - epsilon) * Z
  if (operation == 'cumprod') {
    Z = Z + epsilon
  }
  
  error.for.length = function (seq.length) {
    if (operation == 'cumsum') {
      target = rowSums(X[,1:seq.length, drop=FALSE])
      pred = rowSums(Z[,1:(seq.length+1), drop=FALSE])
    } else if (operation == 'cumprod') {
      target = rowProds(X[,1:seq.length, drop=FALSE])
      pred = rowProds(Z[,1:(seq.length+1), drop=FALSE])
    } 
    errors = mean((pred - target)**2)
  }
  
  df = data.frame(
      operation=operation, extrapolation.length=extrapolation.lengths
  ) %>%
    rowwise() %>%
    mutate(
      threshold = error.for.length(extrapolation.length)
    )
  return(df)
}

# TODO: make cummucative
mse = rbind(
  simulate.mse(0.0001, 0.01, 100000, 'cumsum', 0:9, c(1, 10, 100, 1000)),
  simulate.mse(0.0001, 0.01, 100000, 'cumprod', 1:9, seq(1,9))
)
print(mse)
write.csv(mse, file="../results/sequential_mnist_mse_expectation.csv", row.names=F)

