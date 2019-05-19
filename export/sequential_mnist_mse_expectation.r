rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(dplyr)

rowProds = function (X) {
  return(apply(X, 1, prod))
}

simulate.mse = function (epsilon, sigma, samples, operation, extrapolation.lengths) {
  max.length = max(extrapolation.lengths)
  X = matrix(sample(0:9, samples*max.length, replace=T), samples, max.length)
  X.noise = X + matrix(rnorm(samples*max.length, 0, sigma), samples, max.length)
  
  # Add an initialization column (either 0 or 1),  this is learned and can also have an error
  if (operation == 'sum') {
    Z = cbind(0 + epsilon, X.noise)
  } else if (operation == 'product') {
    Z = cbind(1 - epsilon, X.noise)
  }
  
  # Run Z though the weight dependent transformation 
  Z = (1 - epsilon) * Z
  if (operation == 'product') {
    Z = Z + epsilon
  }
  
  error.for.length = function (seq.length) {
    if (operation == 'sum') {
      target = rowSums(X[,1:seq.length, drop=FALSE])
      pred = rowSums(Z[,1:(seq.length+1), drop=FALSE])
    } else if (operation == 'product') {
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
      mse = error.for.length(extrapolation.length)
    )
  return(df)
}

mse = rbind(
  simulate.mse(0.1, 0.1, 100000, 'sum', c(1, 10, 100, 1000)),
  simulate.mse(0.1, 0.1, 100000, 'product', seq(1,9))
)
print(mse)
write.csv(mse, file="../results/sequential_mnist_mse_expectation.csv", row.names=F)

