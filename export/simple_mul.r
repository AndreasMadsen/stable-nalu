rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(dplyr)
library(readr)

extract.part.from.name = function (name, index) {
  flattened_split = unlist(strsplit(name, "_"))
  return(flattened_split[seq(index, length(flattened_split), 6)])
}

eps = 0.2
median_range = 10

dat = read_csv('../results/simple_mul.csv') %>%
  mutate(
    model = extract.part.from.name(name, 1),
    operation = extract.part.from.name(name, 2),
    seed = extract.part.from.name(name, 6)
  )

dat.last = dat %>%
  group_by(name) %>%
  #filter(n() == 201) %>%
  summarise(
    interpolation.last = median(tail(interpolation, median_range)),
    extrapolation.last = median(tail(extrapolation, median_range)),
    interpolation.index.solved = first(which(interpolation < eps)) * 100,
    extrapolation.index.solved = first(which(extrapolation < eps)) * 100,
    sparse.error.max = median(tail(sparse.error.max, median_range)),
    sparse.error.mean = median(tail(sparse.error.mean, median_range)),
    solved = median(tail(extrapolation, median_range)) < eps,
    model = last(model),
    operation = last(operation),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation) %>%
  summarise(
    size=n(),
    rate.interpolation = mean(interpolation.last < eps),
    rate.extrapolation = mean(extrapolation.last < eps),
    interpolation.solved = mean(interpolation.index.solved[solved]),
    extrapolation.solved = mean(extrapolation.index.solved[solved]),
    mean.sparse.error.max = mean(sparse.error.max[solved]),
    mean.sparse.error.mean = mean(sparse.error.mean[solved])
  )

print(dat.last.rate)
