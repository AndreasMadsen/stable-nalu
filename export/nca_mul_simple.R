rm(list = ls())

library(ggplot2)
library(dplyr)
library(readr)

extract.part.from.name = function (name, index) {
  flattened_split = unlist(strsplit(name, "_"))
  return(flattened_split[seq(index, length(flattened_split), 6)])
}

dat = read_csv('./data/nac_mul_simple.csv') %>%
  mutate(
    model = extract.part.from.name(name, 1),
    problem = extract.part.from.name(name, 2),
    seed = extract.part.from.name(name, 6)
  )

dat.last = dat %>%
  group_by(name) %>%
  filter(n() == 201) %>%
  summarise(
    interpolation = last(interpolation),
    extrapolation = last(extrapolation),
    sparse.error.max = last(sparse.error.max),
    sparse.error.mean = last(sparse.error.mean),
    solved = last(extrapolation) < 0.001,
    model = last(model),
    problem = last(problem),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, problem) %>%
  summarise(
    size=n(),
    rate.interpolation = mean(interpolation < 0.001),
    rate.extrapolation = mean(extrapolation < 0.001),
    mean.sparse.error.max = mean(sparse.error.max[extrapolation < 0.001]),
    mean.sparse.error.mean = mean(sparse.error.mean[extrapolation < 0.001])
  )

print(dat.last.rate)