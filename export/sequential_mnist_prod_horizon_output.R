rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(xtable)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(kableExtra)
source('./_sequential_mnist_expand_name.r')

best.range = 1000

best.model.step.fn = function (errors) {
  best.step = max(length(errors) - best.range, 0) + which.min(tail(errors, best.range))
  if (length(best.step) == 0) {
    return(length(errors))
  } else {
    return(best.step)
  }
}

first.solved.step = function (steps, errors, threshold) {
  index = first(which(errors < threshold))
  if (is.na(index)) {
    return(NA)
  } else {
    return(steps[index])
  }
}

safe.interval = function (alpha, vec) {
  if (length(vec) <= 1) {
    return(NA)
  }
  
  return(abs(qt((1 - alpha) / 2, length(vec) - 1)) * (sd(vec) / sqrt(length(vec))))
}

dat = expand.name(read_csv('../results/sequential_mnist_prod_outputs.csv')) %>%
  gather(
    key="test.extrapolation.length", value="test.extrapolation.mse",
    loss.test.extrapolation.1.mse, loss.test.extrapolation.2.mse,
    loss.test.extrapolation.3.mse, loss.test.extrapolation.4.mse,
    loss.test.extrapolation.5.mse, loss.test.extrapolation.6.mse,
    loss.test.extrapolation.7.mse, loss.test.extrapolation.8.mse,
    loss.test.extrapolation.9.mse
  ) %>%
  mutate(
    valid.mse=loss.valid.validation.mse,
    valid.acc.all=loss.valid.validation.acc.all,
    valid.acc.last=loss.valid.validation.acc.last,
    valid.mnist.acc=loss.valid.mnist.acc,
    test.mnist.acc=loss.test.mnist.acc,
    test.extrapolation.length=as.integer(substring(test.extrapolation.length, 25, 25))
  ) %>%
  select(-starts_with("loss.test")) %>%
  select(-starts_with("loss.valid"))

dat.last = dat %>%
  group_by(name, test.extrapolation.length) %>%
  summarise(
    best.model.step = best.model.step.fn(valid.mse),

    test.mse = test.extrapolation.mse[best.model.step],
    recursive.weight = recursive.weight[best.model.step],

    model = last(model),
    model.type = ifelse(as.character(model) %in% c('$\\mathrm{NAC}_{\\bullet,\\sigma}$', '$\\mathrm{NAC}_{\\bullet}$'), 'NAC', as.character(model)),
    operation = last(operation),
    hidden.size = last(hidden.size),
    seed = last(seed),
    size = n()
  ) %>%
  filter(test.extrapolation.length == 1 & hidden.size == 1) %>%
  select(-test.extrapolation.length, -hidden.size)
dat.last$model.type = as.factor(dat.last$model.type)

num.samples = 100
digits = 1:9
sigma.2 = 0.25

obs = cbind(dat.last[rep(1:nrow(dat.last), num.samples), ], obs.index = rep(1:num.samples, each = nrow(dat.last)))
obs$prev.value = 1
obs$prev.ref.value = 1
obs$error = 0
 
###
dat.errors.list = list()
for (i in 1:1000) {
  obs$mnist.input = NULL
  obs$mnist.error = NULL
  samples = data.frame(
    obs.index = 1:num.samples,
    mnist.input = sample(digits, num.samples, replace=T),
    mnist.error = rnorm(num.samples, 0, 1)
  )
  obs = merge(obs, samples)
  obs[,'mnist.error'] = obs$mnist.input * sqrt(obs$test.mse)
  
  subset.select = obs$model.type == 'NAC'
  obs.model.subset = obs[subset.select,]
  obs[subset.select,'prev.value'] = obs.model.subset$prev.value ** obs.model.subset$recursive.weight * (obs.model.subset$mnist.input + obs.model.subset$mnist.error)
  
  subset.select = obs$model.type == 'NMU'
  obs.model.subset = obs[subset.select,]
  obs[subset.select,'prev.value'] = (obs.model.subset$recursive.weight * obs.model.subset$prev.value + 1 - obs.model.subset$recursive.weight) * (obs.model.subset$mnist.input + obs.model.subset$mnist.error)
  
  obs[,'prev.ref.value'] = obs$prev.ref.value * obs$mnist.input
  
  if (i %% 10 == 0) {
    errors = obs %>%
      group_by(name) %>%
      summarize(
        error.mse = mean((prev.ref.value - prev.value) ** 2),
        extrapolation.length = i
      )
    dat.errors.list = c(dat.errors.list, list(merge(dat.last, errors)))
  }
}

dat.errors = bind_rows(dat.errors.list) %>%
  group_by(model, extrapolation.length) %>%
  summarize(
    error.mse = median(error.mse)
  )

