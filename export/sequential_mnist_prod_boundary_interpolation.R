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

eps = read_csv('../results/sequential_mnist_mse_expectation.csv') %>%
  filter(operation == 'cumprod') %>%
  mutate(
    test.extrapolation.length=extrapolation.length
  ) %>%
  select(operation, test.extrapolation.length, threshold)

dat = expand.name(read_csv('../results/sequential_mnist_prod_interpolation.csv')) %>%
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
  select(-starts_with("loss.valid")) %>%
  merge(eps)

dat.last = dat %>%
  group_by(name, test.extrapolation.length) %>%
  summarise(
    threshold = last(threshold),
    best.model.step = best.model.step.fn(valid.mse),

    valid.mse.last = valid.mse[best.model.step],
    test.mnist.acc.last = test.mnist.acc[best.model.step],
    test.extrapolation.mse.last = test.extrapolation.mse[best.model.step],

    extrapolation.epoch.solved = first.solved.step(epoch, test.extrapolation.mse, threshold),

    sparse.error.max = sparse.error.max[best.model.step],
    sparse.error.mean = sparse.error.sum[best.model.step] / sparse.error.count[best.model.step],

    solved = replace_na(test.extrapolation.mse[best.model.step] < threshold, FALSE),
    model = last(model),
    operation = last(operation),
    interpolation.length = last(interpolation.length),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation, interpolation.length, test.extrapolation.length) %>%
  summarise(
    size=n(),
    success.rate.mean = mean(solved) * 100,
    success.rate.upper = NA,
    success.rate.lower = NA,
    
    converged.at.mean = mean(extrapolation.epoch.solved[solved]),
    converged.at.upper = converged.at.mean + safe.interval(0.95, extrapolation.epoch.solved[solved]),
    converged.at.lower = converged.at.mean - safe.interval(0.95, extrapolation.epoch.solved[solved]),
    
    sparse.error.mean = mean(sparse.error.max[solved]),
    sparse.error.upper = sparse.error.mean + safe.interval(0.95, sparse.error.max[solved]),
    sparse.error.lower = sparse.error.mean - safe.interval(0.95, sparse.error.max[solved])
  )

dat.gather.mean = dat.last.rate %>%
  mutate(
    success.rate = success.rate.mean,
    converged.at = converged.at.mean,
    sparse.error = sparse.error.mean
  ) %>%
  select(model, operation, interpolation.length, test.extrapolation.length, success.rate, converged.at, sparse.error) %>%
  gather('key', 'mean.value', success.rate, converged.at, sparse.error)

dat.gather.upper = dat.last.rate %>%
  mutate(
    success.rate = success.rate.upper,
    converged.at = converged.at.upper,
    sparse.error = sparse.error.upper
  ) %>%
  select(model, operation, interpolation.length, test.extrapolation.length, success.rate, converged.at, sparse.error) %>%
  gather('key', 'upper.value', success.rate, converged.at, sparse.error)

dat.gather.lower = dat.last.rate %>%
  mutate(
    success.rate = success.rate.lower,
    converged.at = converged.at.lower,
    sparse.error = sparse.error.lower
  ) %>%
  select(model, operation, interpolation.length, test.extrapolation.length, success.rate, converged.at, sparse.error) %>%
  gather('key', 'lower.value', success.rate, converged.at, sparse.error)


dat.gather = merge(merge(dat.gather.mean, dat.gather.upper), dat.gather.lower) %>%
  mutate(
    model=droplevels(model),
    lower.value=as.numeric(lower.value),
    upper.value=as.numeric(upper.value),
    key = factor(key, levels = c("success.rate", "converged.at", "sparse.error"))
  )

p = ggplot(dat.gather, aes(x = interpolation.length, colour=model)) +
  geom_point(aes(y = mean.value)) +
  geom_line(aes(y = mean.value)) +
  geom_errorbar(aes(ymin = lower.value, ymax = upper.value), width = 0.5) +
  scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
  scale_x_continuous(name = 'Interpolation length') +
  scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
  facet_wrap(~ key + test.extrapolation.length, scales='free', labeller = labeller(
    key = c(
      success.rate = "Success rate in %",
      converged.at = "Solved at epoch",
      sparse.error = "Sparsity error"
    )
  )) +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
print(p)
# ggsave('../paper/results/sequential_mnist_prod.pdf', p, device="pdf", width = 13.968, height = 5, scale=1.4, units = "cm")

