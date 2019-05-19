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

best.range = 5

best.model.step.fn = function (errors) {
  best.step = max(length(errors) - best.range, 0) + which.min(tail(errors, best.range))
  if (length(best.step) == 0) {
    return(length(errors))
  } else {
    return(best.step)
  }
}

first.solved.step = function (steps, errors, epsilon) {
  index = first(which(errors < epsilon))
  if (is.na(index)) {
    return(NA)
  } else {
    return(steps[index])
  }
}

safe.quantile = function (vec, prop) {
  if (length(vec) <= 1) {
    return(NA)
  } else if (length(vec) <= 3) {
    return(ifelse(prop < 0.5, min(vec), max(vec)))
  } else {
    return(median(vec, prop))
  }
} 

eps = read_csv('../results/sequential_mnist_mse_expectation.csv') %>%
  mutate(
    epsilon = mse
  ) %>%
  select(operation, extrapolation.length, epsilon)

dat = expand.name(read_csv('../results/sequential_mnist_sum.csv')) %>%
  gather(
    key="extrapolation.length", value="extrapolation.error",
    loss.valid.extrapolation.1, loss.valid.extrapolation.10, loss.valid.extrapolation.100, loss.valid.extrapolation.1000
  ) %>%
  mutate(
    validation=loss.valid.validation,
    mnist=loss.valid.mnist_classification,
    extrapolation.length=as.integer(substring(extrapolation.length, 26))
  ) %>%
  merge(eps)

dat.last = dat %>%
  group_by(name, extrapolation.length) %>%
  summarise(
    epsilon = last(epsilon),
    best.model.step = best.model.step.fn(validation),

    validation.last = validation[best.model.step],
    mnist.last = mnist[best.model.step],
    extrapolation.error.last = extrapolation.error[best.model.step],

    extrapolation.step.solved = first.solved.step(step, extrapolation.error, epsilon),
    
    sparse.error.max = sparse.error.max[best.model.step],
    sparse.error.mean = sparse.error.sum[best.model.step] / sparse.error.count[best.model.step],
    
    solved = replace_na(extrapolation.error[best.model.step] < epsilon, FALSE),
    model = last(model),
    operation = last(operation),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation, extrapolation.length) %>%
  summarise(
    size=n(),
    success.rate.mean = mean(solved),
    success.rate.upper = NA,
    success.rate.lower = NA,
    
    converged.at.mean = mean(extrapolation.step.solved[solved]),
    converged.at.upper = safe.quantile(extrapolation.step.solved[solved], 0.9),
    converged.at.lower = safe.quantile(extrapolation.step.solved[solved], 0.1),
    
    sparse.error.mean = mean(sparse.error.max[solved]),
    sparse.error.upper = safe.quantile(sparse.error.max[solved], 0.9),
    sparse.error.lower = safe.quantile(sparse.error.max[solved], 0.1)
  )

dat.gather.mean = dat.last.rate %>%
  mutate(
    success.rate = success.rate.mean,
    converged.at = converged.at.mean,
    sparse.error = sparse.error.mean
  ) %>%
  select(model, operation, extrapolation.length, success.rate, converged.at, sparse.error) %>%
  gather('key', 'mean.value', success.rate, converged.at, sparse.error)

dat.gather.upper = dat.last.rate %>%
  mutate(
    success.rate = success.rate.upper,
    converged.at = converged.at.upper,
    sparse.error = sparse.error.upper
  ) %>%
  select(model, operation, extrapolation.length, success.rate, converged.at, sparse.error) %>%
  gather('key', 'upper.value', success.rate, converged.at, sparse.error)

dat.gather.lower = dat.last.rate %>%
  mutate(
    success.rate = success.rate.lower,
    converged.at = converged.at.lower,
    sparse.error = sparse.error.lower
  ) %>%
  select(model, operation, extrapolation.length, success.rate, converged.at, sparse.error) %>%
  gather('key', 'lower.value', success.rate, converged.at, sparse.error)


dat.gather = merge(merge(dat.gather.mean, dat.gather.upper), dat.gather.lower) %>%
  mutate(
    model=droplevels(model),
    key = factor(key, levels = c("success.rate", "converged.at", "sparse.error"))
  )

p = ggplot(dat.gather, aes(x = extrapolation.length, colour=model)) +
  geom_point(aes(y = mean.value)) +
  geom_line(aes(y = mean.value)) +
  geom_errorbar(aes(ymin = lower.value, ymax = upper.value), width = 0.5) +
  scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
  scale_x_continuous(name = 'Extrapolation length', trans='log10', breaks=unique(dat.gather$extrapolation.length)) +
  scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
  facet_wrap(~ key, scales='free_y', labeller = labeller(
    key = c(
      success.rate = "Success rate",
      converged.at = "Converged at",
      sparse.error = "Sparsity error"
    )
  )) +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
print(p)
ggsave('../paper/results/sequential_mnist_sum.pdf', p, device="pdf", width = 13.968, height = 5, scale=1.4, units = "cm")

