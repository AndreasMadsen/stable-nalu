rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(xtable)
source('./_function_task_expand_name.r')

best.range = 100

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

eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
  filter(simple == FALSE & parameter != 'default') %>%
  mutate(
    input.size = as.integer(input.size),
    operation = revalue(operation, operation.full.to.short),
    epsilon = mse
  ) %>%
  select(operation, input.size, overlap.ratio, subset.ratio, extrapolation.range, epsilon)

name.parameter = 'regualizer'
name.label = 'Sparse regualizer'
name.file = '../results/function_task_static_regualization.csv'
name.output = '../paper/results/function_task_static_regualization.pdf'

dat = expand.name(read_csv(name.file)) %>%
  merge(eps) %>%
  mutate(
    parameter = !!as.name(name.parameter)
  )

dat.last = dat %>%
  group_by(name) %>%
  #filter(n() == 201) %>%
  summarise(
    epsilon = last(epsilon),
    best.model.step = best.model.step.fn(loss.valid.interpolation),
    interpolation.last = loss.valid.interpolation[best.model.step],
    extrapolation.last = loss.valid.extrapolation[best.model.step],
    interpolation.step.solved = first.solved.step(step, loss.valid.interpolation, epsilon),
    extrapolation.step.solved = first.solved.step(step, loss.valid.extrapolation, epsilon),
    sparse.error.max = sparse.error.max[best.model.step],
    sparse.error.mean = sparse.error.sum[best.model.step] / sparse.error.count[best.model.step],
    solved = replace_na(loss.valid.extrapolation[best.model.step] < epsilon, FALSE),
    parameter = last(parameter),
    model = last(model),
    operation = last(operation),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation, parameter) %>%
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
  select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
  gather('key', 'mean.value', success.rate, converged.at, sparse.error)

dat.gather.upper = dat.last.rate %>%
  mutate(
    success.rate = success.rate.upper,
    converged.at = converged.at.upper,
    sparse.error = sparse.error.upper
  ) %>%
  select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
  gather('key', 'upper.value', success.rate, converged.at, sparse.error)

dat.gather.lower = dat.last.rate %>%
  mutate(
    success.rate = success.rate.lower,
    converged.at = converged.at.lower,
    sparse.error = sparse.error.lower
  ) %>%
  select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
  gather('key', 'lower.value', success.rate, converged.at, sparse.error)

dat.gather = merge(merge(dat.gather.mean, dat.gather.upper), dat.gather.lower) %>%
  mutate(
    model=droplevels(model),
    key = factor(key, levels = c("success.rate", "converged.at", "sparse.error"))
  )

p = ggplot(dat.gather, aes(x = as.factor(parameter), colour=model, group=model)) +
  geom_point(aes(y = mean.value)) +
  geom_line(aes(y = mean.value)) +
  geom_errorbar(aes(ymin = lower.value, ymax = upper.value)) +
  scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
  xlab(name.label) +
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
ggsave(name.output, p, device="pdf", width = 13.968, height = 5, scale=1.4, units = "cm")
