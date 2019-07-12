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

first.solved.step = function (steps, errors, threshold) {
  index = first(which(errors < threshold))
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
  filter(operation == 'cumprod') %>%
  mutate(
    test.extrapolation.length=extrapolation.length
  ) %>%
  select(operation, test.extrapolation.length, threshold)

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
  merge(eps)

dat.last = dat %>%
  group_by(name, test.extrapolation.length) %>%
  summarise(
    threshold = last(threshold),
    best.model.step = best.model.step.fn(valid.mse),

    valid.mse.last = valid.mse[best.model.step],
    test.mnist.acc.last = test.mnist.acc[best.model.step],
    test.extrapolation.mse.last = test.extrapolation.mse[best.model.step],

    extrapolation.step.solved = first.solved.step(step, test.extrapolation.mse, threshold),

    sparse.error.max = sparse.error.max[best.model.step],
    sparse.error.mean = sparse.error.sum[best.model.step] / sparse.error.count[best.model.step],

    solved = replace_na(test.extrapolation.mse[best.model.step] < threshold, FALSE),
    model = last(model),
    operation = last(operation),
    hidden.size = last(hidden.size),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation, hidden.size, extrapolation.length) %>%
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

# dat.gather.mean = dat.last.rate %>%
#   mutate(
#     success.rate = success.rate.mean,
#     converged.at = converged.at.mean,
#     sparse.error = sparse.error.mean
#   ) %>%
#   select(model, operation, extrapolation.length, success.rate, converged.at, sparse.error) %>%
#   gather('key', 'mean.value', success.rate, converged.at, sparse.error)
# 
# dat.gather.upper = dat.last.rate %>%
#   mutate(
#     success.rate = success.rate.upper,
#     converged.at = converged.at.upper,
#     sparse.error = sparse.error.upper
#   ) %>%
#   select(model, operation, extrapolation.length, success.rate, converged.at, sparse.error) %>%
#   gather('key', 'upper.value', success.rate, converged.at, sparse.error)
# 
# dat.gather.lower = dat.last.rate %>%
#   mutate(
#     success.rate = success.rate.lower,
#     converged.at = converged.at.lower,
#     sparse.error = sparse.error.lower
#   ) %>%
#   select(model, operation, extrapolation.length, success.rate, converged.at, sparse.error) %>%
#   gather('key', 'lower.value', success.rate, converged.at, sparse.error)
# 
# 
# dat.gather = merge(merge(dat.gather.mean, dat.gather.upper), dat.gather.lower) %>%
#   mutate(
#     model=droplevels(model),
#     key = factor(key, levels = c("success.rate", "converged.at", "sparse.error"))
#   )
# 
# p = ggplot(dat.gather, aes(x = extrapolation.length, colour=model)) +
#   geom_point(aes(y = mean.value)) +
#   geom_line(aes(y = mean.value)) +
#   geom_errorbar(aes(ymin = lower.value, ymax = upper.value), width = 0.5) +
#   scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
#   scale_x_continuous(name = 'Extrapolation length', breaks=unique(dat.gather$extrapolation.length)) +
#   scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
#   facet_wrap(~ key, scales='free_y', labeller = labeller(
#     key = c(
#       success.rate = "Success rate",
#       converged.at = "Converged at",
#       sparse.error = "Sparsity error"
#     )
#   )) +
#   theme(legend.position="bottom") +
#   theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
# print(p)
# ggsave('../paper/results/sequential_mnist_prod.pdf', p, device="pdf", width = 13.968, height = 5, scale=1.4, units = "cm")

