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

eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
  filter(simple == FALSE & parameter != 'default') %>%
  mutate(
    input.size = as.integer(input.size),
    operation = revalue(operation, operation.full.to.short),
    epsilon = mse
  ) %>%
  select(operation, input.size, overlap.ratio, subset.ratio, extrapolation.range, epsilon)

plot.parameter = function(name.parameter, name.label, name.file, name.output) {
  dat = expand.name(read_csv(name.file)) %>%
    merge(eps) %>%
    mutate(
      parameter = !!as.name(name.parameter)
    )

  dat.last = dat %>%
    group_by(name) %>%
    #filter(n() == 5001) %>%
    summarise(
      epsilon = last(epsilon),
      best.model.step = best.model.step.fn(interpolation),
      interpolation.last = interpolation[best.model.step],
      extrapolation.last = extrapolation[best.model.step],
      interpolation.step.solved = first.solved.step(step, interpolation, epsilon),
      extrapolation.step.solved = first.solved.step(step, extrapolation, epsilon),
      sparse.error.max = sparse.error.max[best.model.step],
      sparse.error.mean = sparse.error.mean[best.model.step],
      solved = replace_na(extrapolation[best.model.step] < epsilon, FALSE),
      model = last(model),
      operation = last(operation),
      parameter = last(parameter),
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
      converged.at.upper = quantile(extrapolation.step.solved[solved], 0.9),
      converged.at.lower = quantile(extrapolation.step.solved[solved], 0.1),

      sparse.error.mean = mean(sparse.error.max[solved]),
      sparse.error.upper = quantile(sparse.error.max[solved], 0.9),
      sparse.error.lower = quantile(sparse.error.max[solved], 0.1)
    )

  dat.gather.mean = dat.last.rate %>%
    mutate(
      success.rate = success.rate.mean,
      converged.at = converged.at.mean,
      sparse.error = ifelse(sparse.error.mean > 0.1, NA, sparse.error.mean)
    ) %>%
    select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
    gather('key', 'mean.value', success.rate, converged.at, sparse.error)

  dat.gather.upper = dat.last.rate %>%
    mutate(
      success.rate = success.rate.upper,
      converged.at = converged.at.upper,
      sparse.error = ifelse(sparse.error.mean > 0.1, NA, sparse.error.upper)
    ) %>%
    select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
    gather('key', 'upper.value', success.rate, converged.at, sparse.error)

  dat.gather.lower = dat.last.rate %>%
    mutate(
      success.rate = success.rate.lower,
      converged.at = converged.at.lower,
      sparse.error = ifelse(sparse.error.mean > 0.1, NA, sparse.error.lower)
    ) %>%
    select(model, operation, parameter, success.rate, converged.at, sparse.error) %>%
    gather('key', 'lower.value', success.rate, converged.at, sparse.error)

  dat.gather = merge(merge(dat.gather.mean, dat.gather.upper), dat.gather.lower) %>%
    mutate(
      model=droplevels(model),
      key = factor(key, levels = c("success.rate", "converged.at", "sparse.error"))
    )

  p = ggplot(dat.gather, aes(x = parameter, colour=model)) +
    geom_point(aes(y = mean.value)) +
    geom_line(aes(y = mean.value)) +
    geom_errorbar(aes(ymin = lower.value, ymax = upper.value)) +
    scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
    xlab(name.label) +
    scale_y_continuous(name = element_blank()) +
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
}

plot.parameter('input.size', 'Input size', '../results/function_task_static_mul_input_size.csv', '../paper/results/simple_function_static_input_size.pdf')
plot.parameter('subset.ratio', 'Subset ratio', '../results/function_task_static_mul_subset.csv', '../paper/results/simple_function_static_subset.pdf')
plot.parameter('overlap.ratio', 'Overlap ratio', '../results/function_task_static_mul_overlap.csv', '../paper/results/simple_function_static_overlap.pdf')
#plot.parameter('interpolation.range', 'Interpolation range', '../results/function_task_static_mul_range.csv', '../paper/results/simple_function_static_range.pdf')
