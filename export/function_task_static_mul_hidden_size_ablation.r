rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
source('./_function_task_expand_name.r')
source('./_compute_summary.r')
source('./_plot_parameter.r')

best.range = 5000

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

eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
  filter(simple == FALSE & parameter == 'default') %>%
  mutate(
    operation = revalue(operation, operation.full.to.short)
  ) %>%
  select(operation, input.size, overlap.ratio, subset.ratio, extrapolation.range, threshold)

name.parameter = 'hidden.size'
plot.label = 'Hidden size'
plot.x.breaks = c(2, 4, 6, 8, 10)
name.input = '../results/function_task_static_mul_hidden_size_ablation.csv'
name.output = '../paper/results/simple_function_static_mul_hidden_size_ablation.pdf'

dat = expand.name(read_csv(name.input)) %>%
  merge(eps) %>%
  mutate(
    parameter = !!as.name(name.parameter),
    model = as.factor(paste0(
      as.character(model),
      ifelse(regualizer == 0, ', no $\\mathcal{R}_{sparse}$', ''),
      ifelse(regualizer.oob == 0, ', no W-clamp', '')
    ))
  )

dat.last = dat %>%
  group_by(name, parameter) %>%
  #filter(n() == 201) %>%
  summarise(
    threshold = last(threshold),
    best.model.step = best.model.step.fn(metric.valid.interpolation),
    interpolation.last = metric.valid.interpolation[best.model.step],
    extrapolation.last = metric.test.extrapolation[best.model.step],
    interpolation.step.solved = first.solved.step(step, metric.valid.interpolation, threshold),
    extrapolation.step.solved = first.solved.step(step, metric.test.extrapolation, threshold),
    sparse.error.max = sparse.error.max[best.model.step],
    solved = replace_na(metric.test.extrapolation[best.model.step] < threshold, FALSE),
    model = last(model),
    operation = last(operation),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation, parameter) %>%
  group_modify(compute.summary)

p = plot.parameter(dat.last.rate, plot.label, plot.x.breaks)
print(p)
ggsave(name.output, p, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")
