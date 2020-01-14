rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)
library(tidyr)
library(readr)
source('./_function_task_expand_name.r')
source('./_function_task_table.r')
source('./_compute_summary.r')

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

best.range = 5000

eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
  filter(simple == TRUE & operation == 'op-mul') %>%
  mutate(
    operation=revalue(operation, operation.full.to.short)
  ) %>%
  select(operation, threshold)


dat = expand.name(read_csv('../results/simple_mul.csv')) %>%
  merge(eps)

dat.last = dat %>%
  group_by(name) %>%
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
  group_by(model, operation) %>%
  group_modify(compute.summary) %>%
  ungroup()

print(dat.last.rate)

save.table(
  dat.last.rate %>% filter(
    (operation %in% c('$\\bm{\\times}$') & model %in% c('Linear', 'NMU', '$\\mathrm{NAC}_{\\bullet}$', 'NALU'))
  ),
  "very-simple-function-results",
  "Comparison of the success-rate, when the model converged, and the sparsity error for all weight matrices, with 95\% confidence interval on the $t = (x_1 + x_2) \cdot (x_1 + x_2 + x_3 + x_4)$ task. Each value is a summary of 100 different seeds.",
  "../paper/results/simple_mul.tex"
)

