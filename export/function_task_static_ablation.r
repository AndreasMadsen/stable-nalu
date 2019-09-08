rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
source('./_function_task_expand_name.r')
source('./_function_task_table.r')

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

t.confidence.interval = function (alpha, vec) {
  return(abs(qt((1 - alpha) / 2, length(vec) - 1)) * (sd(vec) / sqrt(length(vec))))
}

safe.median = function (vec) {
  if (length(vec) == 0) {
    return(NA)
  } else {
    return(median(vec))
  }
}

eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
  filter(simple == FALSE & parameter == 'default') %>%
  mutate(
    operation = revalue(operation, operation.full.to.short)
  ) %>%
  select(operation, input.size, overlap.ratio, subset.ratio, extrapolation.range, threshold)

name.input = '../results/function_task_static_ablation.csv'

dat = expand.name(read_csv(name.input)) %>%
  merge(eps)

dat$model = as.character(dat$model)
dat$model = ifelse(dat$regualizer == 0, paste0(dat$model, ', no $\\mathcal{R}_{bias}$'), dat$model)
dat$model = ifelse(dat$regualizer.oob == 0, paste0(dat$model, ', no W-clamp'), dat$model)

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
  summarise(
    rate.interpolation = mean(interpolation.last < threshold),
    rate.extrapolation = mean(solved),
    
    median.interpolation.solved = safe.median(interpolation.step.solved[solved]),
    mean.interpolation.solved = mean(interpolation.step.solved[solved]),
    
    median.extrapolation.solved = safe.median(extrapolation.step.solved[solved]),
    mean.extrapolation.solved = mean(extrapolation.step.solved[solved]),
    ci.extrapolation.solved = t.confidence.interval(0.95, extrapolation.step.solved[solved]),
    
    median.sparse.error.max = safe.median(sparse.error.max[solved]),
    mean.sparse.error.max = mean(sparse.error.max[solved]),
    ci.sparse.error.max = t.confidence.interval(0.95, sparse.error.max[solved]),
    
    size = n() 
  )

print(dat.last.rate)

save.table(
  dat.last.rate,
  "function-task-static-ablation",
  "Shows the success-rate for $\\mathcal{L}_{\\mathbf{W}_1, \\mathbf{W}_2} < \\mathcal{L}_{\\mathbf{W}_1^\\epsilon, \\mathbf{W}_2^*}$, at what global step the model converged at and the sparsity error for all weight matrices, with 95\\% confidence interval. The dataset is the multiplication problem with default parameters.",
  "../paper/results/function_task_static_ablation.tex",
  show.operation=FALSE,
  highlight.best=FALSE
)
