rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)
library(tidyr)
library(readr)
source('./_function_task_expand_name_v2.r')
source('./_function_task_table.r')

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

safe.median = function (vec) {
  if (length(vec) == 0) {
    return(NA)
  } else {
    return(median(vec))
  }
}

t.confidence.interval = function (alpha, vec) {
  return(abs(qt((1 - alpha) / 2, length(vec) - 1)) * (sd(vec) / sqrt(length(vec))))
}

best.range = 100

eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
  filter(simple == TRUE & operation == 'o-mul') %>%
  mutate(
    input.size = as.integer(input.size),
    operation = revalue(paste0("op", substring(operation, 2)), operation.full.to.short),
    epsilon = mse
  ) %>%
  select(operation, epsilon)


dat = expand.name(read_csv('../results/simple_mul.csv')) %>%
  merge(eps)

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
    model = last(model),
    operation = last(operation),
    oob.control = last(oob.control),
    regualizer.shape = last(regualizer.shape),
    epsilon.zero = last(epsilon.zero),
    operation = last(operation),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation, oob.control, regualizer.shape, epsilon.zero) %>%
  summarise(
    rate.interpolation = mean(interpolation.last < epsilon),
    rate.extrapolation = mean(solved),
    
    median.interpolation.solved = safe.median(interpolation.step.solved[solved]),
    mean.interpolation.solved = mean(interpolation.step.solved[solved]),

    median.extrapolation.solved = safe.median(extrapolation.step.solved[solved]),
    mean.extrapolation.solved = mean(extrapolation.step.solved[solved]),
    ci.extrapolation.solved = t.confidence.interval(0.95, extrapolation.step.solved[solved]),
    
    median.sparse.error.max = safe.median(sparse.error.max[solved]),
    mean.sparse.error.max = mean(sparse.error.max[solved]),
    ci.sparse.error.max = t.confidence.interval(0.95, sparse.error.max[solved]),
    
    mean.sparse.error.mean = mean(sparse.error.mean[solved]),
    size = n()
  )

print(dat.last.rate)

save.table(
  dat.last.rate,
  "very-simple-function-results",
  "Shows the success-rate for $\\mathcal{L}_{\\mathbf{W}_1, \\mathbf{W}_2} < \\mathcal{L}_{\\mathbf{W}_1^\\epsilon, \\mathbf{W}_2^*}$, at what global step the model converged at and the sparsity error for all weight matrices, with 95\\% confidence interval. Best result is highlighed without considering significance.",
  "../paper/results/simple_mul.tex"
)

