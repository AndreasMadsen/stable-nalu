rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(plyr)
library(dplyr)
library(tidyr)
library(readr)
require(MASS)
library(KScorrect) # citation("KScorrect")

source('./_function_task_expand_name.r')
source('./_function_task_table.r')
source('./_compute_summary.r')

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
    operation = revalue(operation, operation.full.to.short),
  ) %>%
  select(operation, threshold)

dat = expand.name(
  read_csv('../results/function_task_static.csv', col_types=cols(sparse.error.max=col_double()))
) %>%
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


dat.last = read_csv('../paper/results/sft-last.csv')

dat.last.rate = dat.last %>%
  group_by(model, operation) %>%
  group_modify(compute.summary) %>%
  ungroup()

dat.last.dist.paramaters = merge(dat.last, dat.last.rate) %>%
  group_by(model, operation) %>%
  group_modify(function (df, ...) {
    
    if (with(df, length(unique(extrapolation.step.solved[solved]))) > 10) {
      converged.at.p.value = with(df, LcKS(
        extrapolation.step.solved[solved], "pgamma",
        parallel = TRUE,#,
        #(last(converged.at.mean) * last(converged.at.mean)) / last(converged.at.sigma.2), 
        #last(converged.at.mean) / last(converged.at.sigma.2)
      ))$p.value
    } else {
      converged.at.p.value = NA
    }
    
    if (sum(df$solved) > 1) {
      sparse.error.p.value = with(df, ks.test(
        sparse.error.max[solved], "pbeta",
        last(sparse.error.mean) * 2 * last(sparse.error.v),
        (1 - last(sparse.error.mean) * 2) * last(sparse.error.v)
      ))$p.value
    }
    
    return(with(df, data.frame(
      converged.at.sigma.2 = last(converged.at.sigma.2),
      converged.at.p.value = converged.at.p.value,
      converged.at.mean = last(converged.at.mean),
      converged.at.lower = last(converged.at.lower),
      converged.at.upper = last(converged.at.upper),
      
      sparse.error.v = last(sparse.error.v),
      sparse.error.mean = last(sparse.error.mean),
      sparse.error.lower = last(sparse.error.lower),
      sparse.error.upper = last(sparse.error.upper)
    )))
  })
  