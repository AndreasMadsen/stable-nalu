rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(xtable)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(kableExtra)
source('./_function_task_expand_name.r')

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

best.range = 10

model.full.to.short = c(
  'linear'='linear',
  'relu6'='ReLU6',
  'nac'='${\\mathrm{NAC}_{+}}$',
  'nac-nac-n'='${\\mathrm{NAC}_\\bullet}$',
  'nalu'='NALU',
  'reregualizedlinearnac'='NAU',
  'reregualizedlinearnac-nac-m'='NMU'
)

eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
  filter(simple == TRUE & operation == 'o-mul') %>%
  mutate(
    input.size = as.integer(input.size),
    operation = revalue(operation, operation.full.to.short),
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
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation) %>%
  summarise(
    rate.interpolation = mean(interpolation.last < epsilon),
    rate.extrapolation = mean(solved),
    interpolation.solved = mean(interpolation.step.solved[solved]),
    extrapolation.solved = mean(extrapolation.step.solved[solved]),
    mean.sparse.error.max = mean(sparse.error.max[solved]),
    mean.sparse.error.mean = mean(sparse.error.mean[solved]),
    size = n()
  )

print(dat.last.rate)


latex.scientific = function (d) {
  return(sub("NaN|NA", "---",
             sanitize.numbers(format(as.numeric(d), scientific = TRUE, digits=2),
                              type = "latex", math.style.exponents = TRUE)))
}

latex.digit = function (d) {
  return(sub("\\$(NaN|NA)\\$", "---", sprintf("$%.0f$", as.numeric(d))))
}

latex.rate = function (d) {
  return(sub("\\$(NaN|NA)\\\\%\\$", "---", sprintf("$%.0f\\%%$", as.numeric(d) * 100)))
}

dat.last.rate %>%
  mutate(
    success.rate = latex.rate(rate.extrapolation),
    converged.at = latex.digit(extrapolation.solved),
    sparse.error = latex.scientific(mean.sparse.error.max)
  ) %>%
  select(operation, model, success.rate, converged.at, sparse.error) %>%
  arrange(operation, model) %>%
  kable(
    "latex", booktabs=T, align = c('r', 'r', 'l', 'l', 'l'), escape=F, label="very-simple-function-results",
    caption="Shows the success-rate for extrapolation < $\\epsilon$, at what global step the model converged at, and the sparsity error for all weight matrices.",
    col.names = c("Operation",
                  "Model",
                  "Success rate",
                  "Converged at",
                  "Sparsity error")
  ) %>%
  kable_styling(latex_options=c('HOLD_position')) %>%
  collapse_rows(columns = c(1,2), latex_hline = "major") %>%
  write("../paper/results/simple_mul.tex")

