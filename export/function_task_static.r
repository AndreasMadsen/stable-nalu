rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(xtable)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(kableExtra)
source('./_expand_name.r')

eps = 0.2
best.range = 100

best.model.step.fn = function (errors) {
  best.step = max(length(errors) - best.range, 0) + which.min(tail(errors, best.range))
  if (length(best.step) == 0) {
    return(length(errors))
  } else {
    return(best.step)
  }
}

dat = expand.name(read_csv('../results/function_task_static.csv'))

dat.last = dat %>%
  group_by(name) %>%
  #filter(n() == 5001) %>%
  summarise(
    best.model.step = best.model.step.fn(interpolation),
    interpolation.last = interpolation[best.model.step],
    extrapolation.last = extrapolation[best.model.step],
    interpolation.step.solved = first(which(interpolation < eps)) * 1000,
    extrapolation.step.solved = first(which(extrapolation < eps)) * 1000,
    sparse.error.max = sparse.error.max[best.model.step],
    sparse.error.mean = sparse.error.mean[best.model.step],
    solved = replace_na(extrapolation[best.model.step] < eps, FALSE),
    model = last(model),
    operation = last(operation),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation) %>%
  summarise(
    size=n(),
    rate.interpolation = mean(interpolation.last < eps),
    rate.extrapolation = mean(extrapolation.last < eps),
    interpolation.solved = mean(interpolation.step.solved[solved]),
    extrapolation.solved = mean(extrapolation.step.solved[solved]),
    mean.sparse.error.max = mean(sparse.error.max[solved]),
    mean.sparse.error.mean = mean(sparse.error.mean[solved])
  )

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
  filter(
    (operation %in% c('$a + b$', '$a - b$') & model %in% c('Linear', 'NAU', '$\\mathrm{NAC}_{+}$', 'NALU')) |
    (operation %in% c('${a \\cdot b}$') & model %in% c('Linear', 'NMU', '$\\mathrm{NAC}_{\\bullet}$', 'NALU'))
  ) %>%
  arrange(operation, model) %>%
  kable(
    "latex", booktabs=T, align = c('r', 'r', 'l', 'l', 'l'), escape=F, label="function-task-static-defaults",
    caption="Shows the success-rate for extrapolation < $\\epsilon$, at what global step the model converged at, and the sparsity error for all weight matrices.",
    col.names = c("Operation",
                  "Model",
                  "Success rate",
                  "Converged at",
                  "Sparsity error")
  ) %>%
  kable_styling(latex_options=c('HOLD_position')) %>%
  collapse_rows(columns = c(1,2), latex_hline = "major") %>%
  write("../paper/results/function_task_static.tex")

