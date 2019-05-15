rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(xtable)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(kableExtra)

extract.part.from.name = function (name, index) {
  flattened_split = unlist(strsplit(name, "_"))
  return(flattened_split[seq(index, length(flattened_split), 6)])
}

best.model.step.fn = function (errors) {
  best.step = max(length(errors) - best.range, 0) + which.min(tail(errors, best.range))
  if (length(best.step) == 0) {
    return(length(errors))
  } else {
    return(best.step)
  }
}

eps = 0.2
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

operation.full.to.short = c(
  'mul'='$a \\cdot b$'
)

dat = read_csv('../results/simple_mul.csv') %>%
  mutate(
    model = revalue(extract.part.from.name(name, 1), model.full.to.short),
    operation = revalue(extract.part.from.name(name, 2), operation.full.to.short),
    seed = extract.part.from.name(name, 6)
  )

dat.last = dat %>%
  group_by(name) %>%
  #filter(n() == 201) %>%
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
    rate.interpolation = mean(interpolation.last < eps),
    rate.extrapolation = mean(extrapolation.last < eps),
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

