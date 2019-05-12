rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(xtable)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(kableExtra)

xtabs.data.first = function (data, formular, ...) {
  return(xtabs(formular, data, ...))
}

extract.part.from.name = function (name, index) {
  flattened_split = unlist(strsplit(name, "_"))
  return(flattened_split[seq(index, length(flattened_split), 6)])
}

eps = 0.2
median_range = 10

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
    interpolation.last = median(tail(interpolation, median_range)),
    extrapolation.last = median(tail(extrapolation, median_range)),
    interpolation.index.solved = first(which(interpolation < eps)) * 100,
    extrapolation.index.solved = first(which(extrapolation < eps)) * 100,
    sparse.error.max = median(tail(sparse.error.max, median_range)),
    sparse.error.mean = median(tail(sparse.error.mean, median_range)),
    solved = median(tail(extrapolation, median_range)) < eps,
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
    interpolation.solved = mean(interpolation.index.solved[solved]),
    extrapolation.solved = mean(extrapolation.index.solved[solved]),
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
    caption="Shows the sucess-rate for extrapolation < $\\epsilon$, at what global step the model converged at, and the sparse error for all weight matrices.",
    col.names = c("Operation",
                  "Model",
                  "Success Rate",
                  "Converged at",
                  "Sparse error")
  ) %>%
  kable_styling(latex_options=c('HOLD_position')) %>%
  collapse_rows(columns = c(1,2), latex_hline = "major") %>%
  write("../paper/results/simple_mul.tex")

