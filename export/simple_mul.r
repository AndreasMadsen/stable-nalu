rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(dplyr)
library(readr)

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

dat = read_csv('../results/simple_mul.csv') %>%
  mutate(
    model = revalue(extract.part.from.name(name, 1), model.full.to.short),
    operation = extract.part.from.name(name, 2),
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
    size=n(),
    rate.interpolation = mean(interpolation.last < eps),
    rate.extrapolation = mean(extrapolation.last < eps),
    interpolation.solved = mean(interpolation.index.solved[solved]),
    extrapolation.solved = mean(extrapolation.index.solved[solved]),
    mean.sparse.error.max = mean(sparse.error.max[solved]),
    mean.sparse.error.mean = mean(sparse.error.mean[solved])
  )

print(dat.last.rate)

result = dat.last.rate %>%
  mutate(
    success.rate = rate.extrapolation,
    converged.at = extrapolation.solved,
    sparse.error = mean.sparse.error.mean
  ) %>%
  select(model, operation, success.rate, converged.at, sparse.error) %>%
  gather('key', 'value', success.rate, converged.at, sparse.error) %>%
  xtabs.data.first(value ~ model + key, exclude = NULL, na.action = na.pass)

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

result[,'success.rate'] = latex.rate(result[,'success.rate'])
result[,'converged.at'] = latex.digit(result[,'converged.at'])
result[,'sparse.error'] = latex.scientific(result[,'sparse.error'])

print(xtableFtable(ftable(result),
                   caption="Shows the sucess-rate for extrapolation < \\epsilon, at what global step the model converged at, and the sparse error for all weight matrices."),
      type='latex', NA.string='---',
      table.placement='H', caption.placement='top', math.style.exponents=T,
      file="../paper/results/simple_mul.tex")
