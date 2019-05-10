rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(xtable)

eps = 0.2
median_range = 100

xtabs.data.first = function (data, formular, ...) {
  return(xtabs(formular, data, ...))
}

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
  'o-add'='$a + b$',
  'o-sub'='$a - b$',
  'o-mul'='${a \\cdot b}$',
  'o-div'='${a \\mathbin{/} b}$',
  'o-squared'='${a^2}$',
  'o-root'='${\\sqrt{a}}$'
)

range.full.to.short = function (range) {
  content = substring(range, 3)
  return(paste0('U[', gsub('-', ',', content), ']'))
}

expand.name = function (df) {
  names = unique(df$name)
  names_split = unlist(strsplit(names, '_'))
  
  subset_config = substring(names_split[seq(6, length(names_split), 8)], 3)
  subset_config_split = unlist(strsplit(subset_config, '-'))
  
  df.expand.name = data.frame(
    name=names,
    model=revalue(names_split[seq(1, length(names_split), 8)], model.full.to.short),
    operation=revalue(names_split[seq(2, length(names_split), 8)], operation.full.to.short),
    regualizer=as.double(substring(names_split[seq(3, length(names_split), 8)], 3)),
    
    interpolation.range=range.full.to.short(names_split[seq(4, length(names_split), 8)]),
    extrapolation.range=range.full.to.short(names_split[seq(5, length(names_split), 8)]),
    
    input.size=as.numeric(subset_config_split[seq(1, length(subset_config_split), 3)]),
    subset.ratio=as.double(subset_config_split[seq(2, length(subset_config_split), 3)]),
    overlap.ratio=as.double(subset_config_split[seq(3, length(subset_config_split), 3)]),
    
    batch.size=as.numeric(substring(names_split[seq(7, length(names_split), 8)], 2)),
    seed=as.numeric(substring(names_split[seq(8, length(names_split), 8)], 2))
  )
  
  return(merge(df, df.expand.name))
}

dat = expand.name(read_csv('../results/function_task_static.csv'))

dat.last = dat %>%
  group_by(name) %>%
  #filter(n() == 5001) %>%
  summarise(
    interpolation.last = median(tail(interpolation, median_range)),
    extrapolation.last = median(tail(extrapolation, median_range)),
    interpolation.step.solved = first(which(interpolation < eps)) * 1000,
    extrapolation.step.solved = first(which(extrapolation < eps)) * 1000,
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
    interpolation.solved = mean(interpolation.step.solved[solved]),
    extrapolation.solved = mean(extrapolation.step.solved[solved]),
    mean.sparse.error.max = mean(sparse.error.max[solved]),
    mean.sparse.error.mean = mean(sparse.error.mean[solved])
  )

result = dat.last.rate %>%
  mutate(
    success.rate = rate.extrapolation,
    converged.at = extrapolation.solved,
    sparse.error = mean.sparse.error.mean
  ) %>%
  select(model, operation, success.rate, converged.at, sparse.error) %>%
  gather('key', 'value', success.rate, converged.at, sparse.error) %>%
  xtabs.data.first(value ~ operation + model + key, exclude = NULL, na.action = na.pass)

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

result[,,'success.rate'] = latex.rate(result[,,'success.rate'])
result[,,'converged.at'] = latex.digit(result[,,'converged.at'])
result[,,'sparse.error'] = latex.scientific(result[,,'sparse.error'])

print(xtableFtable(ftable(result),
      caption="Shows the sucess-rate for extrapolation < \\epsilon, at what global step the model converged at, and the sparse error for all weight matrices."),
      type='latex', NA.string='---',
      table.placement='H', caption.placement='top', math.style.exponents=T,
      file="../paper/results/function_task_static.tex")
