rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(xtable)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(kableExtra)
source('./_sequential_mnist_expand_name.r')
source('./_compute_summary.r')
source('./_plot_parameter.r')

best.range = 1000
alpha = 0.01

plot.label = paste0("Extrapolation length")
plot.x.breaks = c(1,seq(2,20,2))

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

safe.interval = function (alpha, vec) {
  if (length(vec) <= 1) {
    return(NA)
  }

  return(abs(qt((1 - alpha) / 2, length(vec) - 1)) * (sd(vec) / sqrt(length(vec))))
}

eps = expand.name(read_csv('../results/sequential_mnist_prod_reference.csv')) %>%
  gather(
    key="parameter", value="test.extrapolation.mse",
    metric.test.extrapolation.1.mse, metric.test.extrapolation.2.mse,
    metric.test.extrapolation.3.mse, metric.test.extrapolation.4.mse,
    metric.test.extrapolation.5.mse, metric.test.extrapolation.6.mse,
    metric.test.extrapolation.7.mse, metric.test.extrapolation.8.mse,
    metric.test.extrapolation.9.mse, metric.test.extrapolation.10.mse,
    metric.test.extrapolation.11.mse, metric.test.extrapolation.12.mse,
    metric.test.extrapolation.13.mse, metric.test.extrapolation.14.mse,
    metric.test.extrapolation.15.mse, metric.test.extrapolation.16.mse,
    metric.test.extrapolation.17.mse, metric.test.extrapolation.18.mse,
    metric.test.extrapolation.19.mse, metric.test.extrapolation.20.mse
  ) %>%
  rowwise() %>%
  mutate(
    parameter = extrapolation.loss.name.to.integer(parameter)
  ) %>%
  group_by(seed, parameter) %>%
  summarise(
    best.model.step = best.model.step.fn(metric.valid.mse),
    threshold = test.extrapolation.mse[best.model.step],
  ) %>%
  filter(seed %in% c(0,2,4,5,6,7,9)) %>% # seed 1, 3, and 8 did not solve it
  group_by(parameter) %>%
  summarise(
    threshold = mean(threshold) + qt(1 - alpha, 8) * (sd(threshold) / sqrt(n()))
  )

dat = expand.name(read_csv('../results/sequential_mnist_prod_long.csv')) %>%
  gather(
    key="parameter", value="test.extrapolation.mse",
    metric.test.extrapolation.1.mse, metric.test.extrapolation.2.mse,
    metric.test.extrapolation.3.mse, metric.test.extrapolation.4.mse,
    metric.test.extrapolation.5.mse, metric.test.extrapolation.6.mse,
    metric.test.extrapolation.7.mse, metric.test.extrapolation.8.mse,
    metric.test.extrapolation.9.mse, metric.test.extrapolation.10.mse,
    metric.test.extrapolation.11.mse, metric.test.extrapolation.12.mse,
    metric.test.extrapolation.13.mse, metric.test.extrapolation.14.mse,
    metric.test.extrapolation.15.mse, metric.test.extrapolation.16.mse,
    metric.test.extrapolation.17.mse, metric.test.extrapolation.18.mse,
    metric.test.extrapolation.19.mse, metric.test.extrapolation.20.mse
  ) %>%
  mutate(
    valid.interpolation.mse=metric.valid.mse,
    train.interpolation.mse=metric.train.mse
  ) %>%
  select(-metric.valid.mse, -metric.train.mse) %>%
  rowwise() %>%
  mutate(
    parameter = extrapolation.loss.name.to.integer(parameter)
  ) %>%
  merge(eps)

dat.last = dat %>%
  group_by(name, parameter) %>%
  summarise(
    threshold = last(threshold),
    best.model.step = best.model.step.fn(valid.interpolation.mse),
    interpolation.last = valid.interpolation.mse[best.model.step],
    extrapolation.last = test.extrapolation.mse[best.model.step],
    interpolation.step.solved = first.solved.step(step, valid.interpolation.mse, threshold),
    extrapolation.step.solved = first.solved.step(step, test.extrapolation.mse, threshold),
    sparse.error.max = sparse.error.max[best.model.step],
    solved = replace_na(test.extrapolation.mse[best.model.step] < threshold, FALSE),

    model = last(model),
    operation = last(operation),
    regualizer.z = last(regualizer.z),
    seed = last(seed),
    size = n()
  )

dat.last.rate = dat.last %>%
  group_by(model, operation, parameter, regualizer.z) %>%
  group_modify(compute.summary) %>%
  ungroup()

plot.by.regualizer.z = function (regualizer.z.show) {
  dat.plot = dat.last.rate %>%
    filter(
      (regualizer.z == regualizer.z.show & model %in% c('$\\mathrm{NAC}_{\\bullet,\\mathrm{NMU}}$', 'NMU')) |
      model %in% c('$\\mathrm{NAC}_{\\bullet}$', '$\\mathrm{NAC}_{\\bullet,\\sigma}$', 'LSTM', 'NALU')
    )

  p = plot.parameter(dat.plot, plot.label, plot.x.breaks)
  return(p)
}

p.with.R.z = plot.by.regualizer.z(1)
print(p.with.R.z)
ggsave('../paper/results/sequential_mnist_prod_long.pdf', p.with.R.z, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")
ggsave('../paper/results/sequential_mnist_prod_long_short.pdf', p.with.R.z, device="pdf", width = 13.968, height = 5.1, scale=1.4, units = "cm")

p.without.R.z = plot.by.regualizer.z(0)
print(p.without.R.z)
ggsave('../paper/results/sequential_mnist_prod_long_ablation.pdf', p.without.R.z, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")
