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
plot.x.breaks = c(1,10,200,400,600,800,1000)

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

eps = expand.name(read_csv('../results/sequential_mnist_sum_reference.csv')) %>%
  gather(
    key="parameter", value="test.extrapolation.mse",
    metric.test.extrapolation.1.mse, metric.test.extrapolation.10.mse,
    metric.test.extrapolation.100.mse, metric.test.extrapolation.200.mse,
    metric.test.extrapolation.300.mse, metric.test.extrapolation.400.mse,
    metric.test.extrapolation.500.mse, metric.test.extrapolation.600.mse,
    metric.test.extrapolation.700.mse, metric.test.extrapolation.800.mse,
    metric.test.extrapolation.900.mse, metric.test.extrapolation.1000.mse
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
  filter(seed %in% c(0,1,2,3,4,5,6,7,8,9)) %>%
  group_by(parameter) %>%
  summarise(
    threshold = mean(threshold) + qt(1 - alpha, 8) * (sd(threshold) / sqrt(n()))
  )

dat = expand.name(read_csv('../results/sequential_mnist_sum_long.csv')) %>%
  gather(
    key="parameter", value="test.extrapolation.mse",
    metric.test.extrapolation.1.mse, metric.test.extrapolation.10.mse,
    metric.test.extrapolation.100.mse, metric.test.extrapolation.200.mse,
    metric.test.extrapolation.300.mse, metric.test.extrapolation.400.mse,
    metric.test.extrapolation.500.mse, metric.test.extrapolation.600.mse,
    metric.test.extrapolation.700.mse, metric.test.extrapolation.800.mse,
    metric.test.extrapolation.900.mse, metric.test.extrapolation.1000.mse
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

dat$model = with(dat, as.factor(ifelse(
  as.character(model) == '$\\mathrm{NAC}_{+}$' & regualizer.z > 0,
  '$\\mathrm{NAC}_{+,R_z}$',
  as.character(model)
)))

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
  dat.gather = dat.last.rate %>% filter(
    (model == 'NAU' & regualizer.z == regualizer.z.show) | model != 'NAU'
  ) %>% mutate(
    parameter = as.factor(parameter)
  ) %>% plot.parameter.make.data()
  
  p = ggplot(dat.gather, aes(x = parameter, colour=model, group=model)) +
    geom_errorbar(aes(ymin = lower.value, ymax = upper.value), alpha=0.4) +
    geom_point(aes(y = mean.value, shape=our.model)) +
    geom_line(aes(y = mean.value)) +
    geom_blank(data = data.frame(
      key = c("success.rate"),
      model = NA,
      y.limit.max = c(1),
      parameter = mean(dat.gather$parameter)
    ), aes(x = parameter, y = y.limit.max)) +
    scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
    scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
    scale_x_discrete(name = plot.label, breaks=plot.x.breaks) +
    scale_shape(guide = FALSE) +
    facet_wrap(~ key, scales='free_y', labeller = labeller(
      key = c(
        success.rate = "Success rate",
        converged.at = "Solved at iteration step",
        sparse.error = "Sparsity error"
      )
    )) +
    guides(colour = guide_legend(nrow = 1)) +
    theme(legend.position="bottom") +
    theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
  
  return(p)
}

p.with.R.z = plot.by.regualizer.z(1)
print(p.with.R.z)
ggsave('../paper/results/sequential_mnist_sum_long.pdf', p.with.R.z, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")

p.without.R.z = plot.by.regualizer.z(0)
print(p.without.R.z)
ggsave('../paper/results/sequential_mnist_sum_long_ablation.pdf', p.without.R.z, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")
