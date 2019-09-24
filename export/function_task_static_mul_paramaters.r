rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
source('./_function_task_expand_name.r')
source('./_compute_summary.r')
source('./_plot_parameter.r')

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

name.output = '../paper/results/simple_function_static_mul_parameter.pdf'
name.output.one.row = '../paper/results/simple_function_static_mul_parameter_onerow.pdf'

data.extracts = rbind(
  c(name.parameter = 'input.size',
    name.reference = 'input.size',
    plot.label='Input Size',
    name.input='../results/function_task_static_mul_input_size.csv'),
  c(name.parameter = 'hidden.size',
    name.reference = 'default',
    plot.label='Hidden Size',
    name.input='../results/function_task_static_mul_hidden_size.csv'),
  c(name.parameter = 'subset.ratio',
    name.reference = 'subset.ratio',
    plot.label='Subset Ratio',
    name.input='../results/function_task_static_mul_subset.csv'),
  c(name.parameter = 'overlap.ratio',
    name.reference = 'overlap.ratio',
    plot.label='Overlap Ratio',
    name.input='../results/function_task_static_mul_overlap.csv')
)

dat.all.rate = NA

for (row in 1:nrow(data.extracts)) {
  eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
    filter(simple == FALSE & parameter == data.extracts[row, 'name.reference']) %>%
    mutate(
      operation = revalue(operation, operation.full.to.short)
    ) %>%
    select(operation, input.size, overlap.ratio, subset.ratio, extrapolation.range, threshold)
  
  dat = expand.name(read_csv(data.extracts[row, 'name.input'])) %>%
    filter(model %in% c('NALU', '$\\mathrm{NAC}_{\\bullet}$')) %>%
    mutate(
      model = droplevels(model)
    ) %>%
    merge(eps) %>%
    mutate(
      parameter = !!as.name(data.extracts[row, 'name.parameter'])
    )

  dat.last = dat %>%
    group_by(name, parameter) %>%
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

  dat.last.rate = dat.last %>%
    group_by(model, operation, parameter) %>%
    group_modify(compute.summary) %>%
    ungroup() %>%
    mutate(
      plot.label = data.extracts[row, 'plot.label'],
      width = max(parameter) / 30
    )

  if (is.na(dat.all.rate)) {
    dat.all.rate = dat.last.rate;
  } else {
    dat.all.rate = rbind(dat.all.rate, dat.last.rate);
  }
}

p = ggplot(dat.all.rate, aes(x = parameter, colour=model)) +
  geom_errorbar(aes(ymin = success.rate.lower, ymax = success.rate.upper, width=width), alpha=0.5) +
  geom_point(aes(y = success.rate.mean)) +
  geom_line(aes(y = success.rate.mean)) +
  scale_color_discrete(labels = model.to.exp(levels(dat.all.rate$model))) +
  scale_y_continuous(name = "Success Rate", limits=c(0,1)) +
  scale_x_continuous(name = element_blank(), limits=c(0,NA)) +
  facet_wrap(~ plot.label, scales='free_x') +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
print(p)
ggsave(name.output, p, device="pdf", width = 13.968, height = 9, scale=1.4, units = "cm")

p = ggplot(dat.all.rate, aes(x = parameter, colour=model)) +
  geom_errorbar(aes(ymin = success.rate.lower, ymax = success.rate.upper, width=width), alpha=0.5) +
  geom_point(aes(y = success.rate.mean)) +
  geom_line(aes(y = success.rate.mean)) +
  scale_color_discrete(labels = model.to.exp(levels(dat.all.rate$model))) +
  scale_y_continuous(name = "Success Rate", limits=c(0,1)) +
  scale_x_continuous(name = element_blank(), limits=c(0,NA)) +
  facet_wrap(~ plot.label, scales='free_x', nrow=1) +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
print(p)
ggsave(name.output.one.row, p, device="pdf", width = 13.968, height = 5.7, scale=1.4, units = "cm")
