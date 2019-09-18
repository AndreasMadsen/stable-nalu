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

eps = read_csv('../results/function_task_static_mse_expectation.csv') %>%
  filter(simple == FALSE & parameter == 'default') %>%
  mutate(
    operation = revalue(operation, operation.full.to.short)
  ) %>%
  select(operation, input.size, overlap.ratio, subset.ratio, extrapolation.range, threshold)

name.parameter = 'learning.rate'
name.label = 'Learning rate'
name.file = '../results/function_task_static_learning.csv'

dat = expand.name(read_csv(name.file)) %>%
  merge(eps) %>%
  mutate(
    parameter = !!as.name(name.parameter)
  )

dat.last = dat %>%
  group_by(name, parameter, learning.optimizer, learning.momentum) %>%
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
  group_by(model, operation, parameter, learning.optimizer, learning.momentum) %>%
  group_modify(compute.summary) %>%
  ungroup() %>%
  mutate(
    learning.optimizer = as.factor(paste0(as.character(learning.optimizer), ifelse(learning.momentum == 0, '', '-Nesterov')))
  )
    
make.plot = function (operation.latex, model.latex, filename) {
  dat.plot = dat.last.rate %>%
    filter(operation == operation.latex & model %in% model.latex) %>%
    mutate(model=droplevels(model)) %>%
    plot.parameter.make.data(learning.optimizer);
  
  p = ggplot(dat.plot, aes(x = parameter, colour=model)) +
    geom_point(aes(y = mean.value)) +
    geom_line(aes(y = mean.value, linetype=learning.optimizer)) +
    geom_errorbar(aes(ymin = lower.value, ymax = upper.value), alpha=0.5, width=0.2) +
    geom_blank(data = data.frame(
      key = c("success.rate", "success.rate", "success.rate"),
      model = NA,
      y.limit.max = c(1,1,1),
      parameter = c(0.01,0.01,0.01),
      learning.optimizer = c('Adam', 'SGD', 'SGD-Nesterov')
    ), aes(x = parameter, y = y.limit.max)) +
    scale_color_discrete(labels = model.to.exp(levels(dat.plot$model))) +
    scale_x_continuous(name = name.label, trans='log10') +
    scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
    facet_wrap(~ key, scales='free_y', labeller = labeller(
      key = c(
        success.rate = "Success rate in %",
        converged.at = "Solved at iteration step",
        sparse.error = "Sparsity error"
      )
    )) +
    labs(linetype = 'Optimizer') +
    theme(legend.position="bottom") +
    theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
  
  print(p)
  ggsave(filename, p, device="pdf", width = 13.968, height = 5, scale=1.4, units = "cm")
}

make.plot('$\\bm{+}$', c('$\\mathrm{NAC}_{+}$', 'NALU'), '../paper/results/simple_function_static_learning_add.pdf')
make.plot('$\\bm{-}$', c('$\\mathrm{NAC}_{+}$', 'NALU'), '../paper/results/simple_function_static_learning_sub.pdf')
make.plot('$\\bm{\\times}$', c('$\\mathrm{NAC}_{\\bullet}$', 'NALU'), '../paper/results/simple_function_static_learning_mul.pdf')
make.plot('$\\bm{\\mathbin{/}}$', c('$\\mathrm{NAC}_{\\bullet}$', 'NALU'), '../paper/results/simple_function_static_learning_div.pdf')
