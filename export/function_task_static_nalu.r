rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
source('./_function_task_expand_name.r')
source('./_function_task_table.r')
source('./_compute_summary.r')

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

name.input = '../results/function_task_static_nalu.csv'
name.output.pdf = '../paper/results/function_task_static_nalu.pdf'
name.output.tex = '../paper/results/function_task_static_nalu.tex'

dat = expand.name(read_csv(name.input)) %>%
  merge(eps) %>%
  rename(
    gate = network.layer_2.nalu.gate.mean
  ) %>%
  mutate(
    model = as.factor(ifelse(model == 'NALU', 'NALU (shared)', as.character(model)))
  )

dat.last = dat %>%
  group_by(name) %>%
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
    size = n(),

    gate.last = gate[best.model.step]
  )

dat.last.rate = dat.last %>%
  group_by(model, operation) %>%
  group_modify(compute.summary) %>%
  ungroup()

save.table(
  dat.last.rate,
  "simple-function-static-nalu-gate-table",
  "Shows the success-rate, when the model converged, and the sparsity error for all weight matrices, with 95\\% confidence interval. Each value is a summary of 100 different seeds.",
  name.output.tex
)

p = ggplot(dat.last, aes(x = gate.last, fill=solved)) +
  geom_histogram(position = "dodge", bins=25) +
  scale_x_continuous(name = 'Gate', labels = function (x.value) {
    return(ifelse(x.value == 1,
                  'add',
                  ifelse(x.value == 0,
                         'mul',
                         sprintf('%.2f', x.value))))
  }) +
  facet_grid(operation ~ model, labeller = labeller(
    operation = c(
      '$\\bm{\\times}$' = "Multiplication",
      '$\\bm{+}$' = "Addition"
    )
  )) +
  theme(legend.position="right") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
print(p)
ggsave(name.output.pdf, p, device="pdf", width = 13.968, height = 7, scale=1.4, units = "cm")

