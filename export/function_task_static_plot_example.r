#rm(list = ls())
setwd(dirname(parent.frame(2)$ofile))

library(ggplot2)
library(xtable)
library(plyr)
library(dplyr)
library(tidyr)
library(readr)
library(kableExtra)
source('./_function_task_expand_name.r')

eps = 0.2
median_range = 100

xtabs.data.first = function (data, formular, ...) {
  return(xtabs(formular, data, ...))
}

dat = expand.name(read_csv('../results/function_task_static.csv'))
dat.ggplot = subset(dat, dat$model %in% c('NMU', 'NALU', '$\\mathrm{NAC}_{\\bullet}$') & dat$operation == '${a \\cdot b}$' & dat$seed == 1) %>%
  mutate(
    model=droplevels(model),
    iteration=step,
    sparse.error = sparse.error.max,
    interpolation.error = interpolation,
    extrapolation.error = extrapolation
  ) %>%
  select(model, operation, iteration, interpolation.error, extrapolation.error, sparse.error) %>%
  gather('measurement', 'error', interpolation.error, extrapolation.error, sparse.error)

dat.eps = data.frame(
  measurement=c('interpolation.error', 'extrapolation.error', 'sparse.error'),
  epsilon=c(NA, 0.2, NA)
)

p = ggplot(dat.ggplot, aes(x=iteration, y=error, colour=model)) +
  geom_line(alpha=0.7) +
  geom_hline(aes(yintercept = epsilon), dat.eps, colour='black') +
  scale_y_continuous(trans="log10", name = element_blank()) +
  scale_color_discrete(labels = model.to.exp(levels(dat.ggplot$model))) +
  xlab('Iteration') +
  facet_wrap(~ measurement, scale='free_y', labeller = labeller(
    measurement = c(
      extrapolation.error = "Extrapolation error [MSE]",
      interpolation.error = "Interpolation error [MSE]",
      sparse.error = "Sparsity error"
    )
  )) +
  theme(legend.position="bottom") +
  theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
print(p)
ggsave('../paper/results/function-task-static-example.pdf', p, device="pdf", width = 13.968, height = 5, scale=1.4, units = "cm")
