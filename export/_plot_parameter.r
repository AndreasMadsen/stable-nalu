
plot.parameter.make.data = function (dat.last.rate, ...) {
  dat.gather.mean = dat.last.rate %>%
    mutate(
      success.rate = success.rate.mean,
      converged.at = converged.at.mean,
      sparse.error = sparse.error.mean
    ) %>%
    select(model, operation, parameter, success.rate, converged.at, sparse.error, ...) %>%
    gather('key', 'mean.value', success.rate, converged.at, sparse.error)
  
  dat.gather.upper = dat.last.rate %>%
    mutate(
      success.rate = success.rate.upper,
      converged.at = converged.at.upper,
      sparse.error = sparse.error.upper
    ) %>%
    select(model, operation, parameter, success.rate, converged.at, sparse.error, ...) %>%
    gather('key', 'upper.value', success.rate, converged.at, sparse.error)
  
  dat.gather.lower = dat.last.rate %>%
    mutate(
      success.rate = success.rate.lower,
      converged.at = converged.at.lower,
      sparse.error = sparse.error.lower
    ) %>%
    select(model, operation, parameter, success.rate, converged.at, sparse.error, ...) %>%
    gather('key', 'lower.value', success.rate, converged.at, sparse.error)
  
  dat.gather = merge(merge(dat.gather.mean, dat.gather.upper), dat.gather.lower) %>%
    mutate(
      model=droplevels(model),
      key = factor(key, levels = c("success.rate", "converged.at", "sparse.error"))
    )
  
  return(dat.gather)
}

plot.parameter = function (dat.last.rate, plot.label, plot.x.breaks, ...) {
  dat.gather = plot.parameter.make.data(dat.last.rate, ...);
  
  p = ggplot(dat.gather, aes(x = parameter, colour=model)) +
    geom_errorbar(aes(ymin = lower.value, ymax = upper.value), alpha=0.5) +
    geom_point(aes(y = mean.value)) +
    geom_line(aes(y = mean.value)) +
    geom_blank(data = data.frame(
      key = c("success.rate"),
      model = NA,
      y.limit.max = c(1),
      parameter = mean(dat.gather$parameter)
    ), aes(x = parameter, y = y.limit.max)) +
    scale_color_discrete(labels = model.to.exp(levels(dat.gather$model))) +
    scale_y_continuous(name = element_blank(), limits=c(0,NA)) +
    scale_x_continuous(name = plot.label, breaks=plot.x.breaks) +
    facet_wrap(~ key, scales='free_y', labeller = labeller(
      key = c(
        success.rate = "Success rate",
        converged.at = "Solved at iteration step",
        sparse.error = "Sparsity error"
      )
    )) +
    theme(legend.position="bottom") +
    theme(plot.margin=unit(c(5.5, 10.5, 5.5, 5.5), "points"))
  
  return(p);
}
