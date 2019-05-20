
library(kableExtra)

save.table = function(dat.last.rate, label, caption, file.out, show.operation=TRUE) {
  
  latex.scientific = function (num) {
    exp = floor(log10(abs(num)))
    mech = sprintf("%.1f", num / (10^exp))
    return(paste0(mech, ' \\cdot 10^{', exp, '}'))
  }
  
  latex.formater = function (num, formater) {
    if (!is.finite(num)) {
      return('---')
    }
    
    return(paste0('$', formater(num), '$'))
  }
  
  latex.ci.formater = function (mean, ci, formater) {
    if (!is.finite(mean)) {
      return('---')
    }
    if (!is.finite(ci)) {
      return(paste0('$', formater(mean), '$'))
    }
    return(paste0('$', formater(mean), ' \\pm ', formater(ci), '$'))
  }
  
  latex.digit = function (d) {
    return(sprintf("%.0f", as.numeric(d)))
  }
  
  latex.rate = function (d) {
    return(sprintf("%.0f\\%%", as.numeric(d) * 100))
  }

  align = c('c', 'r', 'l', 'l', 'l', 'l')
  header.1 = c("Operation", "Model", "Success", "Converged at"=2, "Sparsity error")
  header.2 = c("", "", "Rate", "Median", "Mean", "Mean")

  if (!show.operation) {
    align = align[2:length(align)]
    header.1 = header.1[2:length(header.1)]
    header.2 = header.2[2:length(header.2)]
  }
  
  dat.table = dat.last.rate %>%
    group_by(model, operation) %>%
    mutate(
      success.rate = latex.formater(rate.extrapolation, latex.rate),
      converged.at.median = latex.formater(median.extrapolation.solved, latex.scientific),
      converged.at.mean = latex.ci.formater(mean.extrapolation.solved, ci.extrapolation.solved, latex.scientific),
      sparse.error = latex.ci.formater(mean.sparse.error.max, ci.sparse.error.max, latex.scientific)
    ) %>%
    select(operation, model, success.rate, converged.at.median, converged.at.mean, sparse.error) %>%
    arrange(operation, model)
  
  if (!show.operation) {
    dat.table$operation = NULL
  }

  dat.table %>%
    kable(
      "latex", booktabs=T, align = align, escape=F, label=label,
      caption=caption,
      col.names = header.2
    ) %>%
    add_header_above(header.1) %>%
    kable_styling(latex_options=c('HOLD_position')) %>%
    collapse_rows(columns = ifelse(show.operation, c(1), c(1,2)), latex_hline = ifelse(show.operation, "major", "none")) %>%
    write(file.out)
}
