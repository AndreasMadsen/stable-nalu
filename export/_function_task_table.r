
library(kableExtra)

save.table = function(dat.last.rate, label, caption, file.out, show.operation=TRUE, highlight.best=TRUE) {
  
  latex.scientific = function (num) {
    exp = floor(log10(abs(num)))
    mech = sprintf("%.1f", num / (10^exp))
    return(paste0(mech, ' \\cdot 10^{', exp, '}'))
  }
  
  latex.digit = function (d) {
    return(sprintf("%.0f", as.numeric(d)))
  }
  
  latex.rate = function (d) {
    return(sprintf("%.0f\\%%", as.numeric(d) * 100))
  }
  
  latex.math.highlighter = function (content, highlight) {
    if (replace_na(highlight, FALSE) && highlight.best) {
      return(paste0('$\\mathbf{', content, '}$'))
    } else {
      return(paste0('$', content, '$'))
    }
  }
  
  
  latex.formater = function (vec, formater, min.best=T) {
    df.format = data.frame(
      is.best = vec == ifelse(min.best, min(vec, na.rm=T), max(vec, na.rm=T)),
      num = vec
    ) %>%
      rowwise() %>%
      mutate(
        formatted = ifelse(is.finite(num),
                           latex.math.highlighter(
                             formater(num),
                             is.best
                           ),
                           '---')
      )
    return(df.format$formatted)
  }
  
  latex.ci.formater = function (vec.mean, vec.ci, formater, min.best=T) {
    df.format = data.frame(
      is.best = vec.mean == ifelse(min.best, min(vec.mean, na.rm=T), max(vec.mean, na.rm=T)),
      mean = vec.mean,
      ci = vec.ci
    ) %>%
      rowwise() %>%
      mutate(
        formatted = ifelse(is.finite(mean), 
                           latex.math.highlighter(
                            ifelse(is.finite(ci), paste0(formater(mean), ' \\pm ', formater(ci)), formater(mean)),
                            is.best
                           ),
                           '---')
      )
    return(df.format$formatted)
  }

  align = c('c', 'r', 'l', 'l', 'l', 'l')
  header.1 = c("Operation", "Model", "Success", "Solved at"=2, "Sparsity error")
  header.2 = c("", "", "Rate", "Median", "Mean", "Mean")

  if (!show.operation) {
    align = align[2:length(align)]
    header.1 = header.1[2:length(header.1)]
    header.2 = header.2[2:length(header.2)]
  }
  
  dat.table = dat.last.rate %>%
    group_by(operation) %>%
    mutate(
      success.rate = latex.formater(rate.extrapolation, latex.rate, min.best=F),
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
