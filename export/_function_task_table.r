
library(kableExtra)

save.table = function(dat.last.rate, label, caption, file.out, show.operation=TRUE, highlight.best=TRUE, longtable=FALSE) {
  
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
      return(paste0('\\mathbf{', content, '}'))
    } else {
      return(content)
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
                           paste0('$', latex.math.highlighter(formater(num), is.best), '$'),
                           '---')
      )
    return(df.format$formatted)
  }
  
  latex.ci.formater = function (vec.mean, vec.lower, vec.upper, formater, min.best=T) {
    df.format = data.frame(
      is.best = vec.mean == ifelse(min.best, min(vec.mean, na.rm=T), max(vec.mean, na.rm=T)),
      mean = vec.mean,
      ci.defined = !is.na(vec.upper + vec.lower),
      ci.upper = vec.upper - vec.mean,
      ci.lower = vec.mean - vec.lower
    ) %>%
      rowwise() %>%
      mutate(
        formatted = ifelse(is.finite(mean), 
                      ifelse(ci.defined,
                             paste0('$', latex.math.highlighter(formater(mean), is.best),' {~}^{+', formater(ci.upper), '}_{-', formater(ci.lower), '}$'),
                             paste0('$', latex.math.highlighter(formater(mean), is.best), '$')
                      ),
                      '---')
      )
    return(df.format$formatted)
  }

  align = c('c', 'r', 'l', 'l', 'l', 'l')
  header.1 = c("Op", "Model", "Success", "Solved at"=2, "Sparsity error")
  header.2 = c("", "", "Rate", "Median", "Mean", "Mean")

  if (!show.operation) {
    align = align[2:length(align)]
    header.1 = header.1[2:length(header.1)]
    header.2 = header.2[2:length(header.2)]
  }
  
  dat.table = dat.last.rate %>%
    group_by(operation) %>%
    mutate(
      success.rate = latex.ci.formater(success.rate.mean, success.rate.lower, success.rate.upper, latex.rate, min.best=F),
      converged.at.median = latex.formater(converged.at.median, latex.scientific),
      converged.at.mean = latex.ci.formater(converged.at.mean, converged.at.lower, converged.at.upper, latex.scientific),
      sparse.error = latex.ci.formater(sparse.error.mean, sparse.error.lower, sparse.error.upper, latex.scientific)
    ) %>%
    select(operation, model, success.rate, converged.at.median, converged.at.mean, sparse.error) %>%
    arrange(operation, model)
  
  no.pagebreak.rows =  which(head(dat.table$operation, -1) == tail(dat.table$operation, -1))
  
  if (!show.operation) {
    dat.table$operation = NULL
  }

  dat.table %>%
    kable(
      "latex", booktabs=T, align = align, escape=F, label=label,
      caption=caption,
      col.names = header.2,
      longtable=longtable
    ) %>%
    add_header_above(header.1) %>%
    kable_styling(latex_options=c('hold_position', 'repeat_header')) %>%
    row_spec(no.pagebreak.rows, extra_latex_after='\\nopagebreak') %>%
    collapse_rows(columns = ifelse(show.operation, c(1), c(1,2)), latex_hline = ifelse(show.operation, "major", "none")) %>%
    write(file.out)
}
