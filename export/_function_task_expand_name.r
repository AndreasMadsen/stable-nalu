
library(latex2exp)

model.full.to.short = c(
  'linear'='Linear',
  'relu'='ReLU',
  'relu6'='ReLU6',
  'nac'='$\\mathrm{NAC}_{+}$',
  'nac-nac-n'='$\\mathrm{NAC}_{\\bullet}$',
  'nalu'='NALU',
  'reregualizedlinearnac'='NAU',
  'reregualizedlinearnac-nac-m'='NMU'
)

model.latex.to.exp = c(
  'Linear'='Linear',
  'ReLU'='ReLU',
  'ReLU6'='ReLU6',
  '$\\mathrm{NAC}_{+}$'=TeX('$\\mathrm{NAC}_{+}$'),
  '$\\mathrm{NAC}_{\\bullet}$'=expression(paste("", "", plain(paste("NAC")), 
                                           phantom()[{
                                             paste("", symbol("\xb7"))
                                           }], "")),
  'NALU'='NALU',
  'NAU'='NAU',
  'NMU'='NMU'
)

model.to.exp = function(v) {
  return(unname(revalue(v, model.latex.to.exp)))
}

operation.full.to.short = c(
  'o-add'='$a + b$',
  'o-sub'='$a - b$',
  'o-mul'='${a \\cdot b}$',
  'o-div'='${a \\mathbin{/} b}$',
  'o-squared'='${a^2}$',
  'o-root'='${\\sqrt{a}}$'
)

range.full.to.short = function (range) {
  content = substring(range, 3)
  return(paste0('U[', gsub('^,', '-', gsub('-', ',', content)), ']'))
}

expand.name = function (df) {
  names = unique(df$name)
  names_split = unlist(strsplit(names, '_'))

  subset_config = substring(names_split[seq(6, length(names_split), 8)], 3)
  subset_config_split = unlist(strsplit(subset_config, '-'))

  df.expand.name = data.frame(
    name=names,
    model=revalue(names_split[seq(1, length(names_split), 8)], model.full.to.short),
    operation=revalue(names_split[seq(2, length(names_split), 8)], operation.full.to.short),
    regualizer=as.double(substring(names_split[seq(3, length(names_split), 8)], 3)),

    interpolation.range=range.full.to.short(names_split[seq(4, length(names_split), 8)]),
    extrapolation.range=range.full.to.short(names_split[seq(5, length(names_split), 8)]),

    input.size=as.numeric(subset_config_split[seq(1, length(subset_config_split), 3)]),
    subset.ratio=as.double(subset_config_split[seq(2, length(subset_config_split), 3)]),
    overlap.ratio=as.double(subset_config_split[seq(3, length(subset_config_split), 3)]),

    batch.size=as.numeric(substring(names_split[seq(7, length(names_split), 8)], 2)),
    seed=as.numeric(substring(names_split[seq(8, length(names_split), 8)], 2))
  )

  return(merge(df, df.expand.name))
}
