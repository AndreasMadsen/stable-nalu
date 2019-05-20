
library(latex2exp)

model.full.to.short = c(
  'linear'='Linear',
  'relu'='ReLU',
  'relu6'='ReLU6',
  'nac'='$\\mathrm{NAC}_{+}$',
  'nac-nac-n'='$\\mathrm{NAC}_{\\bullet}$',
  'posnac-nac-n'='$\\mathrm{NAC}_{\\bullet}$ ($\\mathbf{W} = \\sigma(\\mathbf{\\hat{W}}$))',
  'nalu'='NALU',
  'reregualizedlinearnac'='NAU',
  'reregualizedlinearnac-nac-m'='NMU',
  'regualizedlinearnac-nac-m'='NMU ($\\mathbf{W} = \\mathbf{\\hat{W}}$)',
  'sillyreregualizedlinearnac-nac-m'='NMU ($\\mathbf{z} = \\mathbf{W} \\odot \\mathbf{x}$)'
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
  'o-add'='$\\bm{+}$',
  'o-sub'='$\\bm{-}$',
  'o-mul'='$\\bm{\\times}$',
  'o-div'='$\\bm{\\mathbin{/}}$',
  'o-squared'='$z^2$',
  'o-root'='$\\sqrt{z}$'
)

range.full.to.short.element = function (range) {
  if (substring(range, 0, 1) == '[') {
    return(paste0('U', gsub('\\]-\\[', '] ∪ U[', gsub(' ', '', range))))
  } else {
    return(paste0('U[', gsub('^,', '-', gsub('-', ',', range)), ']'))
  }
}

range.full.to.short = function (range) {
  content = substring(range, 3)
  return(sapply(content, range.full.to.short.element))
}

regualizer.get.sparse = function (regualizer) {
  return(sapply(regualizer, function (str) {
    split = unlist(strsplit(str, '-'))
    return(split[1])
  }))
}

regualizer.get.oob = function (regualizer) {
  return(sapply(regualizer, function (str) {
    split = unlist(strsplit(str, '-'))
    if (length(split) == 1) {
      return(1)
    } else {
      return(split[2])
    }
  }))
}

expand.name = function (df) {
  names = unique(df$name)
  names_split = unlist(strsplit(names, '_'))

  subset_config = substring(names_split[seq(6, length(names_split), 8)], 3)
  if (subset_config[1] == 'simple') {
    input.size = 4
    subset.ratio = NA
    overlap.ratio = NA
  } else {
    subset_config_split = unlist(strsplit(subset_config, '-'))
    input.size = as.numeric(subset_config_split[seq(1, length(subset_config_split), 3)])
    subset.ratio = as.double(subset_config_split[seq(2, length(subset_config_split), 3)])
    overlap.ratio = as.double(subset_config_split[seq(3, length(subset_config_split), 3)])
  }
  
  regualizer = substring(names_split[seq(3, length(names_split), 8)], 3)
  
  df.expand.name = data.frame(
    name=names,
    model=revalue(names_split[seq(1, length(names_split), 8)], model.full.to.short),
    operation=revalue(names_split[seq(2, length(names_split), 8)], operation.full.to.short),
    regualizer=as.double(regualizer.get.sparse(regualizer)),
    regualizer.oob=as.double(regualizer.get.oob(regualizer)),

    interpolation.range=range.full.to.short(names_split[seq(4, length(names_split), 8)]),
    extrapolation.range=range.full.to.short(names_split[seq(5, length(names_split), 8)]),

    input.size=input.size,
    subset.ratio=subset.ratio,
    overlap.ratio=overlap.ratio,

    batch.size=as.numeric(substring(names_split[seq(7, length(names_split), 8)], 2)),
    seed=as.numeric(substring(names_split[seq(8, length(names_split), 8)], 2))
  )

  return(merge(df, df.expand.name))
}
