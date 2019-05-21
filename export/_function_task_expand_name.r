
library(latex2exp)

model.full.to.short = c(
  'linear'='Linear',
  'relu'='ReLU',
  'relu6'='ReLU6',
  'nac'='$\\mathrm{NAC}_{+}$',
  'nac-nac-n'='$\\mathrm{NAC}_{\\bullet}$',
  'posnac-nac-n'='$\\mathrm{NAC}_{\\bullet}$, $\\mathbf{W} = \\sigma(\\mathbf{\\hat{W}})$',
  'nalu'='NALU',
  'reregualizedlinearnac'='NAU',
  'reregualizedlinearnac-nac-m'='NMU',
  'regualizedlinearnac-nac-m'='NMU, $\\mathbf{W} = \\mathbf{\\hat{W}}$',
  'sillyreregualizedlinearnac-nac-m'='NMU, $\\mathbf{z} = \\mathbf{W} \\odot \\mathbf{x}$'
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

extract.by.split = function (name, index, default=NA) {
  split = strsplit(as.character(name), '_')[[1]]
  if (length(split) >= index) {
    return(split[index])
  } else {
    return(default)
  }
}

range.full.to.short = function (range) {
  range = substring(range, 3)
  
  if (substring(range, 0, 1) == '[') {
    return(paste0('U', gsub('\\]-\\[', '] ∪ U[', gsub(' ', '', range))))
  } else {
    return(paste0('U[', gsub('^,', '-', gsub('-', ',', range)), ']'))
  }
}

regualizer.get.sparse = function (regualizer) {
  split = strsplit(regualizer, '-')[[1]]
  return(as.double(split[2]))
}

regualizer.get.oob = function (regualizer) {
  split = strsplit(regualizer, '-')[[1]]
  if (length(split) == 2) {
    return(as.double(1))
  } else {
    return(as.double(split[3]))
  }
}

dataset.get.part = function (dataset, index, simple.value) {
  split = strsplit(dataset, '-')[[1]]
  if (split[2] == 'simple') {
    return(simple.value)
  } else {
    return(as.numeric(split[index + 1]))
  }
}

expand.name = function (df) {
  names = data.frame(name=unique(df$name))
  
  df.expand.name = names %>%
    rowwise() %>%
    mutate(
      model=revalue(extract.by.split(name, 1), model.full.to.short, warn_missing=FALSE),
      operation=revalue(extract.by.split(name, 2), operation.full.to.short, warn_missing=FALSE),
      
      regualizer=regualizer.get.sparse(extract.by.split(name, 3)),
      regualizer.oob=regualizer.get.oob(extract.by.split(name, 3)),
      
      interpolation.range=range.full.to.short(extract.by.split(name, 4)),
      extrapolation.range=range.full.to.short(extract.by.split(name, 5)),
      
      input.size=dataset.get.part(extract.by.split(name, 6), 1, 4),
      subset.ratio=dataset.get.part(extract.by.split(name, 6), 2, NA),
      overlap.ratio=dataset.get.part(extract.by.split(name, 6), 3, NA),

      batch.size=as.integer(substring(extract.by.split(name, 7), 2)),
      seed=as.integer(substring(extract.by.split(name, 8), 2)),
      hidden.size=as.integer(substring(extract.by.split(name, 9, 'b2'), 2)),
    )
  
  df.expand.name$name = as.factor(df.expand.name$name)
  df.expand.name$operation = factor(df.expand.name$operation, c('$\\bm{\\times}$', '$\\bm{+}$', '$\\bm{-}$'))
  df.expand.name$model = as.factor(df.expand.name$model)
  df.expand.name$interpolation.range = as.factor(df.expand.name$interpolation.range)
  df.expand.name$extrapolation.range = as.factor(df.expand.name$extrapolation.range)
  
  return(merge(df, df.expand.name))
}
