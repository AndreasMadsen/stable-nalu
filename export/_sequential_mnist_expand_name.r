
model.full.to.short = c(
  'lstm'='LSTM',
  'nac'='$\\mathrm{NAC}_{+}$',
  'nac-nac-n'='$\\mathrm{NAC}_{\\bullet}$',
  'posnac-nac-n'='$\\mathrm{NAC}_{\\bullet, \\sigma}$',
  'nalu'='NALU',
  'reregualizedlinearnac'='NAU',
  'reregualizedlinearnac-nac-m'='NMU'
)

model.latex.to.exp = c(
  '$\\mathrm{NAC}_{+}$'=expression(paste("", "", plain(paste("NAC")), 
                                         phantom()[{
                                           paste("", "+")
                                         }], "")),
  '$\\mathrm{NAC}_{\\bullet}$'=expression(paste("", "", plain(paste("NAC")), 
                                           phantom()[{
                                             paste("", symbol("\xb7"))
                                           }], "")),
  'LSTM'='LSTM',
  'NALU'='NALU',
  'NAU'='NAU',
  'NMU'='NMU'
)

model.to.exp = function(v) {
  return(unname(revalue(v, model.latex.to.exp)))
}

operation.full.to.short = c(
  'op-cumsum'='cumsum',
  'op-cumprod'='cumprod'
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

regualizer.get.part = function (regualizer, index) {
  split = strsplit(regualizer, '-')[[1]]
  return(as.double(split[index + 1]))
}

dataset.get.part = function (dataset, index, simple.value) {
  split = strsplit(dataset, '-')[[1]]
  if (split[2] == 'simple') {
    return(simple.value)
  } else {
    return(as.numeric(split[index + 1]))
  }
}

model.get.mnist.setup = function (model) {
  split = strsplit(model, '-')[[1]]
  if (split[2] == 'l') {
    return('linear')
  } else if (split[2] == 's') {
    return('softmax')
  } else {
    return(NA)
  }
}

model.get.simplification.setup = function (model) {
  split = strsplit(model, '-')[[1]]
  if (split[3] == 'n') {
    return('none')
  } else if (split[3] == 's') {
    return('solved-accumulator')
  } else if (split[3] == 'p') {
    return('pass-though')
  } else {
    return(NA)
  }
}

expand.name = function (df) {
  names = data.frame(name=unique(df$name))
  
  df.expand.name = names %>%
    rowwise() %>%
    mutate(
      model=revalue(extract.by.split(name, 1), model.full.to.short, warn_missing=FALSE),
      digits=substring(extract.by.split(name, 2), 3),
      hidden.size=as.integer(substring(extract.by.split(name, 3), 3)),
      operation=revalue(extract.by.split(name, 4), operation.full.to.short, warn_missing=FALSE),
      
      oob.control = ifelse(substring(extract.by.split(name, 5), 5) == "r", "regualized", "clip"),
      regualizer.shape = substring(extract.by.split(name, 6), 4),
      epsilon.zero = as.numeric(substring(extract.by.split(name, 7), 5)),
      
      regualizer=regualizer.get.part(extract.by.split(name, 8), 1),
      regualizer.z=regualizer.get.part(extract.by.split(name, 8), 2),
      regualizer.oob=regualizer.get.part(extract.by.split(name, 8), 3),
      
      model.mnist = model.get.mnist.setup(extract.by.split(name, 9)),
      model.final = model.get.simplification.setup(extract.by.split(name, 9)),
      
      interpolation.range=as.integer(substring(extract.by.split(name, 10), 3)),
      extrapolation.range=substring(extract.by.split(name, 11), 3),
      
      batch.size=as.integer(substring(extract.by.split(name, 12), 2)),
      seed=as.integer(substring(extract.by.split(name, 13), 2))
    )
  
  df.expand.name$name = as.factor(df.expand.name$name)
  df.expand.name$operation = factor(df.expand.name$operation, c('cumsum', 'cumprod'))
  df.expand.name$model = as.factor(df.expand.name$model)
  df.expand.name$interpolation.range = as.factor(df.expand.name$interpolation.range)
  df.expand.name$extrapolation.range = as.factor(df.expand.name$extrapolation.range)

  return(merge(df, df.expand.name))
}
