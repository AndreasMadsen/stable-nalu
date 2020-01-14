
model.full.to.short = c(
  'lstm'='LSTM',
  'nac'='$\\mathrm{NAC}_{+}$',
  'nac-nac-n'='$\\mathrm{NAC}_{\\bullet}$',
  'posnac-nac-n'='$\\mathrm{NAC}_{\\bullet,\\sigma}$',
  'reregualizedlinearposnac-nac-n'='$\\mathrm{NAC}_{\\bullet,\\mathrm{NMU}}$',
  'nalu'='NALU',
  'reregualizedlinearnac'='NAU',
  'reregualizedlinearnac-nac-m'='NMU'
)

model.latex.to.exp = c(
  '$\\mathrm{NAC}_{+}$'=expression(paste("", "", plain(paste("NAC")), 
                                         phantom()[{
                                           paste("", "+")
                                         }], "")),
  '$\\mathrm{NAC}_{+,R_z}$'=expression(paste("", "", plain(paste("NAC")), 
                                         phantom()[{
                                           paste("", "+", ",", "R", phantom()[{
                                             paste("z")
                                           }])
                                         }], "")),
  '$\\mathrm{NAC}_{\\bullet}$'=expression(paste("", "", plain(paste("NAC")), 
                                           phantom()[{
                                             paste("", symbol("\xb7"))
                                           }], "")),
  '$\\mathrm{NAC}_{\\bullet,\\sigma}$'=expression(paste("", "", plain(paste("NAC")), 
                                                        phantom()[{
                                                          paste("", symbol("\xb7"), ",", sigma)
                                                        }], "")),
  '$\\mathrm{NAC}_{\\bullet,\\mathrm{NMU}}$'=expression(paste("", "", plain(paste("NAC")), 
                                                              phantom()[{
                                                                paste("", symbol("\xb7"), ",", plain(paste("NMU")))
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

regualizer.get.type = function (regualizer, index) {
  split = strsplit(regualizer, '-')[[1]]
  return(split[index + 1])
}

regualizer.scaling.get = function (regualizer, index) {
  split = strsplit(regualizer, '-')[[1]]
  return(as.numeric(split[index + 1]))
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

extrapolation.loss.name.to.integer = function (loss.name) {
  split = strsplit(loss.name, '\\.')[[1]]
  return(as.integer(split[4]))
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
      regualizer.scaling = regualizer.get.type(extract.by.split(name, 6), 1), # rs[1]
      regualizer.shape = regualizer.get.type(extract.by.split(name, 6), 2), # rs[2]
      epsilon.zero = as.numeric(substring(extract.by.split(name, 7), 5)),
      
      regualizer.scaling.start=regualizer.scaling.get(extract.by.split(name, 8), 1),
      regualizer.scaling.end=regualizer.scaling.get(extract.by.split(name, 8), 2),
      
      regualizer=regualizer.get.part(extract.by.split(name, 9), 1),
      regualizer.z=regualizer.get.part(extract.by.split(name, 9), 2),
      regualizer.oob=regualizer.get.part(extract.by.split(name, 9), 3),
      
      model.mnist = model.get.mnist.setup(extract.by.split(name, 10)),
      model.final = model.get.simplification.setup(extract.by.split(name, 10)),
      
      interpolation.length=as.integer(substring(extract.by.split(name, 11), 3)),
      extrapolation.length=substring(extract.by.split(name, 12), 3),
      
      batch.size=as.integer(substring(extract.by.split(name, 13), 2)),
      seed=as.integer(substring(extract.by.split(name, 14), 2))
    )
  
  df.expand.name$name = as.factor(df.expand.name$name)
  df.expand.name$operation = factor(df.expand.name$operation, c('cumsum', 'cumprod'))
  df.expand.name$model = as.factor(df.expand.name$model)

  return(merge(df, df.expand.name))
}
