
model.full.to.short = c(
  'lstm'='LSTM',
  'nac'='$\\mathrm{NAC}_{+}$',
  'nac-nac-n'='$\\mathrm{NAC}_{\\bullet}$',
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
  'o-sum'='sum',
  'o-prod'='product'
)

expand.name = function (df) {
  names = unique(df$name)
  names_split = unlist(strsplit(names, '_'))

  df.expand.name = data.frame(
    name=names,
    model=revalue(names_split[seq(1, length(names_split), 7)], model.full.to.short),
    operation=revalue(names_split[seq(2, length(names_split), 7)], operation.full.to.short),
    regualizer=as.double(substring(names_split[seq(3, length(names_split), 7)], 3)),

    interpolation.length=as.integer(substring(names_split[seq(4, length(names_split), 7)],3)),
    extrapolation.lengths=paste0('[', gsub('-', ',', substring(names_split[seq(5, length(names_split), 7)], 3)), ']'),

    batch.size=as.integer(substring(names_split[seq(6, length(names_split), 7)], 2)),
    seed=as.integer(substring(names_split[seq(7, length(names_split), 7)], 2))
  )

  return(merge(df, df.expand.name))
}
