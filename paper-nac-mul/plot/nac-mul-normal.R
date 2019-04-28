
library(ggplot2)

dat.train = read.csv("../data/nac-mul-normal_mul_0-tag-loss_train_critation.csv")
dat.train$loss = 'train'

dat.extra = read.csv("../data/nac-mul-normal_mul_0-tag-loss_valid_extrapolation.csv")
dat.extra$loss = 'extrapolation'

dat.inter = read.csv("../data/nac-mul-normal_mul_0-tag-loss_valid_interpolation.csv")
dat.inter$loss = 'interpolation'

dat = rbind(dat.train, dat.extra)

p = ggplot(dat, aes(x=Step, y=Value, colour=loss)) +
  geom_line() +
  scale_y_continuous(trans='log10', limits=c(1, NA)) + 
  theme(text = element_text(size=10)) + 
  ylab('MSE')
ggsave('../graphics/nac-mul-normal.pdf', p, width=12, height=4.5, units='cm', device='pdf')
print(p)
