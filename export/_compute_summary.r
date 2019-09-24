
library(binom)
library(bbmle)

compute.gamma.stat = function (x, p.lower, p.upper) {
  gamma.mean.variance.loglik = function (mu, sigma.2) {
    alpha = mu * mu / sigma.2;
    beta = mu / sigma.2;
    loglik = -sum(dgamma(x, alpha, beta, log=TRUE));
    return(loglik);
  }
  
  if (sum(!is.na(x)) > 1 && sd(x) > 1e-5) {
    stat = mle2(gamma.mean.variance.loglik,
                start = list(mu = mean(x), sigma.2 = var(x)),
                method = 'L-BFGS-B',
                lower = p.lower, upper = p.upper);
    
    ci = confint(stat, 'mu', quitely=T)
    while (is(ci, 'mle2')) {
      stat = mle2(gamma.mean.variance.loglik,
                  start = list(
                    mu = unname(coef(ci)['mu.mu']),
                    sigma.2 = unname(coef(ci)['sigma.2'])
                  ),
                  method = 'L-BFGS-B',
                  lower = p.lower, upper = p.upper);
      
      ci = confint(stat, 'mu', quitely=T)
    }

    return(list(
      mean = unname(coef(stat)['mu']),
      sigma.2 = unname(coef(stat)['sigma.2']),
      lower = unname(ci[1]),
      upper = unname(ci[2])
    ));
  } else {
    return(compute.normal.stat(x, p.lower, p.upper, sigma.2 = var(x)));
  }
}

compute.beta.stat = function (x, p.lower, p.upper) {
  x = pmin(0.5 - 1e-16, pmax(1e-16, x))

  beta.mean.v.loglik = function (mu, v) {
    mu.2x = mu * 2;
    alpha = mu.2x * v;
    beta = (1 - mu.2x) * v;
    loglik = -sum(pmin(1e6, pmax(-1e6, dbeta(x, alpha, beta, log=TRUE))));
    return(loglik);
  }
  
  v.init = (mean(x) * (0.5 - mean(x))) / var(x) - 1
  
  if (sum(!is.na(x)) > 1 && sd(x) > 1e-10) {
    stat = mle2(beta.mean.v.loglik,
                start = list(
                  mu = mean(x),
                  v = v.init
                ),
                method = 'L-BFGS-B',
                lower = c(mu=0, v=1e-5), upper = c(mu=0.5, v=Inf));
    
    if (any(is.na(unname(summary(stat)@coef[, "Std. Error"])) &
        is.na(sqrt(1/diag(stat@details$hessian))))) {
      return(compute.normal.stat(x, p.lower, p.upper, v = v.init));
    }
    ci = confint(stat, 'mu', quitely=T)
    
    while (is(ci, 'mle2')) {
      stat = mle2(beta.mean.v.loglik,
                  start = list(
                    mu = unname(coef(ci)['mu.mu']),
                    v = unname(coef(ci)['v'])
                  ),
                  method = 'L-BFGS-B',
                  lower = c(mu=0, v=1e-5), upper = c(mu=0.5, v=Inf));
      
      if (any(is.na(unname(summary(stat)@coef[, "Std. Error"])) &
          is.na(sqrt(1/diag(stat@details$hessian))))) {
        return(compute.normal.stat(x, p.lower, p.upper, v = v.init));
      }
      ci = confint(stat, 'mu', quitely=T)
    }
    
    return(list(
      mean = unname(coef(stat)['mu']),
      v = unname(coef(stat)['v']),
      lower = unname(ci[1]),
      upper = unname(ci[2])
    ));
  } else {
    return(compute.normal.stat(x, p.lower, p.upper, v = v.init));
  }
}

compute.normal.stat = function (x, p.lower, p.upper, ...) {
  mu = mean(x);
  alpha = 0.95;

  if (length(x) > 1) {
    ci = abs(qt((1 - alpha) / 2, length(x) - 1)) * (sd(x) / sqrt(length(x)));
    
    return(list(
      mean = mu,
      lower = min(p.upper['mu'], max(p.lower['mu'], mu - ci)),
      upper = min(p.upper['mu'], max(p.lower['mu'], mu + ci)),
      ...
    ));
  } else {
    return(list(mean = mu, lower = NA, upper = NA, ...));
  }
}

compute.summary = function (df.group, vars, assume.normal = FALSE) {
  solved = with(df.group, solved);
  converged.at = with(df.group, extrapolation.step.solved[solved]);
  sparse.error = with(df.group, sparse.error.max[solved]);
  
  success.rate.stat = binom.confint(
    sum(solved), length(solved),
    conf.level = 0.95, method = 'wilson'
  );
  
  if (assume.normal) {
    sparse.error.stat = compute.normal.stat(
      sparse.error,
      c(mu=0, sigma.2=1e-5),
      c(mu=0.5, sigma.2=Inf),
      v = (mean(x) * (0.5 - mean(x))) / var(x) - 1
    );
  } else {
    sparse.error.stat = compute.beta.stat(
      sparse.error,
      c(mu=0, sigma.2=1e-5),
      c(mu=0.5, sigma.2=Inf)
    );
  }
  
  if (assume.normal) {
    converged.at.stat = compute.normal.stat(
      converged.at,
      c(mu=0, sigma.2=1e-5),
      c(mu=Inf, sigma.2=Inf),
      sigma.2 = var(x)
    );
  } else {
    converged.at.stat = compute.gamma.stat(
      converged.at,
      c(mu=0, sigma.2=1e-5),
      c(mu=Inf, sigma.2=Inf)
    );
  }
  
  return(data.frame(
    success.rate.mean = success.rate.stat$mean,
    success.rate.lower = success.rate.stat$lower,
    success.rate.upper = success.rate.stat$upper,
    
    converged.at.median = ifelse(length(converged.at) > 0, median(converged.at), NA),
    converged.at.sigma.2 = converged.at.stat$sigma.2,
    converged.at.mean = converged.at.stat$mean,
    converged.at.lower = converged.at.stat$lower,
    converged.at.upper = converged.at.stat$upper,
    
    sparse.error.v = sparse.error.stat$v,
    sparse.error.mean = sparse.error.stat$mean,
    sparse.error.lower = sparse.error.stat$lower,
    sparse.error.upper = sparse.error.stat$upper
  ));
}
