data{
  int<lower=1> N;                      // Number of basis functions
  vector[N] x;                         // Observations
}


parameters{
  real<lower=0> sigma;         // Standard deviation of NIG noise
  real<lower=-1,upper=1> rho;  // Spatial range parameter
  real c;                      // Drift parameter
}

model{
  real theta_s;

  for (n in 2:N)
    x[n] ~ normal(c + rho*x[n-1], sigma);

  //prior layer---------------------------------
  //prior for mu
  c ~ normal(0,1);
  //prior for sigma
  theta_s = 0.02/sqrt(1-rho^2);
  target +=  log(theta_s) - theta_s*sigma;
  //prior for rho
  target += - sqrt(1 - rho)*1.15 - 0.5*log(1 - rho); 

}

generated quantities { 
  vector[N-1] log_lik;
  for(n in 1:N-1)
    log_lik[n] = normal_lpdf(x[n+1] | c + rho*x[n], sigma); 
}

