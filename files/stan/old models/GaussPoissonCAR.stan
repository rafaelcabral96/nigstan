data{
  int<lower=1> N;                      // Number of observations
  int<lower=1> N_edges;                // Number of edges
  int<lower=0> N_covariates;           // Number of covariates
  int<lower=0> O[N];                   // Observations
  vector[N] logE;                      // Log of exposure

  matrix[N_edges,N] D;                 // Matrix containing the neightborhood structure
  matrix[N,N_covariates] B;            // Design matrix
}


parameters{
  real<lower=0> sigma;           // Standard deviation of spatial effects
  vector[N_covariates] beta;     // Regression coefficients

  vector[N] X;                   // Spatial effects
}

model{

  //observation layer---------------------------
  O ~ poisson_log(logE + B*beta + sigma * X);

  //latent field layer--------------------------
  target += -0.5*dot_self(D*X);   
  sum(X) ~ normal(0, 0.001 * N);  //soft sum to zero constraint

  //prior layer---------------------------------
  //prior for intercept
  beta[1]     ~ normal(0, 1000);
  //prior for regression coefficients
  beta[2]     ~ normal(0, 10);
  //prior for sigma
  sigma    ~ normal(0,5);

}

generated quantities {
  vector[N] RR = exp(B*beta + sigma * X);                           // Relative Risk
  vector[N] mu = logE + log(RR);                                    // Mean of y
  vector[N] log_lik;                                                // Log likelihood at each location

  for (n in 1:N){
    log_lik[n] = poisson_log_lpmf(O[n] | mu[n]);                    
  }
}

