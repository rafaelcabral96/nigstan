data{
  int<lower=1> N;                      // Number of observations
  int<lower=0> N_covariates;           // Number of covariates
  vector[N] y;                         // Observations
  matrix[N,N] A;                       // Laplacian matrix containing the neightborhood structure
  matrix[N,N_covariates] B;            // Design matrix
}

parameters{
  real<lower=0> sigmae;          // Standard deviation of observations y
  real<lower=0> sigma;           // Standard deviation of NIG noise
  real<lower=0> kappa;           // Spatial range parameter
  vector[N] X;                   // Spatial effects
  vector[N_covariates] beta;     // Regression coefficients
  
}

model{
  matrix[N,N] D;

  //observation layer---------------------------
  y ~ normal(B*beta + sigma*X, sigmae);

  //latent field layer--------------------------
  D = add_diag(A, kappa^2);
  target += log_determinant(D) -0.5*dot_self(D*X);   

  //prior layer---------------------------------
  //prior for intercept
  beta[1]     ~ normal(0.0, 1000);
  //prior for regression coefficients
  beta[2]     ~ normal(0.0, 10);
  beta[3]     ~ normal(0.0, 10);
  //prior for sigmae
  sigmae  ~ normal(0,1);
  //prior for kappa
  kappa   ~ normal(0,1);
  //prior for sigma
  sigma   ~ normal(0,5);

}

generated quantities {
  vector[N] my = B*beta + sigma*X;                                            //mean of y
  vector[N] log_lik = -0.5*log(2*pi())-log(sigmae)-(0.5/sigmae^2)*(y-my)^2;   //Log likelihood at each location
}

