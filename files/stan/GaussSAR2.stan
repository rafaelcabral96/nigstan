data{
  int<lower=1> N;                      // Number of observations
  int<lower=0> N_covariates;           // Number of covariates
  vector[N] y;                         // Observations
  matrix[N,N] W;                       // Adjacency matrix
  matrix[N,N_covariates] B;            // Design matrix
  vector[N] v;                         // Eigenvalues of W
}

parameters{
  real<lower=0> sigma;           // Standard deviation of NIG noise
  real<lower=0, upper=1> rho;    // Spatial range parameter
  vector[N_covariates] beta;     // Regression coefficients  
}

transformed parameters{
  vector[N] X = (y - B*beta)/sigma;                   // Spatial effects
}

model{
  matrix[N,N] D = add_diag(-rho*W, 1);                // D = I - rho W;
  target += -0.5*dot_self(D*X);   
  target += sum(log(1-rho*v));                        // Log determinant of D

  //prior layer---------------------------------
  //prior for intercept
  beta[1]     ~ normal(0.0, 1000);
  //prior for regression coefficients
  beta[2]     ~ normal(0.0, 10);
  beta[3]     ~ normal(0.0, 10);
  //prior for kappa
  rho     ~ uniform(0,1);
  //prior for sigma
  sigma   ~ normal(0,5);

}

