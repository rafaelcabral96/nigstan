functions{
  /**
   * Returns the interpoled value y evaluated at x inside the range of x0, and 
   * returns the linear extrapolated value if x is outside the range of x0, for 
   * several weight functions
   * https://discourse.mc-stan.org/t/linear-interpolation-and-searchsorted-in-stan/13318/5
   *
   * @param x
   * @param x0 Precomputed x-values
   * @param y0 Precomputed y-values
   * @return Interpolated value at x
   */
  real interpolation(real x, vector x0, vector y0){
    int K = rows(x0);
    vector[K] deltas = x - x0;
    real ans;
    real t;
    real w;
    int i;
    if(x < x0[1]){
      real slope = (y0[2] - y0[1])/(x0[2] - x0[1]);
      ans = y0[1] + slope*(x - x0[1]);}
    else if(x > x0[K]){
      real slope = (y0[K] - y0[K-1])/(x0[K] - x0[K-1]);
      ans = y0[K] + slope*(x -x0[K]);}
    else{
      i = sort_indices_asc(fabs(deltas))[1];  //this is which.min()
      if(deltas[i]<=0) i -= 1;
      ans = y0[i];
      real x1 = x0[i];
      real x2 = x0[i + 1];
      real y1 = y0[i];
      real y2 = y0[i + 1];
      t = (x-x1)/(x2-x1);
      w = 1-t;
      //w = 1 - 3*pow(t,2) + 2*pow(t,3);
      //w = 1/(1 + exp(1/(1-t) - 1/t));
      ans = w*y1 + (1-w)*y2;}
    return ans;
   }
}

data{
  int<lower=1> N;                      // Number of observations
  int<lower=0> N_covariates;           // Number of covariates
  int<lower=0> O[N];                   // Observations
  vector[N] logE;                      // Log of exposure

  int inter_points;                    // Interpolation points for log determiant
  vector[inter_points] kappa_log;      // Log of precomputed kappas
  vector[inter_points] det_log;        // Log of precomputed det(kappa^2*C + G)

  matrix[N,N] A;                       // Laplacian matrix containing the neightborhood structure
  matrix[N,N_covariates] B;            // Design matrix
}

transformed data{
  int sizew = rows(csr_extract_w(A));     // Number of non-zero values of matrix D
  int Dv[size(csr_extract_u(A))];         // Collumn indeces (in compressed row storage format)
  int Du[size(csr_extract_u(A))];         // Row indeces (in compressed row storage format)
  int sizes[3];                           // Vector containing number of rows, collumns, and number of non-zero elements in D

  Dv = csr_extract_v(A);
  Du = csr_extract_u(A);
  sizes[1] = N;
  sizes[2] = N;
  sizes[3] = sizew;

  vector[N] h = rep_vector(1,N);          // In this problem h is a vector of ones
}

parameters{
  real<lower=0> sigma_theta;     // Standard deviation of heterogeneous effects
  real<lower=0> sigma_X;         // Standard deviation of spatial effects
  real<lower=0> kappa;           // Spatial range parameter
  vector[N_covariates] beta;     // Regression coefficients

  vector[N] X;                   // Spatial effects
  vector[N] theta;               // Heterogeneous effects
}

model{
  matrix[N,N] D;

  //observation layer---------------------------
  O ~ poisson_log(logE + B*beta + sigma_X * X + sigma_theta * theta);

  //latent field layer--------------------------
  D = add_diag(A, kappa^2);
  target += -0.5*dot_self(D*X);   
  target += interpolation(log(kappa), kappa_log, det_log);

  //Heterogeneous spatial effects
  theta ~ normal(0,1);

  //prior layer---------------------------------
  //prior for intercept
  beta[1]     ~ normal(0, 1000);
  //prior for regression coefficients
  beta[2]     ~ normal(0, 10);
  //prior for sigmae
  sigma_theta ~ inv_gamma(1,1);
  //prior for kappa
  sigma_X     ~ normal(0,1);
  //prior for sigma
  kappa       ~ normal(0,1);  

}

generated quantities {
  vector[N] mu = logE + B*beta + sigma_X * X + sigma_theta * theta; //
  vector[N] log_lik;                                                //

  for (n in 1:N){
    log_lik[n] = poisson_log_lpmf(O[n] | mu[n]);                    //Log likelihood at each location
  }
}

