functions {

  /**
   * Returns the log-likelihood of the NIG distribution with variance 
   * correction at x (Var[x] = sigma^2*h)
   *
   * @param x 
   * @param sigma Standard deviation 
   * @param eta   First flexibility parameter (controls heaviness of tails)
   * @param mu    Second flexibility parameter (controls asymmetry of tails)
   * @param h     Distance between locations or area of the basis function
   * @return      Log-likelihood of the NIG distribution with variance correction
   */
  real NIG_var_correction_lpdf(real x, real eta, real mu, real h){
    real sigmas     = 1/sqrt(1+mu^2*eta); //variance correction
    real hyp_alpha  = sqrt(1/eta+mu^2)/sigmas;
    real hyp_beta   = mu/sigmas;
    real hyp_delta  = sigmas*sqrt(1/eta)*h;
    real hyp_mu     = -sigmas*mu*h;
    return (sqrt(hyp_alpha^2 - hyp_beta^2)*hyp_delta + hyp_beta*(x - hyp_mu) + log(hyp_alpha) + 
            log(hyp_delta) - log(pi()) - 0.5*log(hyp_delta^2 + (x - hyp_mu)^2) + 
            log(modified_bessel_second_kind(1, hyp_alpha*sqrt(hyp_delta^2 + (x - hyp_mu)^2)))); 
  }

  /**
   * Returns the log-likelihood of the NIG distribution with standardized and orthogonal parameterization
   *
   * @param x 
   * @param m     Mean
   * @param sigma Standard deviation 
   * @param eta   First flexibility parameter (controls heaviness of tails)
   * @param mu    Second flexibility parameter (controls asymmetry of tails)
   * @param h     Distance between locations or area of the basis function
   * @return      Log-likelihood of the NIG distribution with standardized and orthogonal parameterization
   */
  real nig_lpdf(real x, real m, real sigma, real etas, real zetas, real h){

    //convert parameters (etas, zetas) to original parameterization (eta, mu) 
    real eta = etas*(1+zetas^2-fabs(zetas)*sqrt(1+zetas^2))^2;
    real mu  = zetas/sqrt(eta);

    return( NIG_var_correction_lpdf( (x-m)/sigma | eta, mu, h) );
  }

}

data {
  int<lower=0> N;
  int<lower=0> N_edges;
  int<lower=1, upper=N> node1[N_edges];  // node1[i] adjacent to node2[i]
  int<lower=1, upper=N> node2[N_edges];  // and node1[i] < node2[i]

  int<lower=0> y[N];              // count outcomes
  vector<lower=0>[N] E;           // exposure
  int<lower=1> K;                 // num covariates
  matrix[N, K] x;                 // design matrix

  real<lower=0> scaling_factor; // scales the variance of the spatial effects

  real<lower=0> theta_etas;       // Rate constant for PC prior of etas
  real<lower=0> theta_zetas;        // Rate constant for PC prior of zetas
}

transformed data {
  vector[N] log_E = log(E);
}

parameters {
  real beta0;            // intercept
  vector[K] betas;       // covariates

  real<lower=0> sigma;        // overall standard deviation
  real<lower=0, upper=1> rho; // proportion unstructured vs. spatially structured variance

  vector[N] theta;       // heterogeneous effects
  vector[N] phi;         // spatial effects

  real<lower=0> etas;            // First flexibility parameter (controls heaviness of tails)   
  real zetas;                      // Second flexibility parameter (controls asymmetry of tails) with tail correction
}

transformed parameters {
  vector[N] convolved_re;
  // variance of each component should be approximately equal to 1
  convolved_re =  sqrt(1 - rho) * theta + sqrt(rho / scaling_factor) * phi;
}

model {
  vector[N_edges] ll;

  y ~ poisson_log(log_E + beta0 + x * betas + convolved_re * sigma);  // co-variates

  // This is the prior for phi! (up to proportionality)
  for(i in 1:N_edges){
      ll[i] =   nig_lpdf( phi[node1[i]] - phi[node2[i]] | 0, 1, etas, zetas, 1);}

  target += sum(ll);


  beta0 ~ normal(0.0, 1.0);
  betas ~ normal(0.0, 1.0);
  theta ~ normal(0.0, 1.0);
  sigma ~ normal(0, 1.0);
  rho ~ beta(0.5, 0.5);
  // soft sum-to-zero constraint on phi)
  sum(phi) ~ normal(0, 0.001 * N);  // equivalent to mean(phi) ~ normal(0,0.001)

  target += -theta_etas*etas;
  target += -theta_zetas*fabs(zetas);
}

generated quantities {
  real logit_rho = log(rho / (1.0 - rho));
  vector[N] RR   = exp(beta0 + x * betas + convolved_re * sigma);
  vector[N] eta  = log_E + log(RR);                                  
  vector[N] mu   = exp(eta);
  vector[N] log_lik;                                                      // Log likelihood for each location
  vector[N] ypred;                                                        // Posterior predictions for each location

  for (n in 1:N){
    log_lik[n] = poisson_log_lpmf(y[n] | eta[n]);
    ypred[n]   = poisson_log_rng(eta[n]);                    
  }
}