functions {

 /**
   * Returns the log-likelihood of the standardized NIG distribution 
   *
   * @param x 
   * @param sigma  Standard deviation 
   * @param eta    First flexibility parameter (controls heaviness of tails)
   * @param zeta   Second flexibility parameter (controls asymmetry of tails)
   * @param h      Distance between locations or area of the basis function
   * @return       Log-likelihood of the NIG distribution with variance correction
   */
  real NIG_var_correction_lpdf(real x, real eta, real zeta, real h){
    real sigmas     = 1/sqrt(1+zeta^2*eta); //variance correction
    real hyp_alpha  = sqrt(1/eta+zeta^2)/sigmas;
    real hyp_beta   = zeta/sigmas;
    real hyp_delta  = sigmas*sqrt(1/eta)*h;
    real hyp_mu     = -sigmas*zeta*h;
    return (sqrt(hyp_alpha^2 - hyp_beta^2)*hyp_delta + hyp_beta*(x - hyp_mu) + log(hyp_alpha) + 
            log(hyp_delta) - log(pi()) - 0.5*log(hyp_delta^2 + (x - hyp_mu)^2) + 
            log(modified_bessel_second_kind(1, hyp_alpha*sqrt(hyp_delta^2 + (x - hyp_mu)^2)))); 
  }

  /**
   * Returns the log-likelihood of the NIG distribution with standardized and orthogonal parameterization
   * @param x 
   * @param m     Mean
   * @param sigma Standard deviation 
   * @param eta   First flexibility parameter (controls heaviness of tails)
   * @param zeta  Second flexibility parameter (controls asymmetry of tails)
   * @param h     Distance between locations or area of the basis function
   * @return      Log-likelihood of the NIG distribution with standardized and orthogonal parameterization
   */
  real nig_lpdf(real x, real m, real sigma, real etas, real zetas, real h){

    //convert tail correction parameters (etas, zetas) to original parameterization (eta, zeta) 
    real eta = etas*(1+zetas^2-fabs(zetas)*sqrt(1+zetas^2))^2;
    real zeta  = zetas/sqrt(eta);

    return( NIG_var_correction_lpdf( (x-m)/sigma | eta, zeta, h) );
  }


}

data{
  int<lower=1> N;                      // Number of basis functions
  vector[N] x;                         // Observations

  real theta_etas;
  real theta_zetas;
}


parameters{
  real<lower=0> sigma;         // Standard deviation of NIG noise
  real<lower=-1,upper=1> rho;  // Spatial range parameter
  real<lower=0> etas;          // First flexibility parameter (controls heaviness of tails)
  real zetas;                  // Second flexibility parameter (controls asymmetry of tails)
  real c;                      // Drift parameter
}

model{
  real theta_s;

  for (n in 2:N)
    x[n] ~ nig(c + rho*x[n-1], sigma, etas, zetas, 1);

  //prior layer---------------------------------
  //prior for mu
  c ~ normal(0,1);
  //prior for sigma
  theta_s = 0.02/sqrt(1-rho^2);
  target += log(theta_s) - theta_s*sigma;
  //prior for rho
  target += - sqrt(1 - rho)*1.15 - 0.5*log(1 - rho); 
  //prior for etas
  target += - theta_etas*etas;
  //prior for zetas
  target += - theta_zetas*fabs(zetas);

}

generated quantities { 
  vector[N-1] log_lik;
  for(n in 1:N-1)
    log_lik[n] = nig_lpdf(x[n+1] | c + rho*x[n], sigma, etas, zetas, 1); 
}

