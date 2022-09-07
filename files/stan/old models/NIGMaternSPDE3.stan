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
   * Returns the partial sums (the log-likelihood terms from 'start' to 'end' (inclusive) )
   * of the  log-likelihood of the vector X defined by DX=Lambda, where D is a matrix and Lambda
   * is independent NIG noise: log dens_X(x) = log determinant(D) + sum_{i=1}^N log dens_NIG([Dx]_i).
   * This partial sum function is the first argument of the reduce_sum() function which
   * automatically chooses partial sums partitioning based on a dynamic scheduling algorithm,
   * allowing for within-chain parallelization
   *
   * @param slice_Dx The subset of DX for which this partial sum is responsible (slice_DX = DX[start:end])
   * @param start    First term in the partial sum
   * @param end      Last term in the partial sum (inclusive)
   * @param sigma    Standard deviation 
   * @param eta      First flexibility parameter (controls heaviness of tails)
   * @param mu       Second flexibility parameter (controls asymmetry of tails)
   * @param h        Vector containing the distance between locations or area of the basis functions
   * @return         Returns the partial sum of the log likelihood of X from start to end
   */
  real partial_sum_lpdf(real[] slice_DX, int start, int end, real eta, real mu, vector h){
    int N = end - start + 1; 
    real llik_partial;
    vector[N] ll;
    for(i in 1:N){
      ll[i] = NIG_var_correction_lpdf( slice_DX[i] | eta, mu, h[start + i - 1]);}
    llik_partial = sum(ll);
    return llik_partial;
  }


  /**
   * Returns the log-likelihood of independent NIG noise evaluated at X
   *
   * @param X
   * @param etas            First  flexibility parameter (controls heaviness of tails) with tail correction
   * @param mus             Second flexibility parameter (controls asymmetry of tails) with tail correction
   * @param h               Vector containing the distance between locations or area of the basis functions
   * @return                Log-likelihood of X
   */
  real multi_NIG_lpdf(vector X, real etas, real mus, vector h){

    //convert tail corrected parameters (etas, mus) to original parameterization (eta, mu) 
    real eta = etas*(1+mus^2-fabs(mus)*sqrt(1+mus^2))^2;
    real mu  = mus/sqrt(eta);

    //log-likelihood of the random vector X
    real llik_total = reduce_sum(partial_sum_lpdf, to_array_1d(X), 1, eta, mu, h);

    return(llik_total);
  }
}


data{
  int<lower=1> N;                      // Number of basis functions
  int<lower=1> Ny;                     // Number of observations
  vector[Ny] y;                        // Observations
  vector[N] h;                         // Area of the basis functions (Diagonal of matrix C) 
  matrix[Ny,N] C1;                     // Matrix C1
  matrix[N,N] C2;                      // Matrix C2
  vector[N] v;                         // Eigenvalues of (C^-1)G

  real lambda1;                        // Rate constant por PC prior of sigma and kappa
  real lambda2;                        // Rate constant por PC prior of sigma and kappa
  real<lower=0, upper = 1> alphaeta;   // Used for PC prior of etas (Likelihood that there is more than twice as much large marginal events in the NIG latent field compared to Gaussian)
  real<lower=0, upper = 1> alphamu;    // Used for PC prior of mus
}

parameters{
  real<lower=0> sigmae;          // Standard deviation of observations y
  real<lower=0> sigma;           // Standard deviation of NIG noise
  real<lower=0> kappa;           // Spatial range parameter
  real<lower=0> etas;            // First flexibility parameter (controls heaviness of tails) with tail correction
  real mus;                      // Second flexibility parameter (controls asymmetry of tails) with tail correction
  vector[N] Lambda;              // Stochastic weights
}

transformed parameters{
  vector[Ny] my = sigma*diag_post_multiply(C1,inv(kappa^2+v))*C2*Lambda;
}

model{
  real theta_etas;                                             // Rate constant for PC prior of etas

  //observation layer---------------------------
  y ~ normal(my, sigmae);

  //latent field layer--------------------------
  Lambda ~ multi_NIG(etas, mus, h);

  //prior layer---------------------------------
  //prior for sigmae
  sigmae ~ inv_gamma(1, 0.00005); 
  //prior for rho and sigmam
  target += - log(kappa) - lambda1*kappa - lambda2*sigma/kappa;  
  //prior for ng_dist
  theta_etas = -4*log(alphaeta)*kappa^2;
  target +=  log(theta_etas) - theta_etas*etas;
  //prior for mus
  target +=  (2*sqrt(2)*log(alphamu))*fabs(mus);
}

generated quantities { 
  vector[Ny] log_lik = -0.5*log(2*pi())-log(sigmae)-(0.5/sigmae^2)*(y-my)^2;  //Log likelihood at each location
}

