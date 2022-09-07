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
   * @param sigmax   Standard deviation 
   * @param eta      First flexibility parameter (controls heaviness of tails)
   * @param zeta     Second flexibility parameter (controls asymmetry of tails)
   * @param h        Vector containing the distance between locations or area of the basis functions
   * @return         Returns the partial sum of the log likelihood of X from start to end
   */
  real partial_sum_lpdf(real[] slice_DX, int start, int end, real eta, real zeta, vector h){
    int N = end - start + 1; 
    real llik_partial;
    vector[N] ll;
    for(i in 1:N){
      ll[i] = NIG_var_correction_lpdf( slice_DX[i] | eta, zeta, h[start + i - 1]);}
    llik_partial = sum(ll);
    return llik_partial;
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

  /**
   * Returns the log-likelihood of independent NIG noise evaluated at x with standardized and orthogonal parameterization
   *
   * @param x
   * @param etas            First  flexibility parameter (controls heaviness of tails) 
   * @param zetas           Second flexibility parameter (controls asymmetry of tails)
   * @param h               Vector containing the distance between locations or area of the basis functions
   * @return                Log-likelihood of x with standardized and orthogonal parameterizations
   */
  real nig_multi_lpdf(vector x, real etas, real zetas, vector h){

    //convert tail corrected parameters (etas, zetas) to original parameterization (eta, zeta) 
    real eta = etas*(1+zetas^2-fabs(zetas)*sqrt(1+zetas^2))^2;
    real zeta  = zetas/sqrt(eta);

    //log-likelihood of the random vector x
    real llik_total = reduce_sum(partial_sum_lpdf, to_array_1d(x), 1, eta, zeta, h);

    return(llik_total);
  }


  real nig_model_lpdf(vector X, matrix D, real etas, real zetas, vector h, int compute_det){

    //convert tail correction parameters (etas, zetas) to original parameterization (eta, zeta) 
    real eta = etas*(1+zetas^2-fabs(zetas)*sqrt(1+zetas^2))^2;
    real zeta  = zetas/sqrt(eta);

    //log-likelihood of the random vector X
    real llik_total = reduce_sum(partial_sum_lpdf, to_array_1d(D*X), 1, eta, zeta, h);

    //If D depends on parameter add log determinant
    if(compute_det==1){
      llik_total  += 2*sum(log(diagonal(cholesky_decompose(D))));    //Only works for spd matrices!!! 
    }
     
    return(llik_total);
   }

}

data {
  int<lower=1> Ny;
  int<lower=1> N;
  vector[Ny] y;
  vector[N] h;
  matrix[N,Ny] D;

  real<lower=0> thetaetas;
  real<lower=0> thetamus;

  real<lower=0> meanx;
  real<lower=0> sigmay;
}

parameters {
  real<lower=0> etas;
  vector[Ny] x;
}


model {
  //observation layer----------------
  y ~ normal(x, sigmay);

  //latent field layer---------------
  x ~ nig_model(D, etas, 0, h, 0);

  //prior layer----------------------
  target += -thetaetas*etas;
}

