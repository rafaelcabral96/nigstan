functions {

  /**
   * Returns the log-likelihood of the Generalized Hyperbolic (GH)
   * distribution with parameter lambda=-1/2
   *
   * @param  x 
   * @param  alpha 
   * @param  beta
   * @param  delta
   * @param  mu
   * @return Log-likelihood of the generalized hyperbolic distribution
   */
  real hyperbolic_lpdf(real x, real alpha, real beta, real delta, real mu){
    return (sqrt(alpha^2 - beta^2)*delta + beta*(x - mu) + log(alpha) + 
            log(delta) - log(pi()) - 0.5*log(delta^2 + (x - mu)^2) + 
            log(modified_bessel_second_kind(1, alpha*sqrt(delta^2 + (x - mu)^2)))); 
  }

  /**
   * Returns the log-likelihood of the NIG distribution with variance 
   * correction at x (Var[x] = sigma^2)
   *
   * @param x 
   * @param sigma Standard deviation 
   * @param eta   First flexibility parameter (controls heaviness of tails)
   * @param mu    Second flexibility parameter (controls asymmetry of tails)
   * @param h     Distance between locations or area of the basis function
   * @return      Log-likelihood of the NIG distribution with variance correction
   */
  real NIG_lpdf(real x, real sigma, real eta, real mu, real h){
    real sigmas     = sigma/sqrt(1+mu^2*eta); //variance correction
    real hyp_alpha  = sqrt(1/eta+mu^2)/sigmas;
    real hyp_beta   = mu/sigmas;
    real hyp_delta  = sigmas*sqrt(1/eta)*h;
    real hyp_mu     = -sigmas*mu*h;
    return( hyperbolic_lpdf(x | hyp_alpha, hyp_beta, hyp_delta, hyp_mu) );
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
  real partial_sum_lpdf(real[] slice_DX, int start, int end,
                        real sigma, real eta, real mu, vector h){
    int N = end - start + 1; 
    real llik_partial;
    vector[N] ll;
    for(i in 1:N){
      ll[i] = NIG_lpdf( slice_DX[i] | sigma, eta, mu, h[start + i - 1]);}
    llik_partial = sum(ll);
    return llik_partial;
  }

  /**
   * Returns a vector with the non-zero elements of the matrix D. More efficient than
   * csr_extract_w() from STAN, because it already knows where the non-zero elements
   * are located based on Dv and Du. Dv and Du are the output of csr_extract_v(D) and
   * csr_extract_u(D) which can be computed once in the transformed data block. Can lead to
   * a speed up if the matrix D changes at every iteration (if it depends on a parameter)
   * otherwise you just need to compute this once in the transformed data block.
   *
   * @param D       Matrix D
   * @param sizew   Number of non-zero elements in D
   * @param Dv      Array of integer column indices for the non-zero values in D
   * @param Du      Array of integer indices indicating where a given row’s values start
   * @return        Vector of the non-zero elements of matrix D
   */
  vector csr_extract_w_efficient(matrix D, int sizew, int[] Dv, int[] Du){
    vector[sizew] Dw;
    int j = 2;
    int crow = 1;
    int next_row_index = Du[2];
    for(i in 1:sizew){
      while(i==next_row_index){
        j=j+1;
        next_row_index = Du[j];
        crow = crow + 1;}
      Dw[i] = D[crow, Dv[i]];}
    return(Dw);    
  }
  

  /**
   * Returns the log-likelihood of X where DX=Lambda, where D is a matrix and 
   * Lambda is a vector of independent NIG noise.
   *
   * @param X
   * @param D               Matrix D
   * @param Dv              Array of integer column indices for the non-zero values in D
   * @param Du              Array of integer indices indicating where a given row’s values start
   * @param sizes           Array of integers containing the number of rows, collums and number non-zero elements of D
   * @param sigma           Standard deviation 
   * @param etas            First  flexibility parameter (controls heaviness of tails) with tail correction
   * @param mus             Second flexibility parameter (controls asymmetry of tails) with tail correction
   * @param h               Vector containing the distance between locations or area of the basis functions
   * @param compute_det     Compute the log determinant of D if D depends on parameter (compute_det=1). Otherwise set compute_det=0.
   * @return                Log-likelihood of X
   */
  real modelNIG_lpdf(vector X, matrix D, int[] Dv, int[] Du, int[] sizes, 
                     real sigma, real etas, real mus, vector h, 
                     int compute_det){

    int N = sizes[1];
    int M = sizes[2];
    int sizew = sizes[3];

    vector[sizew] Dw = csr_extract_w_efficient(D, sizew, Dv, Du); //non-zero elements of D

    real DX[N] = to_array_1d(csr_matrix_times_vector( N, M, Dw , Dv , Du , X)); //sparse multiplication of D*X

    //convert tail correction parameters (etas, mus) to original parameterization (eta, mu) 
    real eta = etas*(1+mus^2-fabs(mus)*sqrt(1+mus^2))^2;
    real mu  = mus/sqrt(eta);

    //log-likelihood of the random vector X
    real llik_total = reduce_sum(partial_sum_lpdf, DX, 1, sigma, eta, mu, h);
     
    return(llik_total);
   }

   real interpolation(real x, vector x_pred, vector y_pred){
    //https://discourse.mc-stan.org/t/linear-interpolation-and-searchsorted-in-stan/13318/5
    int K = rows(x_pred);
    vector[K] deltas = x - x_pred;
    real ans;
    real t;
    real w;
    int i;
    if(x < x_pred[1]){
      real slope = (y_pred[2] - y_pred[1])/(x_pred[2] - x_pred[1]);
      ans = y_pred[1] + slope*(x - x_pred[1]);}
    else if(x > x_pred[K]){
      real slope = (y_pred[K] - y_pred[K-1])/(x_pred[K] - x_pred[K-1]);
      ans = y_pred[K] + slope*(x -x_pred[K]);
    }
    else{
      i = sort_indices_asc(fabs(deltas))[1];  //this is which.min()
      if(deltas[i]<=0) i -= 1;
      ans = y_pred[i];
      real x1 = x_pred[i];
      real x2 = x_pred[i + 1];
      real y1 = y_pred[i];
      real y2 = y_pred[i + 1];
      t = (x-x1)/(x2-x1);
      w = 1-t;
      //w = 1 - 3*pow(t,2) + 2*pow(t,3);
      //w = 1/(1 + exp(1/(1-t) - 1/t));
      ans = w*y1 + (1-w)*y2;}
    return ans;
   }
}

data{
  int<lower=1> N;
  int<lower=1> Ny;
  vector[Ny] y;
  vector[N] h;
  matrix[N,N] G;
  matrix[Ny,N] A;

  int inter_points; 
  vector[inter_points] kappa_log;
  vector[inter_points] det_log;

  real lambda1;
  real lambda2;
  real<lower=0, upper = 1> alphaeta;
  real<lower=0> thetamu;
}

transformed data{
  matrix[N,N] Graph = add_diag(G, h);
  int sizew = rows(csr_extract_w(Graph));
  int Dv[size(csr_extract_u(Graph))];
  int Du[size(csr_extract_u(Graph))];
  int sizes[3];

  Dv = csr_extract_v(Graph);
  Du = csr_extract_u(Graph);
  sizes[1] = N;
  sizes[2] = N;
  sizes[3] = sizew;
}

parameters{
  real<lower=0> sigmae;
  real<lower=0> sigma;
  real<lower=0> kappa;
  real<lower=0> etas;
  real mus;
  vector[N] W;
}

transformed parameters{
  vector[N] my = sigma*A*W;
}

model{
  matrix[N,N] D;
  real theta_etas;

  //observation layer---------------------------
  y ~ normal(my, sigmae);

  //latent field layer--------------------------
  D = add_diag(G, kappa^2*h);
  W ~ modelNIG(D, Dv, Du, sizes, 1, etas, mus, h, 1);
  target += interpolation(log(kappa), kappa_log, det_log);

  //prior layer---------------------------------
  //prior for sigmae
  sigmae  ~ inv_gamma(1.0, 1.0);
  //prior for rho and sigmam
  target += - log(kappa) - lambda1*kappa - lambda2*sigma/kappa;  
  //prior for ng_dist
  theta_etas = -4*log(alphaeta)*kappa^2;
  target += - log(theta_etas) - theta_etas*etas;
  //prior for mus
  target += - thetamu*fabs(mus);

}


generated quantities {
  vector[Ny] log_lik;

  for (n in 1:Ny){
    log_lik[n] = normal_lpdf(y[n] | my[n], sigmae);}
}


