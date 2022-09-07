
data{
  int<lower=1> N;                      // Number of basis functions
  int<lower=1> Ny;                     // Number of observations
  vector[Ny] y;                        // Observations
  vector[N] h;                         // Area of the basis functions (Diagonal of matrix C) 
  matrix[Ny,N] A;                      // Projector matrix A
  matrix[N,N] C1;                      // Matrix C1
  matrix[N,N] C2;                      // Matrix C2
  vector[N] v;                         // Eigenvalues of (C^-1)G

  real lambda1;                        // Rate constant por PC prior of sigmax and kappa
  real lambda2;                        // Rate constant por PC prior of sigmax and kappa
}

parameters{
  real<lower=0> sigmae;          // Standard deviation of observations y
  real<lower=0> sigmax;          // Standard deviation of NIG noise
  real<lower=0> kappa;           // Spatial range parameter
  vector[N] Lambda;              // Stochastic weights

}

transformed parameters{
  vector[N] w  = diag_post_multiply(C1,inv(kappa^2+v))*C2*Lambda;   //stochastic weights
  vector[Ny] my = sigmax*A*w;                                       //mean of observations y
} 

model{
  real theta_etas;                                         

  //observation layer---------------------------
  y ~ normal(my, sigmae);

  //latent field layer--------------------------
  target += -0.5*sum(Lambda^2 ./ h);

  //prior layer---------------------------------
  //prior for sigmae
  sigmae ~ inv_gamma(1, 0.00005); 
  //prior for rho and sigmam
  target += - log(kappa) - lambda1*kappa - lambda2*sigmax/kappa;  

}

generated quantities {
  vector[Ny] log_lik = -0.5*log(2*pi())-log(sigmae)-(0.5/sigmae^2)*(y-my)^2;  //Log likelihood at each location
}