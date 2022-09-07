functions{
  real invGauss_lpdf(real x, real mu, real lambda){
    return(0.5*log(lambda/(2*pi()*x^3)) - lambda*(x-mu)^2/(2*mu^2*x));}
}

data {
  int<lower=1> Ny;
  int<lower=1> N;
  vector[Ny] y;
  vector[N] h;
  matrix[N,Ny] D;

  real<lower=0> thetaetas;

  real<lower=0> meanx;
  real<lower=0> sigmay;
}

parameters {
  real<lower=0> etas;
  vector<lower=0>[N] V;
  vector[Ny] x;
}



model {
  //data layer
  y ~ normal(x, sigmay);

  //latent layer
  target +=  -0.5*sum(log(V)) -0.5*sum(square(D*x) ./ V); //Simplify

  //Mixing distributions
  for (i in 1:N)
    V[i] ~ invGauss(1, 1/etas);

  // priors layer
  target += -thetaetas*etas;
}



