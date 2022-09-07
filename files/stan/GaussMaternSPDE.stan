functions {

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

 
}

data{
  int<lower=1> N;                      // Number of basis functions
  int<lower=1> Ny;                     // Number of observations
  vector[Ny] y;                        // Observations
  vector[N] h;                         // Area of the basis functions (Diagonal of matrix C) 
  matrix[N,N] G;                       // Matrix G
  matrix[Ny,N] A;                      // Projector matrix
  vector[N] v;                         // Eigenvalues of C^{-1}*G
  
  real lambda1;                        // Rate constant por PC prior of sigma and kappa
  real lambda2;                        // Rate constant por PC prior of sigma and kappa
}

transformed data{
  matrix[N,N] Graph = add_diag(G, h);         // Underlying graph (non-zero values of D = kappa^2*C+G)
  int sizew = rows(csr_extract_w(Graph));     // Number of non-zero values of matrix D
  int Dv[size(csr_extract_u(Graph))];         // Collumn indeces (in compressed row storage format)
  int Du[size(csr_extract_u(Graph))];         // Row indeces (in compressed row storage format)
  int sizes[3];                               // Vector containing number of rows, collumns, and number of non-zero elements in D

  Dv = csr_extract_v(Graph);
  Du = csr_extract_u(Graph);
}

parameters{
  real<lower=0> sigmae;          // Standard deviation of observations y
  real<lower=0> sigma;           // Standard deviation of NIG noise
  real<lower=0> kappa;           // Spatial range parameter
  vector[N] W;                   // Stochastic weights
}

model{
  matrix[N,N] D;
  vector[sizew] Dw;
  vector[N] DW;

  //observation layer---------------------------
  y ~ normal(sigma*A*W, sigmae);

  //latent field layer--------------------------
  D  = add_diag(G, kappa^2*h);                                //D = kappa^2*C+ G
  Dw = csr_extract_w_efficient(D, sizew, Dv, Du);             //extract non-zero elements from D
  DW = csr_matrix_times_vector(N, N, Dw , Dv , Du , W);
  target += -0.5*sum(square(DW) ./ h);   
  target += sum(log(h)+log(kappa^2 + v));                     //Log determinant of D

  //prior layer---------------------------------
  //prior for sigmae
  sigmae  ~ inv_gamma(1, 0.00005);
  //prior for rho and sigmam
  target += - log(kappa) - lambda1*kappa - lambda2*sigma/kappa;  

}

generated quantities { 
  vector[Ny] my = sigma*A*W;                                                  //mean of y
  vector[Ny] log_lik = -0.5*log(2*pi())-log(sigmae)-(0.5/sigmae^2)*square(y-my);  //Log likelihood at each location
}
