path.make.D <- function(n, n.sim, etas, mus, h=1, Dm, m=0){ #NIG noise
  
  eta = etas*(1+mus^2-abs(mus)*sqrt(1+mus^2))^2;
  mu  = mus/sqrt(eta);
  
  paths  <- matrix(0, nrow = n.sim, ncol = n)
  sigmas = 1/sqrt(1+mu^2*eta)
  for (i in 1:n.sim){
    V         <- rinvGauss(n-m,nu=h*1,lambda = h^2/eta)
    L         <- rnorm(n-m,sigmas*mu*(V-1) ,sigmas^2*V)
    paths[i,] <- (Dm %*% L)[,1] 
  }
  return(paths)
}

rNIG <- function(n, m, sigma, etas, mus, h = 1){
  eta    <- etas*(1+mus^2-abs(mus)*sqrt(1+mus^2))^2
  mu     <- mus/sqrt(eta)
  sigmas <- 1/sqrt(1+mu^2*eta)
  
  V         <- rinvGauss(n, nu=h, lambda = h^2/eta)
  L         <- rnorm(n, mu*(V-h), V)
  
  return(m + sigmas*L)
}

RW1.matrix <-function(n, boundary.condition=FALSE, type = 1){
  
  D1 <- bandSparse(n-1, n,
                   (-1):1,
                   list(rep(0, n-2),
                        rep(-1, n-1),
                        rep(1, n-1)))
  
  if(boundary.condition){
    if(type==1){
      D1 <- rbind(D1,c(1,rep(0,n-2),-1))
    }
    if(type==2){
      D1 <- rbind(c(1,rep(0,n-1)),D1)
    }
  }
  
  return(D1)
}


AR1.matrix <-function(n, rho){
  
  D <- bandSparse(n-1, n,
                  (-1):1,
                  list(rep(0, n-1),
                       rep(-rho, n),
                       rep(1, n-1)))
  
  D <- rbind(c(sqrt(1-rho^2),rep(0,n-1)),D)
  
  return(D)
}


RW2.matrix <-function(n, boundary.condition=FALSE, type=1){
  
  D2 <- bandSparse(n-2, n,
                   0:2,
                   list(rep(1, n-2),
                        rep(-2, n-1),
                        rep(1, n-1)))
  
  if(boundary.condition){
    if(type==1){
      D2 <- rbind(D2,c(1, rep(0,n-3), 1, -2))
      D2 <- rbind(D2,c(-2, 1, rep(0,n-3), 1))
    }
    if(type==2){
      D2 <- rbind(D2,c(1, rep(0,n-3), 1, -2))
      D2 <- rbind(D2,c(1, 0, rep(0,n-3), -1))
    }
    
  }
  
  return(D2)
}

SPDE.matrix <- function(loc,kappa){
  mesh <- inla.mesh.1d(loc)
  fem  <- inla.mesh.fem(mesh, order=1)
  
  
  C  <- fem$c1
  G  <- fem$g1
  C0   <- fem$c0
  
  K  <- kappa^2*C0 + G
  
  
  return(list(C0=C0, K=K, G=G))
}


#Draws posterior samples from V, given samples from: 
#X      Standardized latent NIG field
#kappa  Spatial range parameter
#G      D = diag(kappa^2*h) + G
#etas   First flexibility parameter
#zetas    Second flexibility parameter
#h      vector h
Vposterior <- function(X, kappa, G, etas, zetas, h){
  
  library(GIGrvg)
  n_draws   <- nrow(X)
  
  eta = etas*(1+zetas^2-abs(zetas)*sqrt(1+zetas^2))^2
  zeta  = zetas/sqrt(eta)
  
  if(kappa[1]==0){
    D  <- G
    n  <- nrow(D)
    DX = t(D%*%t(X))
  }
  else{
    n  <- ncol(X)
    DX <- matrix(NA, nrow = n_draws, ncol = n)
    for(i in 1:n_draws){
      D         <- Diagonal(n, kappa[i]^2*h) + G
      DX[i,]    <- as.numeric(D%*%X[i,])}
  }

  #V = matrix(NA, nrow= n_draws, ncol = n)
  #for(i in 1:n){
  #  V[,i] <- GIGrvg::rgig(n_draws, lambda= -1, psi = 1/eta+zeta^2, chi = h[i]^2/eta + (DX[,i] + zeta*h[i])^2)}
  
  V = matrix(NA, nrow= n_draws, ncol = n)
  for(i in 1:n_draws){
    for(j in 1:n){
      V[i,j] <- GIGrvg::rgig(1, lambda= -1, psi = 1/eta[i]+zeta[i]^2, chi = h[j]^2/eta[i] + (DX[i,j] + zeta[i]*h[j])^2)
    }
  }
  
  return(V%*%diag(1/h)) 
}

VposteriorSAR <- function(X, rho, W, etas, zetas, h){
  
  library(GIGrvg)
  n_draws   <- nrow(X)
  
  eta = etas*(1+zetas^2-abs(zetas)*sqrt(1+zetas^2))^2
  zeta  = zetas/sqrt(eta)
  sigmas = 1/sqrt(1+zetas^2)
  
  n  <- ncol(X)
  DX <- matrix(NA, nrow = n_draws, ncol = n)
  for(i in 1:n_draws){
    D         <- diag(rep(1,n)) - rho[i]*W
    DX[i,]    <- as.numeric(D%*%X[i,])}
  
  #V = matrix(NA, nrow= n_draws, ncol = n)
  #for(i in 1:n){
  #  V[,i] <- GIGrvg::rgig(n_draws, lambda= -1, psi = 1/eta+zeta^2, chi = h[i]^2/eta + (DX[,i]/sigmas + zeta*h[i])^2)}
  
  V = matrix(NA, nrow= n_draws, ncol = n)
  for(i in 1:n_draws){
    for(j in 1:n){
      V[i,j] <- GIGrvg::rgig(1, lambda= -1, psi = 1/eta[i]+zeta[i]^2, chi = h[j]^2/eta[i] + (DX[i,j] + zeta[i]*h[j])^2)
    }
  }
  
  return(V%*%diag(1/h)) 
}

VposteriorCAR <- function(X, node1, node2, etas, zetas){
  
  n_draws   <- nrow(X)
  n_edges   <- length(node1)
  
  eta = etas*(1+zetas^2-abs(zetas)*sqrt(1+zetas^2))^2
  zeta  = zetas/sqrt(eta)
  sigmas = 1/sqrt(1+zetas^2)
  
  #V = matrix(NA, nrow= n_draws, ncol = n_edges)
  #for(i in 1:n_edges){
  #  V[,i] <- GIGrvg::rgig(n_draws, lambda= -1, psi = 1/eta+zeta^2, chi = 1/eta + ((X[,node1[i]]-X[,node2[i]])/sigmas + zeta)^2)}
  
  V = matrix(NA, nrow= n_draws, ncol = n_edges)
  for(i in 1:n_draws){
    for(j in 1:n_edges){
      V[i,j] <- GIGrvg::rgig(1, lambda= -1, psi = 1/eta[i]+zeta[i]^2, chi = 1/eta[i] + ((X[i,node1[j]]-X[i,node2[j]])/sigmas + zeta[i])^2)
    }
  }
  
  return(V) 
}

#Convert INLA mesh object to sp object
#taken from https://groups.google.com/g/r-inla-discussion-group/c/z1n1exlZrKM
inla.mesh2sp <- function(mesh) {
  crs <- inla.CRS(inla.CRSargs(mesh$crs))
  isgeocentric <- identical(inla.as.list.CRS(crs)[["proj"]], "geocent")
  if (isgeocentric || (mesh$manifold == "S2")) {
    stop(paste0(
      "'sp' doesn't support storing polygons in geocentric coordinates.\n",
      "Convert to a map projection with inla.spTransform() before
calling inla.mesh2sp()."))
  }
  
  triangles <- SpatialPolygonsDataFrame(
    Sr = SpatialPolygons(lapply(
      1:nrow(mesh$graph$tv),
      function(x) {
        tv <- mesh$graph$tv[x, , drop = TRUE]
        Polygons(list(Polygon(mesh$loc[tv[c(1, 3, 2, 1)],
                                       1:2,
                                       drop = FALSE])),
                 ID = x)
      }
    ),
    proj4string = crs
    ),
    data = as.data.frame(mesh$graph$tv[, c(1, 3, 2), drop = FALSE]),
    match.ID = FALSE
  )
  vertices <- SpatialPoints(mesh$loc[, 1:2, drop = FALSE], proj4string = crs)
  
  list(triangles = triangles, vertices = vertices)
}


#Allow for decreasing plot legends. I do not like the default increasing scale of Leaflet
#Taken from https://stackoverflow.com/questions/40276569/reverse-order-in-r-leaflet-continuous-legend
addLegend_decreasing <- function (map, position = c("topright", "bottomright", "bottomleft","topleft"),
                                  pal, values, na.label = "NA", bins = 7, colors,
                                  opacity = 0.5, labels = NULL, labFormat = labelFormat(),
                                  title = NULL, className = "info legend", layerId = NULL,
                                  group = NULL, data = getMapData(map), decreasing = FALSE) {
  
  position <- match.arg(position)
  type <- "unknown"
  na.color <- NULL
  extra <- NULL
  if (!missing(pal)) {
    if (!missing(colors))
      stop("You must provide either 'pal' or 'colors' (not both)")
    if (missing(title) && inherits(values, "formula"))
      title <- deparse(values[[2]])
    values <- evalFormula(values, data)
    type <- attr(pal, "colorType", exact = TRUE)
    args <- attr(pal, "colorArgs", exact = TRUE)
    na.color <- args$na.color
    if (!is.null(na.color) && col2rgb(na.color, alpha = TRUE)[[4]] ==
        0) {
      na.color <- NULL
    }
    if (type != "numeric" && !missing(bins))
      warning("'bins' is ignored because the palette type is not numeric")
    if (type == "numeric") {
      cuts <- if (length(bins) == 1)
        pretty(values, bins)
      else bins
      if (length(bins) > 2)
        if (!all(abs(diff(bins, differences = 2)) <=
                 sqrt(.Machine$double.eps)))
          stop("The vector of breaks 'bins' must be equally spaced")
      n <- length(cuts)
      r <- range(values, na.rm = TRUE)
      cuts <- cuts[cuts >= r[1] & cuts <= r[2]]
      n <- length(cuts)
      p <- (cuts - r[1])/(r[2] - r[1])
      extra <- list(p_1 = p[1], p_n = p[n])
      p <- c("", paste0(100 * p, "%"), "")
      if (decreasing == TRUE){
        colors <- pal(rev(c(r[1], cuts, r[2])))
        labels <- rev(labFormat(type = "numeric", cuts))
      }else{
        colors <- pal(c(r[1], cuts, r[2]))
        labels <- rev(labFormat(type = "numeric", cuts))
      }
      colors <- paste(colors, p, sep = " ", collapse = ", ")
    }
    else if (type == "bin") {
      cuts <- args$bins
      n <- length(cuts)
      mids <- (cuts[-1] + cuts[-n])/2
      if (decreasing == TRUE){
        colors <- pal(rev(mids))
        labels <- rev(labFormat(type = "bin", cuts))
      }else{
        colors <- pal(mids)
        labels <- labFormat(type = "bin", cuts)
      }
    }
    else if (type == "quantile") {
      p <- args$probs
      n <- length(p)
      cuts <- quantile(values, probs = p, na.rm = TRUE)
      mids <- quantile(values, probs = (p[-1] + p[-n])/2, na.rm = TRUE)
      if (decreasing == TRUE){
        colors <- pal(rev(mids))
        labels <- rev(labFormat(type = "quantile", cuts, p))
      }else{
        colors <- pal(mids)
        labels <- labFormat(type = "quantile", cuts, p)
      }
    }
    else if (type == "factor") {
      v <- sort(unique(na.omit(values)))
      colors <- pal(v)
      labels <- labFormat(type = "factor", v)
      if (decreasing == TRUE){
        colors <- pal(rev(v))
        labels <- rev(labFormat(type = "factor", v))
      }else{
        colors <- pal(v)
        labels <- labFormat(type = "factor", v)
      }
    }
    else stop("Palette function not supported")
    if (!any(is.na(values)))
      na.color <- NULL
  }
  else {
    if (length(colors) != length(labels))
      stop("'colors' and 'labels' must be of the same length")
  }
  legend <- list(colors = I(unname(colors)), labels = I(unname(labels)),
                 na_color = na.color, na_label = na.label, opacity = opacity,
                 position = position, type = type, title = title, extra = extra,
                 layerId = layerId, className = className, group = group)
  invokeMethod(map, data, "addLegend", legend)
}

#taken from https://mc-stan.org/users/documentation/case-studies/icar_stan.html
mungeCARdata4stan = function(adjBUGS,numBUGS) {
  N = length(numBUGS);
  nn = numBUGS;
  N_edges = length(adjBUGS) / 2;
  node1 = vector(mode="numeric", length=N_edges);
  node2 = vector(mode="numeric", length=N_edges);
  iAdj = 0;
  iEdge = 0;
  for (i in 1:N) {
    for (j in 1:nn[i]) {
      iAdj = iAdj + 1;
      if (i < adjBUGS[iAdj]) {
        iEdge = iEdge + 1;
        node1[iEdge] = i;
        node2[iEdge] = adjBUGS[iAdj];
      }
    }
  }
  return (list("N"=N,"N_edges"=N_edges,"node1"=node1,"node2"=node2));
}


