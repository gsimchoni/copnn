rn2 <- function(n, sig, n_sig) {
  classes <- rbinom(n, 1, prob = 0.5)
  n1 <- sum(classes)
  n2 <- n - n1
  z <- numeric(n)
  if (n1 > 0) {
    z[classes == 1] <- rnorm(n1, mean = -n_sig * sig, sd = sig)
  }
  if (n2 > 0) {
    z[classes == 0] <- rnorm(n2, mean = n_sig * sig, sd = sig)
  }
  z
}
sig2e <- 1
n_sig <- 3
sig <- sqrt(sig2e / (1 + n_sig^2))
z <- rn2(n, sig, n_sig)

n <- 1000000
alpha <- 1
k <- 1.42625512
par(mar = c(3, 4, 2, 2))
plot(density(rnorm(n, sd = 1)), col = 1, xlim = c(-5, 5), ylim = c(0, 0.7), main = "", xlab = "x", ylab = "f(x)")
lines(density(VGAM::rlaplace(n, scale = 1/sqrt(2))), col = 2, lwd = 2)
lines(density(ordinal::rgumbel(n, location = (sqrt(6)/pi)*digamma(1), scale = sqrt(6)/pi)), col = 3, lwd = 2)
lines(density(VGAM::rlgamma(n, location = -digamma(k), scale = 1, shape = k)), col = 4, lwd = 2)
lines(density(rlogis(n, scale = sqrt(3 / (pi^2)))), col = 5, lwd = 2)
lines(density(sn::rsn(n, xi = -alpha * sqrt(2*1 / (pi * (1 + alpha^2) - 2 * alpha^2)), omega = sqrt(pi * 1 * (1 + alpha^2) / (pi * (1 + alpha^2) - 2 * alpha^2)), alpha = alpha)), col = 6, lwd = 2)
lines(density(z), col = 7, lwd = 2)
legend("topright", legend = c("Gaussian", "Laplace", "Gumbel", "Log-Gamma", "Logistic", "Skew-Norm", "GaussMix"), lty = rep(1, 7), lwd = 2, col = 1:7, bty = "n")

par(mfcol = c(2, 3))
n <- 1000000
alpha <- 1.5
k <- 1.42625512
par(mar = c(3, 4, 2, 2))
plot(density(rnorm(n, sd = 1)), col = 1, xlim = c(-5, 5), ylim = c(0, 0.7), main = "Laplace", xlab = "x", ylab = "f(x)")
lines(density(VGAM::rlaplace(n, scale = 1/sqrt(2))), col = 2, lwd = 2)
plot(density(rnorm(n, sd = 1)), col = 1, xlim = c(-5, 5), ylim = c(0, 0.7), main = "Gumbel", xlab = "x", ylab = "f(x)")
lines(density(ordinal::rgumbel(n, location = (sqrt(6)/pi)*digamma(1), scale = sqrt(6)/pi)), col = 3, lwd = 2)
plot(density(rnorm(n, sd = 1)), col = 1, xlim = c(-5, 5), ylim = c(0, 0.7), main = "LogGamma", xlab = "x", ylab = "")
lines(density(VGAM::rlgamma(n, location = -digamma(k), scale = 1, shape = k)), col = 4, lwd = 2)
plot(density(rnorm(n, sd = 1)), col = 1, xlim = c(-5, 5), ylim = c(0, 0.7), main = "Logistic", xlab = "x", ylab = "")
lines(density(rlogis(n, scale = sqrt(3 / (pi^2)))), col = 5, lwd = 2)
plot(density(rnorm(n, sd = 1)), col = 1, xlim = c(-5, 5), ylim = c(0, 0.7), main = "SkewNorm", xlab = "x", ylab = "")
lines(density(sn::rsn(n, xi = -alpha * sqrt(2*1 / (pi * (1 + alpha^2) - 2 * alpha^2)), omega = sqrt(pi * 1 * (1 + alpha^2) / (pi * (1 + alpha^2) - 2 * alpha^2)), alpha = alpha)), col = 6, lwd = 2)
plot(density(rnorm(n, sd = 1)), col = 1, xlim = c(-5, 5), ylim = c(0, 0.7), main = "GaussMix", xlab = "x", ylab = "")
lines(density(z), col = 7, lwd = 2)
# legend("topright", legend = c("Gaussian", "Laplace", "Gumbel", "Log-Gamma", "Logistic", "Skew-Norm", "GaussMix"), lty = rep(1, 7), lwd = 2, col = 1:7, bty = "n")
