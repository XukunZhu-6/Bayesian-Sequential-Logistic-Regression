functions {
  vector conditional_to_prob(row_vector logit_p) {
    int J = num_elements(logit_p) + 1;
    vector[J] pi;
    vector[J - 1] conditional_prob;
    conditional_prob = to_vector(inv_logit(logit_p));
    real prod = 1.0;
    for (j in 1:(J - 1)) {
      pi[j] = prod * (1 - conditional_prob[j]);
      prod *= conditional_prob[j];
    }
    pi[J] = prod;
    return pi;
  }
}

data {
  int<lower=1> N;                    
  int<lower=1> P_tv;                  
  int<lower=1> P_fixed;               
  int<lower=2> J;                  
  matrix[N, P_tv] X_tv_stage1;
  matrix[N, P_tv] X_tv_stage2;
  matrix[N, P_tv] X_tv_stage3;
  matrix[N, P_tv] X_tv_stage4;
  matrix[N, P_fixed] X_fixed;
  matrix[P_tv, J-1] mask_tv;
  array[N] int<lower=1, upper=J> Y;
}

parameters {
  vector[P_tv] beta_tv_global;
  matrix[P_tv, J-1] beta_tv_local;
  vector<lower=0>[P_tv] sigma_tv_global;
  vector<lower=0>[P_tv] sigma_tv_local;
  vector[P_fixed] beta_fixed_global;
  matrix[P_fixed, J-1] beta_fixed_local;
  vector<lower=0>[P_fixed] sigma_fixed_global;
  vector<lower=0>[P_fixed] sigma_fixed_local;
}

transformed parameters {
  matrix[P_tv, J-1] beta_tv_local_constrained;
  matrix[P_tv, J-1] beta_tv_total;
  vector[P_tv] sum_mask_tv;
  vector[P_tv] mean_tv_local;
  
  matrix[P_fixed, J-1] beta_fixed_local_constrained;
  matrix[P_fixed, J-1] beta_fixed_total;
  vector[P_fixed] mean_fixed_local;
  for (p in 1:P_tv) {
    sum_mask_tv[p] = sum(mask_tv[p, ]);
  }
  for (p in 1:P_tv) {
    real sum_local = 0;
    for (j in 1:(J-1)) {
      sum_local += mask_tv[p, j] * beta_tv_local[p, j];
    }
    if (sum_mask_tv[p] > 0) {
      mean_tv_local[p] = sum_local / sum_mask_tv[p];
    } else {
      mean_tv_local[p] = 0;
    }
  }
  for (p in 1:P_tv) {
    for (j in 1:(J-1)) {
      if (mask_tv[p, j] == 1) {
        beta_tv_local_constrained[p, j] = beta_tv_local[p, j] - mean_tv_local[p];
        beta_tv_total[p, j] = beta_tv_global[p] + beta_tv_local_constrained[p, j];
      } else {
        beta_tv_local_constrained[p, j] = 0;
        beta_tv_total[p, j] = 0;
      }
    }
  }
  for (p in 1:P_fixed) {
    mean_fixed_local[p] = mean(beta_fixed_local[p, ]);
  }
  
  for (p in 1:P_fixed) {
    for (j in 1:(J-1)) {
      beta_fixed_local_constrained[p, j] = beta_fixed_local[p, j] - mean_fixed_local[p];
      beta_fixed_total[p, j] = beta_fixed_global[p] + beta_fixed_local_constrained[p, j];
    }
  }
}

model {
  sigma_tv_global ~ student_t(3, 0, 3);
  sigma_tv_local ~ student_t(3, 0, 3);
  sigma_fixed_global ~ student_t(3, 0, 3);
  sigma_fixed_local ~ student_t(3, 0, 3);
  for (p in 1:P_tv) {
    beta_tv_global[p] ~ normal(0, sigma_tv_global[p]);
    beta_tv_local[p, ] ~ normal(0, sigma_tv_local[p]);
  }
  

  for (p in 1:P_fixed) {
    beta_fixed_global[p] ~ normal(0, sigma_fixed_global[p]);
    beta_fixed_local[p, ] ~ normal(0, sigma_fixed_local[p]);
  }
  

  for (n in 1:N) {
    vector[J-1] logit_p; 
    
    logit_p[1] = X_tv_stage1[n] * beta_tv_total[, 1] 
               + X_fixed[n] * beta_fixed_total[, 1];
    
    logit_p[2] = X_tv_stage2[n] * beta_tv_total[, 2]
               + X_fixed[n] * beta_fixed_total[, 2];
    
    logit_p[3] = X_tv_stage3[n] * beta_tv_total[, 3]
               + X_fixed[n] * beta_fixed_total[, 3];
    
    logit_p[4] = X_tv_stage4[n] * beta_tv_total[, 4]
               + X_fixed[n] * beta_fixed_total[, 4];
    
    vector[J] pi = conditional_to_prob(to_row_vector(logit_p));
    Y[n] ~ categorical(pi);
  }
}

generated quantities {
  vector[N] log_lik;
  
  for (n in 1:N) {
    vector[J-1] logit_p;
    logit_p[1] = X_tv_stage1[n] * beta_tv_total[, 1] + X_fixed[n] * beta_fixed_total[, 1];
    logit_p[2] = X_tv_stage2[n] * beta_tv_total[, 2] + X_fixed[n] * beta_fixed_total[, 2];
    logit_p[3] = X_tv_stage3[n] * beta_tv_total[, 3] + X_fixed[n] * beta_fixed_total[, 3];
    logit_p[4] = X_tv_stage4[n] * beta_tv_total[, 4] + X_fixed[n] * beta_fixed_total[, 4];
    
    vector[J] pi = conditional_to_prob(to_row_vector(logit_p));
    log_lik[n] = categorical_lpmf(Y[n] | pi);
  }
}
