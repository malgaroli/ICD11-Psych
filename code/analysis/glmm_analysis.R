# =============================================================================
# glmm_reanalysis_v2.R
# GLMM reanalysis of the LLM vs clinician ICD-11 diagnostic vignette study.
#
# Input : glmm_input_clinician_llm_long.csv  (from export_llm_ratings_long.py)
#         one row per rating: vignette_id, rater_id, rater_group, rater_type,
#         domain, language, ground_truth, answer, correct
#
# ANALYSES
#   (1) Primary GLMM
#         correct ~ rater_group + domain + (1 | vignette_id) + (0 + is_clinician | rater_id)
#       Family A: 10 LLM-minus-clinician contrasts (superiority / NI / equivalence)
#       Family B: all pairwise contrasts among rater groups (optional; only
#                 needed if the manuscript makes LLM-vs-LLM ranking claims)
#   (2) Domain moderation: LRT of primary vs rater_group * domain model
#   (3) Sensitivity S5a: two-stage cluster bootstrap (vignettes x clinicians),
#       ALL clinician languages, with a bootstrap max-t band
#   (4) Sensitivity S5b: clinician group split by profession; LRT pooled vs
#       split; each LLM vs each profession
#
# ESTIMAND. Accuracy of a typical rater of each group (clinician random
# intercept = 0; LLMs have no rater random intercept), averaged with equal
# weight over the 43 vignettes at their estimated difficulties
# (re.form = ~(1 | vignette_id)). Differences are accuracy-scale risk
# differences (LLM minus clinician), not log-odds.
#
# UNCERTAINTY. Delta-method standard errors from the fixed-effect covariance of
# the GLMM (which accounts for the crossed vignette x clinician dependence);
# vignette conditional modes are held at their estimates. The two-stage
# bootstrap is the assumption-light check on this.
#
# MULTIPLICITY. One scheme throughout: single-step max-t (Hothorn, Bretz &
# Westfall 2008). Within each family, the simultaneous critical value and the
# adjusted p-values both come from the same multivariate normal distribution
# of the contrast z-statistics, so "adjusted p < 0.05" <=> "simultaneous CI
# excludes 0". For Family A (each LLM vs one common control) this is the
# Dunnett-type many-to-one adjustment.
#
# All contrasts are computed as linear combinations of the group-level
# benchmark accuracies from ONE model fit, with their joint delta-method
# covariance. This is numerically identical to avg_comparisons() on the same
# grid (difference of averages = average of differences) but lets every family
# (vs clinicians, pairwise, vs each profession, within domain) use one code path.
# =============================================================================

suppressPackageStartupMessages({
  library(readr)
  library(lme4)
  library(marginaleffects)
  library(mvtnorm)
  library(broom.mixed)
  library(DHARMa)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
})

set.seed(20260912)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE    <- "/Users/muellv01/Library/CloudStorage/OneDrive-NYULangoneHealth/Projects/ICD11_WHO/MixedEffects"
INPUT   <- file.path(BASE, "glmm_input_clinician_llm_long.csv")
INPUT_BY_PROFESSION <- file.path(BASE, "glmm_input_clinician_by_profession_llm_long.csv")
OUTDIR  <- file.path(BASE, "glmm_outputs_v2")
dir.create(OUTDIR, showWarnings = FALSE, recursive = TRUE)

MARGIN  <- 0.10     # accuracy-scale NI / equivalence margin (delta)
CONF    <- 0.95     # two-sided confidence level
N_BOOT  <- 2000     # cluster bootstrap replicates

# NULL = clinicians in all languages (primary). c("english") = English-only.
CLINICIAN_LANGUAGES <- NULL
# Bootstrap uses the same clinicians as the primary model (all languages).
BOOT_LANGUAGES      <- NULL

RUN_PAIRWISE              <- TRUE   # Family B; report only if ranking claims are made
RUN_DOMAIN_INTERACTION    <- TRUE
RUN_DOMAIN_CONTRASTS      <- TRUE   # per-domain contrasts (supplement only)
RUN_PROFESSION_STRATIFIED <- TRUE
RUN_DIAGNOSTICS           <- TRUE

CLIN_GROUPS <- c("clinician_Medicine", "clinician_Psychology", "clinician_Other")

# Numerical settings for the multivariate normal integrals (Genz-Bretz is
# Monte Carlo; tight tolerances make crit values / p-values reproducible to
# ~3 decimals).
MVN_ALGO <- mvtnorm::GenzBretz(maxpts = 5e5, abseps = 1e-5, releps = 0)

log_msg <- function(...) cat(sprintf(...), "\n")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
to01 <- function(x) {
  if (is.logical(x))   return(as.integer(x))
  if (is.character(x)) return(as.integer(tolower(trimws(x)) %in% c("1", "true", "yes")))
  as.integer(x)
}

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

check_fit <- function(model, label) {
  msgs <- model@optinfo$conv$lme4$messages
  if (is.null(msgs)) log_msg("%s converged cleanly.", label)
  else               log_msg("%s CONVERGENCE NOTE(S): %s", label, paste(msgs, collapse = " | "))
  if (isSingular(model)) log_msg("%s is SINGULAR (a variance component ~ 0).", label)
}

# Grid: every rater group x the 43 vignettes (with their own domain).
make_grid <- function(d, groups) {
  vd <- d %>% distinct(vignette_id, domain)
  tidyr::crossing(rater_group = groups, vd) %>%
    mutate(rater_group = factor(rater_group, levels = levels(d$rater_group)),
           vignette_id = factor(vignette_id, levels = levels(d$vignette_id)),
           domain      = factor(domain,      levels = levels(d$domain)))
}

# Benchmark accuracy per group (optionally per group x domain) and the joint
# delta-method covariance of those accuracies.
bench_acc <- function(model, grid, by = "rater_group") {
  pr <- avg_predictions(model, by = by, newdata = grid,
                        re.form = ~ (1 | vignette_id), type = "response")
  df <- as.data.frame(pr)
  key <- if (length(by) == 1) as.character(df[[by]])
         else do.call(paste, c(lapply(by, function(b) as.character(df[[b]])), sep = "|"))
  V <- vcov(pr)
  dimnames(V) <- list(key, key)
  list(est = setNames(df$estimate, key), V = V, table = df)
}

# Contrast matrix. `pairs` is a two-column character matrix (minuend, subtrahend).
contrast_matrix <- function(keys, pairs, labels = paste(pairs[, 1], "-", pairs[, 2])) {
  L <- matrix(0, nrow(pairs), length(keys), dimnames = list(labels, keys))
  for (i in seq_len(nrow(pairs))) { L[i, pairs[i, 1]] <- 1; L[i, pairs[i, 2]] <- -1 }
  L
}

# Single-step max-t for one family: simultaneous CI + adjusted p-values.
# Works for rank-deficient correlation matrices (e.g. all-pairwise families).
maxt_family <- function(est, V, conf = CONF) {
  k  <- length(est)
  se <- sqrt(diag(V))
  z  <- est / se
  if (k == 1) {
    crit  <- qnorm(1 - (1 - conf) / 2)
    p_adj <- 2 * pnorm(-abs(z))
  } else {
    R <- cov2cor(V)
    crit <- mvtnorm::qmvnorm(conf, tail = "both.tails", corr = R,
                             algorithm = MVN_ALGO)$quantile
    p_adj <- vapply(abs(z), function(t)
      1 - mvtnorm::pmvnorm(lower = rep(-t, k), upper = rep(t, k),
                           corr = R, algorithm = MVN_ALGO)[1],
      numeric(1))
    p_adj <- pmin(1, pmax(p_adj, 0))
  }
  data.frame(estimate = est, std.error = se, z = z,
             ci_low_pt  = est - qnorm(1 - (1 - conf) / 2) * se,   # pointwise (auxiliary)
             ci_high_pt = est + qnorm(1 - (1 - conf) / 2) * se,
             ci_low     = est - crit * se,                        # simultaneous (governing)
             ci_high    = est + crit * se,
             p_unadj    = 2 * pnorm(-abs(z)),
             p_maxt     = p_adj,
             crit_maxt  = crit,
             k_family   = k)
}

# Apply L to a bench_acc() result and run max-t on the resulting family.
run_family <- function(ba, L, family) {
  est <- as.numeric(L %*% ba$est[colnames(L)])
  V   <- L %*% ba$V[colnames(L), colnames(L)] %*% t(L)
  out <- maxt_family(est, V)
  cbind(family = family, contrast = rownames(L), out, row.names = NULL)
}

# Superiority / NI / equivalence verdicts from the simultaneous CI.
classify <- function(tab, margin = MARGIN) {
  tab %>% mutate(
    superior     = ci_low  > 0,
    inferior     = ci_high < 0,
    non_inferior = ci_low  > -margin,
    equivalent   = ci_low  > -margin & ci_high < margin
  )
}

fmt_p <- function(p) ifelse(p < 0.001, "<0.001", sprintf("%.3f", p))

prep <- function(d) {
  d$correct     <- to01(d$correct)
  d$rater_id    <- factor(d$rater_id)
  d$vignette_id <- factor(d$vignette_id)
  d$domain      <- factor(d$domain)
  d
}

# ---------------------------------------------------------------------------
# Load and prepare (pooled clinicians)
# ---------------------------------------------------------------------------
dat <- read_csv(INPUT, show_col_types = FALSE)
if (!"rater_group" %in% names(dat)) {
  dat$rater_group <- ifelse(dat$rater_type == "clinician", "clinician", dat$rater_id)
}
if (!is.null(CLINICIAN_LANGUAGES)) {
  dat <- dat %>% filter(rater_group != "clinician" | language %in% CLINICIAN_LANGUAGES)
}
dat <- prep(dat)
dat$rater_group  <- relevel(factor(dat$rater_group), ref = "clinician")
dat$is_clinician <- as.integer(dat$rater_group == "clinician")
if (sum(dat$is_clinician) == 0) stop("No clinician rows (rater_group == 'clinician').")

llm_levels <- setdiff(levels(dat$rater_group), "clinician")

sample_desc <- data.frame(
  n_ratings            = nrow(dat),
  n_clinician_ratings  = sum(dat$is_clinician),
  n_clinicians         = n_distinct(dat$rater_id[dat$is_clinician == 1]),
  n_llms               = length(llm_levels),
  n_vignettes          = n_distinct(dat$vignette_id),
  clinician_languages  = paste(sort(unique(dat$language[dat$is_clinician == 1])), collapse = ";"),
  domains              = paste(levels(dat$domain), collapse = ";")
)
write_csv(sample_desc, file.path(OUTDIR, "sample_description.csv"))
print(sample_desc)

# ---------------------------------------------------------------------------
# (1) Primary GLMM
# ---------------------------------------------------------------------------
form <- correct ~ rater_group + domain + (1 | vignette_id) + (0 + is_clinician | rater_id)
m <- glmer(form, data = dat, family = binomial, control = ctrl)
check_fit(m, "Primary model")
print(summary(m))

vc <- as.data.frame(VarCorr(m))
write_csv(broom.mixed::tidy(m), file.path(OUTDIR, "primary_model_tidy.csv"))
write_csv(vc, file.path(OUTDIR, "variance_components.csv"))

grid_full <- make_grid(dat, levels(dat$rater_group))
ba <- bench_acc(m, grid_full)

# Benchmark accuracy per group (model-based; for tables / figures).
acc_tab <- ba$table %>%
  transmute(rater_group, accuracy = estimate, ci_low = conf.low, ci_high = conf.high)
write_csv(acc_tab, file.path(OUTDIR, "group_accuracies_benchmark.csv"))

# Family A: each LLM minus clinicians.
L_A <- contrast_matrix(names(ba$est), cbind(llm_levels, "clinician"))
fam_A <- classify(run_family(ba, L_A, "A: LLM vs clinicians"))

# Sanity check: must match avg_comparisons() on the clinician grid.
chk <- as.data.frame(avg_comparisons(
  m, variables = "rater_group",
  newdata = make_grid(dat, "clinician"),
  re.form = ~ (1 | vignette_id), type = "response"))
chk_diff <- max(abs(sort(chk$estimate) - sort(fam_A$estimate)))
log_msg("Check vs avg_comparisons: max |diff| in estimates = %.2e", chk_diff)
if (chk_diff > 1e-6) warning("Contrast estimates do not match avg_comparisons().")

write_csv(fam_A, file.path(OUTDIR, "claims_llm_vs_clinician.csv"))
log_msg("\nFamily A (LLM - clinician), max-t crit = %.3f:", fam_A$crit_maxt[1])
print(fam_A %>% mutate(p_maxt = fmt_p(p_maxt)) %>%
        select(contrast, estimate, ci_low, ci_high, p_maxt,
               superior, non_inferior, equivalent))

# Family B: all pairwise among rater groups (LLM ranking).
if (RUN_PAIRWISE) {
  g <- levels(dat$rater_group)
  prs <- t(combn(g, 2))
  L_B <- contrast_matrix(names(ba$est), prs)
  fam_B <- run_family(ba, L_B, "B: all pairwise")
  write_csv(fam_B, file.path(OUTDIR, "pairwise_rater_groups.csv"))
  log_msg("Family B (all %d pairwise), max-t crit = %.3f", nrow(fam_B), fam_B$crit_maxt[1])
}

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
if (RUN_DIAGNOSTICS) {
  sim  <- DHARMa::simulateResiduals(fittedModel = m, n = 1000)
  simV <- DHARMa::recalculateResiduals(sim, group = dat$vignette_id)
  png(file.path(OUTDIR, "dharma_residuals.png"), width = 1100, height = 900)
  op <- par(mfrow = c(2, 1))
  plot(sim,  main = "DHARMa (per rating)")
  plot(simV, main = "DHARMa (aggregated by vignette)")
  par(op); dev.off()

  disp_test    <- DHARMa::testDispersion(sim, plot = FALSE)
  unif_test    <- DHARMa::testUniformity(sim, plot = FALSE)
  outlier_test <- DHARMa::testOutliers(sim, type = "bootstrap", plot = FALSE)
  write_csv(data.frame(
    test      = c("KS / uniformity", "dispersion", "outliers (bootstrap)"),
    statistic = c(unif_test$statistic, disp_test$statistic, outlier_test$statistic),
    p_value   = c(unif_test$p.value,   disp_test$p.value,   outlier_test$p.value)),
    file.path(OUTDIR, "dharma_tests.csv"))

  re_vig  <- ranef(m)$vignette_id[["(Intercept)"]]
  re_clin_df <- ranef(m)$rater_id
  re_clin <- re_clin_df[rownames(re_clin_df) %in%
                          unique(as.character(dat$rater_id[dat$is_clinician == 1])), 1]
  png(file.path(OUTDIR, "random_effects_qq.png"), width = 1100, height = 500)
  op <- par(mfrow = c(1, 2))
  qqnorm(re_vig,  main = sprintf("Vignette intercepts (n = %d)", length(re_vig)));  qqline(re_vig,  col = "red")
  qqnorm(re_clin, main = sprintf("Clinician intercepts (n = %d)", length(re_clin))); qqline(re_clin, col = "red")
  par(op); dev.off()
  sw <- function(x) if (length(x) >= 3 && length(x) <= 5000) shapiro.test(x) else list(statistic = NA, p.value = NA)
  write_csv(data.frame(
    effect    = c("vignette", "clinician"),
    n         = c(length(re_vig), length(re_clin)),
    sd_blup   = c(sd(re_vig), sd(re_clin)),
    shapiro_W = c(unname(sw(re_vig)$statistic), unname(sw(re_clin)$statistic)),
    shapiro_p = c(sw(re_vig)$p.value, sw(re_clin)$p.value)),
    file.path(OUTDIR, "random_effects_normality.csv"))
}

# ---------------------------------------------------------------------------
# (2) Domain moderation
# ---------------------------------------------------------------------------
if (RUN_DOMAIN_INTERACTION) {
  form_int <- correct ~ rater_group * domain + (1 | vignette_id) + (0 + is_clinician | rater_id)
  m_int <- glmer(form_int, data = dat, family = binomial, control = ctrl)
  check_fit(m_int, "Domain interaction model")
  lrt_dom <- anova(m, m_int)
  print(lrt_dom)
  write_csv(broom.mixed::tidy(lrt_dom), file.path(OUTDIR, "domain_interaction_lrt.csv"))

  if (RUN_DOMAIN_CONTRASTS) {
    # Within each domain: LLM - clinician over that domain's vignettes;
    # max-t within domain (10 contrasts per family).
    ba_d <- bench_acc(m_int, grid_full, by = c("rater_group", "domain"))
    fam_D <- bind_rows(lapply(levels(dat$domain), function(dm) {
      L <- contrast_matrix(names(ba_d$est),
                           cbind(paste(llm_levels, dm, sep = "|"), paste("clinician", dm, sep = "|")),
                           labels = paste(llm_levels, "- clinician"))
      cbind(domain = dm, classify(run_family(ba_d, L, paste("Domain:", dm))))
    }))
    write_csv(fam_D, file.path(OUTDIR, "claims_by_domain.csv"))
  }
}

# ---------------------------------------------------------------------------
# (3) Sensitivity S5a: two-stage cluster bootstrap with max-t band
# ---------------------------------------------------------------------------
# Resamples (i) the 43 vignettes and (ii) whole clinicians, with replacement,
# independently within each replicate. LLMs are deterministic (one response per
# vignette) and contribute no sampling term. Estimand: LLM minus clinician
# observed accuracy, each averaged with equal weight over vignettes (clinician
# per-vignette accuracy = mean over the ratings of that vignette). No
# distributional assumption on vignette or clinician effects.
bsub <- if (is.null(BOOT_LANGUAGES)) dat else
  dat %>% filter(rater_group != "clinician" | language %in% BOOT_LANGUAGES)

clin_long <- bsub %>% filter(rater_group == "clinician") %>%
  select(rater_id, vignette_id, correct) %>% as.data.frame()
clin_ids  <- unique(as.character(clin_long$rater_id))
n_clin    <- length(clin_ids)
clin_idx  <- match(as.character(clin_long$rater_id), clin_ids)
cc        <- clin_long$correct
vg        <- as.character(clin_long$vignette_id)
all_vigs  <- sort(unique(as.character(bsub$vignette_id)))

clin_acc_vig <- function(w_rows) {
  num <- rowsum(w_rows * cc, vg); den <- rowsum(w_rows, vg)
  a <- setNames(as.numeric(num) / as.numeric(den), rownames(num))
  a[all_vigs]                                   # NA where a vignette got no weight
}

llm_mat <- sapply(llm_levels, function(mod) {
  lr <- bsub %>% filter(rater_group == mod)
  tapply(lr$correct, as.character(lr$vignette_id), function(z) z[1])[all_vigs]
})                                              # vignettes x LLMs

clin_obs <- clin_acc_vig(rep(1, length(cc)))
est_obs  <- colMeans(llm_mat - clin_obs, na.rm = TRUE)
clin_obs_mean <- mean(clin_obs, na.rm = TRUE)

S      <- matrix(NA_real_, N_BOOT, length(llm_levels), dimnames = list(NULL, llm_levels))
S_clin <- numeric(N_BOOT)
n_dropped_vig <- integer(N_BOOT)
for (b in seq_len(N_BOOT)) {
  w  <- tabulate(sample.int(n_clin, n_clin, replace = TRUE), nbins = n_clin)
  ca <- clin_acc_vig(w[clin_idx])
  dv <- sample(all_vigs, length(all_vigs), replace = TRUE)
  diffs <- llm_mat[dv, , drop = FALSE] - ca[dv]
  S[b, ]    <- colMeans(diffs, na.rm = TRUE)
  S_clin[b] <- mean(ca[dv], na.rm = TRUE)
  n_dropped_vig[b] <- sum(is.na(ca[dv]))
}
if (any(n_dropped_vig > 0))
  log_msg("Bootstrap: %d replicates had >=1 vignette with no resampled clinician rating (dropped within replicate).",
          sum(n_dropped_vig > 0))

a       <- (1 - CONF) / 2
se_boot <- apply(S, 2, sd)
tmax    <- apply(S, 1, function(r) max(abs(r - est_obs) / se_boot))
crit_b  <- quantile(tmax, CONF, names = FALSE)
# Max-t adjusted p-value: share of replicates whose max centred statistic
# reaches the observed |t| of that contrast (resolution 1 / N_BOOT).
t_obs   <- abs(est_obs) / se_boot
p_boot  <- vapply(t_obs, function(t) (1 + sum(tmax >= t)) / (N_BOOT + 1), numeric(1))

boot_tab <- data.frame(
  contrast   = paste(llm_levels, "- clinician"),
  estimate   = est_obs,
  std.error  = se_boot,
  ci_low_pt  = apply(S, 2, quantile, probs = a,     names = FALSE),
  ci_high_pt = apply(S, 2, quantile, probs = 1 - a, names = FALSE),
  ci_low     = est_obs - crit_b * se_boot,
  ci_high    = est_obs + crit_b * se_boot,
  p_maxt     = p_boot,
  crit_maxt  = crit_b,
  row.names  = NULL
) %>% classify()
write_csv(boot_tab, file.path(OUTDIR, "supp_cluster_bootstrap.csv"))

# Descriptive clinician accuracy with a cluster-bootstrap CI (replaces the
# normal-approximation interval that ignores clustering).
write_csv(data.frame(
  clinician_accuracy_observed = clin_obs_mean,
  ci_low  = quantile(S_clin, a,     names = FALSE),
  ci_high = quantile(S_clin, 1 - a, names = FALSE),
  n_clinicians = n_clin, n_ratings = length(cc)),
  file.path(OUTDIR, "clinician_accuracy_bootstrap_ci.csv"))

log_msg("\nCluster bootstrap (all languages), max-t crit = %.3f:", crit_b)
print(boot_tab %>% select(contrast, estimate, ci_low, ci_high, p_maxt, superior, non_inferior))

# Agreement between GLMM and bootstrap verdicts.
agree <- fam_A %>% select(contrast, glmm_sup = superior, glmm_ni = non_inferior, glmm_inf = inferior) %>%
  inner_join(boot_tab %>% select(contrast, boot_sup = superior, boot_ni = non_inferior, boot_inf = inferior),
             by = "contrast") %>%
  mutate(same_verdict = glmm_sup == boot_sup & glmm_ni == boot_ni & glmm_inf == boot_inf)
write_csv(agree, file.path(OUTDIR, "glmm_vs_bootstrap_verdicts.csv"))
log_msg("GLMM and bootstrap verdicts agree for %d of %d LLMs.", sum(agree$same_verdict), nrow(agree))

# ---------------------------------------------------------------------------
# (4) Sensitivity S5b: clinician profession
# ---------------------------------------------------------------------------
# Same model, clinician level split into CLIN_GROUPS. Clinicians with missing
# profession are excluded upstream, so the pooled comparison model is refitted
# on the SAME subset for the LRT and for the "did any verdict change" table.
if (RUN_PROFESSION_STRATIFIED) {
  outdir_ps <- file.path(OUTDIR, "profession")
  dir.create(outdir_ps, showWarnings = FALSE, recursive = TRUE)

  dps <- read_csv(INPUT_BY_PROFESSION, show_col_types = FALSE)
  is_clin_row <- startsWith(as.character(dps$rater_group), "clinician")
  if (!is.null(CLINICIAN_LANGUAGES)) {
    dps <- dps[!is_clin_row | dps$language %in% CLINICIAN_LANGUAGES, ]
    is_clin_row <- startsWith(as.character(dps$rater_group), "clinician")
  }
  unexpected <- setdiff(unique(dps$rater_group[is_clin_row]), CLIN_GROUPS)
  if (length(unexpected) > 0) stop("Unexpected clinician groups: ", paste(unexpected, collapse = ", "))

  dps <- prep(dps)
  llm_ps <- sort(setdiff(unique(as.character(dps$rater_group)), CLIN_GROUPS))
  dps$rater_group  <- factor(dps$rater_group, levels = c(CLIN_GROUPS, llm_ps))
  dps$is_clinician <- as.integer(is_clin_row)

  # Pooled version of the same subset (for the LRT and verdict comparison).
  dpool <- dps
  dpool$rater_group <- factor(ifelse(dpool$is_clinician == 1, "clinician",
                                     as.character(dpool$rater_group)),
                              levels = c("clinician", llm_ps))

  counts <- dps %>% filter(is_clinician == 1) %>% group_by(rater_group) %>%
    summarise(n_clinicians = n_distinct(rater_id), n_ratings = n(),
              raw_accuracy = mean(correct), .groups = "drop")
  write_csv(counts, file.path(outdir_ps, "counts_by_profession.csv"))
  print(as.data.frame(counts))

  m_split <- glmer(form, data = dps,   family = binomial, control = ctrl)
  m_pool  <- glmer(form, data = dpool, family = binomial, control = ctrl)
  check_fit(m_split, "Profession-split model")
  check_fit(m_pool,  "Pooled model (profession subset)")
  write_csv(broom.mixed::tidy(m_split), file.path(outdir_ps, "model_tidy.csv"))

  # Do professions differ? (2 df: three clinician levels vs one)
  # (anova() refuses two different data objects, so the LRT is computed
  # directly; both models are ML fits on identical rows.)
  stopifnot(nobs(m_pool) == nobs(m_split))
  ll_p <- logLik(m_pool); ll_s <- logLik(m_split)
  lrt_prof <- data.frame(
    model  = c("pooled clinicians", "split by profession"),
    npar   = c(attr(ll_p, "df"), attr(ll_s, "df")),
    AIC    = c(AIC(m_pool), AIC(m_split)),
    logLik = c(as.numeric(ll_p), as.numeric(ll_s)),
    chisq  = c(NA, 2 * (as.numeric(ll_s) - as.numeric(ll_p))),
    df     = c(NA, attr(ll_s, "df") - attr(ll_p, "df")))
  lrt_prof$p_value <- c(NA, pchisq(lrt_prof$chisq[2], lrt_prof$df[2], lower.tail = FALSE))
  print(lrt_prof)
  write_csv(lrt_prof, file.path(outdir_ps, "profession_lrt.csv"))

  ba_ps <- bench_acc(m_split, make_grid(dps, levels(dps$rater_group)))
  write_csv(ba_ps$table, file.path(outdir_ps, "group_accuracies_benchmark.csv"))

  # Between-profession differences (descriptive; one family of 3).
  fam_cc <- run_family(ba_ps, contrast_matrix(names(ba_ps$est), t(combn(CLIN_GROUPS, 2))),
                       "Profession pairwise")
  write_csv(fam_cc, file.path(outdir_ps, "profession_pairwise.csv"))

  # Each LLM vs each profession; max-t within comparator (10 per family).
  fam_P <- bind_rows(lapply(CLIN_GROUPS, function(g) {
    L <- contrast_matrix(names(ba_ps$est), cbind(llm_ps, g))
    cbind(comparator = g, classify(run_family(ba_ps, L, paste("LLM vs", g))))
  }))
  write_csv(fam_P, file.path(outdir_ps, "claims_llm_vs_profession.csv"))

  # Pooled claims on the same subset, and which verdicts change by comparator.
  ba_pool <- bench_acc(m_pool, make_grid(dpool, levels(dpool$rater_group)))
  fam_pool <- classify(run_family(ba_pool,
                                  contrast_matrix(names(ba_pool$est), cbind(llm_ps, "clinician")),
                                  "LLM vs pooled (profession subset)"))
  fam_pool$llm <- llm_ps
  verdicts <- fam_P %>%
    mutate(llm = sub(" - .*$", "", contrast)) %>%
    select(comparator, llm, estimate, ci_low, ci_high, superior, non_inferior, inferior) %>%
    left_join(fam_pool %>% select(llm, pooled_superior = superior,
                                  pooled_non_inferior = non_inferior, pooled_inferior = inferior),
              by = "llm") %>%
    mutate(verdict_changed = superior != pooled_superior |
                             non_inferior != pooled_non_inferior |
                             inferior != pooled_inferior)
  write_csv(verdicts, file.path(outdir_ps, "verdicts_vs_pooled.csv"))
  log_msg("\nProfession: %d of %d LLM x comparator verdicts differ from the pooled model.",
          sum(verdicts$verdict_changed), nrow(verdicts))
  print(verdicts %>% filter(verdict_changed))

  fp_ps <- fam_P %>%
    mutate(label = sub(" - .*$", "", contrast),
           comparator = factor(sub("^clinician_", "vs. ", comparator),
                               levels = sub("^clinician_", "vs. ", CLIN_GROUPS)))
  ord <- fp_ps %>% filter(comparator == levels(comparator)[1]) %>% arrange(estimate) %>% pull(label)
  fp_ps$label <- factor(fp_ps$label, levels = ord)
  p_ps <- ggplot(fp_ps, aes(estimate, label)) +
    annotate("rect", xmin = -MARGIN, xmax = MARGIN, ymin = -Inf, ymax = Inf, alpha = 0.12) +
    geom_vline(xintercept = 0, linewidth = 0.4) +
    geom_vline(xintercept = c(-MARGIN, MARGIN), linetype = "dashed", linewidth = 0.3) +
    geom_pointrange(aes(xmin = ci_low, xmax = ci_high), size = 0.3) +
    facet_wrap(~ comparator, nrow = 1) +
    labs(x = "Accuracy difference (LLM minus clinician group)", y = NULL,
         subtitle = sprintf("Max-t simultaneous %.0f%% CIs within each comparator; band = +/- %.2f",
                            100 * CONF, MARGIN)) +
    theme_minimal(base_size = 11)
  ggsave(file.path(outdir_ps, "fig_forest_by_profession.png"), p_ps, width = 12, height = 5, dpi = 150)
}

# ---------------------------------------------------------------------------
# Figures (primary)
# ---------------------------------------------------------------------------
theme_set(theme_minimal(base_size = 12))

fp <- fam_A %>% mutate(label = sub(" - clinician$", "", contrast)) %>%
  arrange(estimate) %>% mutate(label = factor(label, levels = label))
p_forest <- ggplot(fp, aes(estimate, label)) +
  annotate("rect", xmin = -MARGIN, xmax = MARGIN, ymin = -Inf, ymax = Inf, alpha = 0.12) +
  geom_vline(xintercept = 0, linewidth = 0.4) +
  geom_vline(xintercept = c(-MARGIN, MARGIN), linetype = "dashed", linewidth = 0.3) +
  geom_pointrange(aes(xmin = ci_low, xmax = ci_high)) +
  labs(x = "Accuracy difference (LLM minus clinicians)", y = NULL,
       subtitle = sprintf("Max-t simultaneous %.0f%% CIs; band = +/- %.2f", 100 * CONF, MARGIN))
ggsave(file.path(OUTDIR, "fig_forest_llm_vs_clinician.png"), p_forest, width = 8, height = 5, dpi = 150)

ag <- acc_tab %>% arrange(accuracy) %>% mutate(rater_group = factor(rater_group, levels = rater_group))
p_acc <- ggplot(ag, aes(accuracy, rater_group)) +
  geom_pointrange(aes(xmin = ci_low, xmax = ci_high)) +
  labs(x = "Model-estimated accuracy (43 vignettes, typical rater)", y = NULL)
ggsave(file.path(OUTDIR, "fig_benchmark_accuracy.png"), p_acc, width = 8, height = 5, dpi = 150)

# ---------------------------------------------------------------------------
writeLines(capture.output(sessionInfo()), file.path(OUTDIR, "sessionInfo.txt"))
log_msg("\nDone. Outputs in %s/", OUTDIR)
