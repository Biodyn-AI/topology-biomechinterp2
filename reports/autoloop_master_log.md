# Autoloop Master Log (Subproject 40)

## Purpose
Running summary of executor + brainstormer iterations for topology/geometric hypothesis screening.

## Iterations

### iter_0001 (historical)
- Outcome: Blocked on missing directly loadable scGPT+Geneformer embedding artifacts within subproject_40 workspace scope.
- Artifact: `iterations/iter_0001/geometric_topology_test_summary.json`.

### iter_0003 (historical)
- Executed bounded screening with two hypothesis families:
  - `persistent_homology`: scGPT lung layer embeddings vs feature-shuffle null.
  - `cross_model_alignment`: scGPT vs Geneformer feature-effect vector alignment with exact permutation null.
- Key quantitative results:
  - H01 positive: top layer L0 mean H1 delta `18.603`, z `3.213`, Fisher p `0.0056`; `11/12` layers had Fisher p `< 0.05`.
  - H02 inconclusive: mean cosine alignment `0.825`, mean Spearman `0.833`, combined permutation p-values `0.349` (cosine) and `0.409` (Spearman).
- Iteration artifacts:
  - `iterations/iter_0003/scgpt_lung_h1_persistence_by_seed_layer.csv`
  - `iterations/iter_0003/scgpt_lung_h1_persistence_layer_summary.csv`
  - `iterations/iter_0003/cross_model_feature_alignment_by_domain.csv`
  - `iterations/iter_0003/cross_model_feature_alignment_summary.json`
  - `iterations/iter_0003/iter0003_screen_summary.json`
  - `iterations/iter_0003/executor_iteration_report.md`
  - `iterations/iter_0003/executor_next_steps.md`
  - `iterations/iter_0003/executor_hypothesis_screen.json`
- Decision:
  - Promote H01 to robustness/generalization checks across additional domains and disjoint split regimes.
  - Keep H02 as tentative until residual-level Geneformer embeddings are surfaced for stronger alignment testing.

### iter_0004 (historical)
- Executed bounded screening with two hypothesis families:
  - `persistent_homology`: cross-domain replication on scGPT immune and external-lung embeddings.
  - `intrinsic_dimensionality`: layer-wise coupling between H1 topology effect size and manifold proxies.
- Key quantitative results:
  - H03 positive: immune `12/12` and external-lung `12/12` layers with Fisher `p < 0.05`; mean layer H1 deltas `12.074` (immune) and `12.482` (external-lung); top-layer Fisher `p = 0.0056` in both domains.
  - H04 mixed/neutral: external-lung shows significant coupling (`participation_ratio` Fisher `p = 0.0229`, `linearity_top5` Fisher `p = 0.0178`), while immune correlations are not significant (`p >= 0.147`).
- Iteration artifacts:
  - `iterations/iter_0004/scgpt_cross_domain_h1_by_seed_layer.csv`
  - `iterations/iter_0004/scgpt_cross_domain_h1_layer_summary.csv`
  - `iterations/iter_0004/scgpt_cross_domain_h1_domain_summary.csv`
  - `iterations/iter_0004/intrinsic_dimensionality_h1_correlation_by_seed.csv`
  - `iterations/iter_0004/intrinsic_dimensionality_h1_correlation_summary.csv`
  - `iterations/iter_0004/iter0004_screen_summary.json`
  - `iterations/iter_0004/executor_iteration_report.md`
  - `iterations/iter_0004/executor_next_steps.md`
  - `iterations/iter_0004/executor_hypothesis_screen.json`
- Decision:
  - Promote H03 to stronger-null and split-robustness stress tests.
  - Keep H04 as neutral until replicated across domains with more stable intrinsic-dimension estimators and explicit biological anchoring.

### iter_0005 (historical)
- Executed bounded robustness packet with two linked hypothesis families:
  - `null_sensitivity`: persistent-homology H1 signal under feature-shuffle and distance-permutation nulls.
  - `split_robustness`: source-disjoint vs target-disjoint split reruns on top/weak layers per domain.
- Key quantitative results:
  - Feature-shuffle branch remains positive: `8/12` layer-tests significant (Fisher `p < 0.05`), mean layer delta `+3.998`.
  - Split-level feature-shuffle summary: source-disjoint `4/6` significant (mean delta `+5.219`), target-disjoint `4/6` significant (mean delta `+2.777`).
  - Stronger distance-permutation stress test is uniformly non-supportive: `0/12` significant, mean layer delta `-850.942`, negative deltas in `10/12` tests.
  - Dual-split robustness pass rate (feature-shuffle): `2/6` domain-layer combinations significant in both splits (lung `L0`, external-lung `L11`).
- Iteration artifacts:
  - `iterations/iter_0005/h1_stronger_null_split_by_seed_layer.csv`
  - `iterations/iter_0005/h1_stronger_null_split_layer_summary.csv`
  - `iterations/iter_0005/h1_stronger_null_split_domain_summary.csv`
  - `iterations/iter_0005/iter0005_screen_summary.json`
  - `iterations/iter_0005/executor_iteration_report.md`
  - `iterations/iter_0005/executor_next_steps.md`
  - `iterations/iter_0005/executor_hypothesis_screen.json`
- Decision:
  - Keep topology branch as provisionally positive under feature-shuffle but not yet promoted to fully robust due split sensitivity in `4/6` tested domain-layers.
  - Treat current distance-permutation null as potentially over-adversarial for biological interpretation; replace with graph rewiring-style stronger null next.

### iter_0006 (historical)
- Executed bounded immune full-layer screening with two linked hypothesis families:
  - `null_sensitivity`: replaced distance permutation with degree-preserving kNN rewiring + geodesic-distance null.
  - `split_robustness`: expanded from top/weak-only checks to all 12 immune layers under source/target disjoint splits.
- Key quantitative results:
  - Feature-shuffle branch remains positive but asymmetric by split: source-disjoint `12/12` significant (mean delta `+6.646`), target-disjoint `4/12` significant (mean delta `+0.875`), total `16/24` significant layer-tests.
  - Dual-split feature-shuffle robustness pass rate: `4/12` layers (`7, 9, 10, 11`) significant in both splits.
  - Degree-preserving geodesic rewiring null is uniformly non-supportive: `0/24` significant, mean layer deltas `-140.519` (source) and `-129.702` (target).
  - Connectivity diagnostics: adaptive kNN used high neighborhood sizes (effective `k` mean `29.903`, max `30`) and component-bridge fallback in `142/144` by-seed rows.
- Iteration artifacts:
  - `iterations/iter_0006/h1_immune_rewire_split_by_seed_layer.csv`
  - `iterations/iter_0006/h1_immune_rewire_split_layer_summary.csv`
  - `iterations/iter_0006/h1_immune_rewire_split_pass_matrix.csv`
  - `iterations/iter_0006/h1_immune_rewire_split_domain_summary.csv`
  - `iterations/iter_0006/h1_immune_rewire_dual_split_summary.csv`
  - `iterations/iter_0006/iter0006_screen_summary.json`
  - `iterations/iter_0006/executor_iteration_report.md`
  - `iterations/iter_0006/executor_next_steps.md`
  - `iterations/iter_0006/executor_hypothesis_screen.json`
- Decision:
  - Mark rewiring-null survival hypothesis as negative in this regime.
  - Keep split-robustness branch as neutral/partial (informative depth structure, but broad dual-split robustness not yet established).

### iter_0007 (historical)
- Executed bounded immune full-layer metric-matched calibration screening with two linked hypothesis families:
  - `null_sensitivity`: geodesic observed H1 vs degree-preserving rewired-geodesic null (and Euclidean comparator in parallel).
  - `manifold_distance`: calibration-shift diagnostic (`delta_geodesic - delta_euclidean`) plus geodesic-distortion lower-tail check.
- Key quantitative results:
  - Stronger rewiring branch remains uniformly non-supportive after metric matching: geodesic Fisher-significant tests `0/24` (minimum p `0.6913`), dual-split geodesic pass `0/12`.
  - Mean layer deltas remain strongly negative: geodesic `-95.356`, Euclidean `-95.536`.
  - Calibration shift is positive but small: mean `+0.180` across layer-split aggregates (`22/24` positive shift), insufficient to alter sign/significance outcomes.
  - Distortion control is non-supportive: `0/24` significant lower-tail distortion tests (minimum p `0.0696`), with mean observed-minus-null distortion delta `+0.105`.
  - Connectivity diagnostics in this run: bridge fallback `61/72` by-seed rows, mean kNN `k=29.181`.
- Iteration artifacts:
  - `iterations/iter_0007/h1_immune_metric_matched_by_seed_layer.csv`
  - `iterations/iter_0007/h1_immune_metric_matched_layer_summary.csv`
  - `iterations/iter_0007/h1_immune_metric_matched_pass_matrix.csv`
  - `iterations/iter_0007/h1_immune_metric_matched_domain_summary.csv`
  - `iterations/iter_0007/h1_immune_metric_calibration_shift_summary.csv`
  - `iterations/iter_0007/iter0007_screen_summary.json`
  - `iterations/iter_0007/executor_iteration_report.md`
  - `iterations/iter_0007/executor_next_steps.md`
  - `iterations/iter_0007/executor_hypothesis_screen.json`
- Decision:
  - Keep rewiring-based null-survival hypothesis negative in immune even after metric-matched calibration.
  - Mark mismatch-dominance hypothesis as inconclusive/neutral (directional shift exists but effect is too small to change outcomes).

### iter_0008 (current)
- Executed bounded immune full-layer connectivity-aware constrained-null screening with two linked hypothesis families:
  - `graph_topology`: bridge-conditioned/k-stratified diagnostics for rewiring negativity (`N35`).
  - `null_sensitivity`: quantile-constrained edge-length-aware rewiring vs unconstrained rewiring (`N36`).
- Key quantitative results:
  - Rewiring survival remained uniformly non-supportive under both null families: H1 Fisher-significant tests `0/24` for unconstrained and `0/24` for quantile-constrained; dual-split H1 pass `0/12` for both.
  - Mean H1 deltas stayed negative and did not improve with constrained rewiring: unconstrained `-19.244`, quantile-constrained `-19.532`.
  - Distortion lower-tail branch remained non-significant for both families (`0/24`, minimum Fisher p `0.0964`).
  - Quantile constraint produced only marginal edge-length-histogram shift in source (L1 `0.3249 -> 0.3129`) and essentially none in target (`0.1461 -> 0.1466`).
  - Bridge diagnostics were split-confounded in this run: source `36/36` bridged vs target `2/36` bridged, limiting causal interpretation of bridge-conditioned gaps.
- Iteration artifacts:
  - `iterations/iter_0008/h1_immune_constrained_rewire_by_seed_layer.csv`
  - `iterations/iter_0008/h1_immune_constrained_rewire_layer_summary.csv`
  - `iterations/iter_0008/h1_immune_constrained_rewire_pass_matrix.csv`
  - `iterations/iter_0008/h1_immune_constrained_rewire_domain_summary.csv`
  - `iterations/iter_0008/h1_immune_constrained_rewire_bridge_k_strata_summary.csv`
  - `iterations/iter_0008/h1_immune_constrained_rewire_bridge_gap_summary.csv`
  - `iterations/iter_0008/h1_immune_constrained_rewire_paired_shift_by_seed_layer.csv`
  - `iterations/iter_0008/h1_immune_constrained_rewire_paired_shift_summary.csv`
  - `iterations/iter_0008/iter0008_screen_summary.json`
  - `iterations/iter_0008/executor_iteration_report.md`
  - `iterations/iter_0008/executor_next_steps.md`
  - `iterations/iter_0008/executor_hypothesis_screen.json`
- Decision:
  - Keep rewiring-survival branch negative in immune under this constrained-null variant.
  - Mark bridge-conditioned explanation as inconclusive/partial due split-confounded strata; rerun with bridge-identifiable k schedules before final closure on that sub-claim.

### iter_0010 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `manifold_distance` (new method): geodesic vs Euclidean edge-distance AUROC on immune residual manifolds.
  - `topology_stability` (new family): bootstrap persistence + filtration sensitivity under feature-shuffle null.
  - `cross_model_alignment` (new method): disagreement-bin trend vs positive-edge rate across domains.
- Key quantitative results:
  - H13 promising: geodesic-minus-euclidean AUROC delta was positive in both splits (source `+0.00519`, target `+0.01319`), with `12/12` dual-split positive layers and `7/12` dual-split significant layers.
  - H14 promising: mean layer H1 delta `+3.870`; `12/12` layers positive and `12/12` layers with combined Fisher `p < 0.05`; filtration packet all-settings-positive fraction `1.0`.
  - H15 mixed/neutral: disagreement-vs-positive-rate Spearman rho was negative in lung (`-0.9758`, `p=3.33e-4`) and external-lung (`-0.5030`, `p=0.141`), but positive in immune (`+0.4012`, `p=0.250`); combined Fisher p-values were significant (`8.99e-4` two-sided), indicating domain-heterogeneous structure.
- Iteration artifacts:
  - `iterations/iter_0010/h13_manifold_distance_by_seed_layer_split.csv`
  - `iterations/iter_0010/h13_manifold_distance_layer_summary.csv`
  - `iterations/iter_0010/h13_manifold_distance_split_summary.csv`
  - `iterations/iter_0010/h13_manifold_distance_pass_matrix.csv`
  - `iterations/iter_0010/h14_topology_stability_bootstrap_records.csv`
  - `iterations/iter_0010/h14_topology_stability_seed_layer_setting_summary.csv`
  - `iterations/iter_0010/h14_topology_stability_layer_summary.csv`
  - `iterations/iter_0010/h14_topology_stability_filtration_sensitivity.csv`
  - `iterations/iter_0010/h14_topology_stability_filtration_layer_summary.csv`
  - `iterations/iter_0010/h15_cross_model_disagreement_trend.csv`
  - `iterations/iter_0010/h15_cross_model_disagreement_summary.json`
  - `iterations/iter_0010/iter0010_screen_summary.json`
  - `iterations/iter_0010/executor_iteration_report.md`
  - `iterations/iter_0010/executor_next_steps.md`
  - `iterations/iter_0010/executor_hypothesis_screen.json`
- Decision:
  - Promote manifold-distance and topology-stability branches.
  - Keep cross-model disagreement branch as domain-conditional/neutral pending per-edge controlled follow-up.

### iter_0011 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `module_structure` (new family): kNN community topology anchoring against regulatory labels in immune scGPT.
  - `cross_model_alignment` (rescue with new method): exact-permutation feature-transfer ranking tests across scGPT and Geneformer.
  - `intrinsic_dimensionality` (new method): split-wise coupling between intrinsic metrics and prior geodesic AUROC gain.
- Key quantitative results:
  - H16 promising: same-community edge indicator showed consistent enrichment for positives (source mean AUC `0.5387`, target `0.5413`), with `12/12` layers above 0.5 and Fisher-significant in both splits.
  - H17 promising (tentative): cross-model shared-feature ranking transfer was positive in all domains (mean Spearman rho `0.8333`, top-feature match `3/3`), with exact global null significance (`p=0.0369` for mean rho, `p=0.0415` for top-match count).
  - H18 mixed/neutral: intrinsic coupling with geodesic gain was significant only in target-disjoint (`local_linearity rho=+0.4079, p=0.0143`; `participation_ratio rho=-0.4079, p=0.0190`) and non-significant in source-disjoint (|rho|=`0.2354`, p~`0.17`).
- Iteration artifacts:
  - `iterations/iter_0011/h16_module_structure_by_seed_layer_split.csv`
  - `iterations/iter_0011/h16_module_structure_layer_summary.csv`
  - `iterations/iter_0011/h16_module_structure_split_summary.csv`
  - `iterations/iter_0011/h17_cross_model_transfer_domain_summary.csv`
  - `iterations/iter_0011/h17_cross_model_transfer_global_null.csv`
  - `iterations/iter_0011/h17_cross_model_transfer_summary.json`
  - `iterations/iter_0011/h18_intrinsic_geodesic_metrics_by_seed_layer_split.csv`
  - `iterations/iter_0011/h18_intrinsic_geodesic_coupling_by_seed.csv`
  - `iterations/iter_0011/h18_intrinsic_geodesic_coupling_summary.csv`
  - `iterations/iter_0011/iter0011_screen_summary.json`
  - `iterations/iter_0011/executor_iteration_report.md`
  - `iterations/iter_0011/executor_next_steps.md`
  - `iterations/iter_0011/executor_hypothesis_screen.json`
- Decision:
  - Promote `module_structure` and the revised cross-model transfer branch to follow-up.
  - Keep intrinsic-dimensionality coupling as split-conditional until cross-domain replication is complete.

### iter_0012 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `module_structure` (new method): confidence-stratified community enrichment test on the H16 branch.
  - `cross_model_alignment` (new method): matched-gene Procrustes/OT transfer with random-map nulls.
  - `intrinsic_dimensionality` (new method): local reconstruction-error mechanistic screen with permutation controls.
- Key quantitative results:
  - H19 negative: confidence-tier monotonicity failed directionally in both splits (mean AUROC slope source `-0.0771`, target `-0.0627`; positive-slope layers `0/12` in each split).
  - H20 promising/mixed: map-aware transfer was reproducibly positive across all domains (mean transferred-edge AUROC `0.5650`, `3/3` domains significant vs random-map null; mean Procrustes top-1 retrieval `0.3954`, `3/3` significant), but unsupervised OT recovery failed (`0/3` significant, mean top-1 `0.0024`).
  - H21 mixed: reconstruction-error edge signal was split-conditional (source mean AUROC `0.5331`, target `0.4780`) and coupling to geodesic lift was inverse in target (mean rho `-0.4079`, two-sided `p=0.0190`), contradicting the hypothesized positive trend.
- Iteration artifacts:
  - `iterations/iter_0012/h19_confidence_community_by_seed_layer_split_bin.csv`
  - `iterations/iter_0012/h19_confidence_community_layer_split_summary.csv`
  - `iterations/iter_0012/h19_confidence_community_monotonicity_tests.csv`
  - `iterations/iter_0012/h20_cross_model_transfer_by_domain_layer.csv`
  - `iterations/iter_0012/h20_cross_model_transfer_null_summary.csv`
  - `iterations/iter_0012/h20_cross_model_transfer_alignment_summary.csv`
  - `iterations/iter_0012/h21_local_reconstruction_edge_features.csv`
  - `iterations/iter_0012/h21_local_reconstruction_trend_summary.csv`
  - `iterations/iter_0012/h21_local_reconstruction_coupling_by_seed.csv`
  - `iterations/iter_0012/iter0012_screen_summary.json`
  - `iterations/iter_0012/executor_iteration_report.md`
  - `iterations/iter_0012/executor_next_steps.md`
  - `iterations/iter_0012/executor_hypothesis_screen.json`
- Decision:
  - Deprioritize/retire the current confidence-monotonicity variant (`H19`) unless rescued with materially changed confidence anchors.
  - Promote map-aware cross-model transfer (`H20` Procrustes branch) and keep OT as a distinct rescue target.
  - Keep local reconstruction branch (`H21`) as neutral with a reframed inverse-coupling follow-up.

### iter_0013 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `intrinsic_dimensionality` (refinement): cross-domain split x depth-phase interaction for local reconstruction AUROC (`H22`).
  - `graph_topology` (new method): Forman-curvature enrichment test on kNN graph edges (`H23`).
  - `cross_model_alignment` (new method): CCA-like scGPT/Geneformer manifold consistency with correspondence-permutation nulls (`H24`).
- Key quantitative results:
  - H22 neutral/mixed: only immune showed significant late negative target-minus-source effect (late diff `-0.0215`, 95% CI `[-0.0314, -0.0127]`, `p=7.5e-4`), while lung/external-lung did not; >=2-domain robustness gate was not met.
  - H23 negative: negative-curvature score underperformed in all domains (mean AUROC immune `0.3406`, lung `0.3894`, external-lung `0.3905`) with negative top-vs-bottom curvature-bin deltas across domains.
  - H24 promising: cross-model geometric consistency was strong in all 3 domains (mean canonical correlation `0.7968`, mean distance Spearman `0.7466`, mean kNN Jaccard `0.1714`, mean top-1 retrieval `0.7229`), each significant vs permutation null; combined Fisher p-values were `3.17e-05` for distance/Jaccard/top-1.
- Iteration artifacts:
  - `iterations/iter_0013/h22_phase_transition_by_seed_layer_split.csv`
  - `iterations/iter_0013/h22_phase_transition_phase_means.csv`
  - `iterations/iter_0013/h22_phase_transition_model_summary.csv`
  - `iterations/iter_0013/h22_phase_transition_null_summary.csv`
  - `iterations/iter_0013/h23_curvature_enrichment_by_seed_layer_split.csv`
  - `iterations/iter_0013/h23_curvature_enrichment_split_summary.csv`
  - `iterations/iter_0013/h23_curvature_enrichment_domain_summary.csv`
  - `iterations/iter_0013/h24_cross_model_cca_domain_summary.csv`
  - `iterations/iter_0013/h24_cross_model_cca_null_summary.csv`
  - `iterations/iter_0013/h24_cross_model_cca_overall_summary.csv`
  - `iterations/iter_0013/iter0013_screen_summary.json`
  - `iterations/iter_0013/executor_iteration_report.md`
  - `iterations/iter_0013/executor_next_steps.md`
  - `iterations/iter_0013/executor_hypothesis_screen.json`
- Decision:
  - Keep `H24` as the promoted branch for next-loop escalation with a correspondence-free rescue test.
  - Keep `H22` as neutral/domain-conditional pending biology-stratified replication.
  - Mark current `H23` curvature variant as negative and retire unless rescued with a materially changed curvature method.

### iter_0014 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `manifold_distance` (new method): diffusion-distance sweep vs Euclidean/geodesic baselines (`H25`).
  - `module_structure` (new method): geometry x multi-prior biological anchoring interactions with permutation calibration (`H26`).
  - `cross_model_alignment` (new method): unseeded Gromov-Wasserstein alignment with random-correspondence controls (`H27`).
- Key quantitative results:
  - H25 promising: mean best-diffusion AUROC exceeded best baseline by `+0.0173` overall (`72` best rows), with positive mean deltas in all domains (immune `+0.0161`, lung `+0.0249`, external-lung `+0.0109`) and domain-level Fisher significance in `3/3` domains.
  - H26 neutral/mixed: interaction term did not replicate (`0/6` rows with positive significant interaction), but full-model calibration gain was significant in `2/6` rows and combined Fisher for AUROC-delta p-values was `0.0140`.
  - H27 mixed/inconclusive: correspondence recovery failed (mean top-1 `0.00119`, combined Fisher `p=0.990`), while coarse geometric consistency was strong (mean distance Spearman `0.8556`, combined Fisher `p=2.33e-05`); edge-transfer AUROC was weak-borderline (mean `0.5186`, combined Fisher `p=0.0511`).
- Iteration artifacts:
  - `iterations/iter_0014/h25_diffusion_distance_by_seed_layer_split.csv`
  - `iterations/iter_0014/h25_diffusion_distance_domain_summary.csv`
  - `iterations/iter_0014/h25_diffusion_distance_null_summary.csv`
  - `iterations/iter_0014/h26_bio_anchor_edge_table.csv`
  - `iterations/iter_0014/h26_bio_anchor_model_summary.csv`
  - `iterations/iter_0014/h26_bio_anchor_permutation_null.csv`
  - `iterations/iter_0014/h27_gw_alignment_domain_summary.csv`
  - `iterations/iter_0014/h27_gw_alignment_null_summary.csv`
  - `iterations/iter_0014/h27_gw_alignment_map_quality.csv`
  - `iterations/iter_0014/iter0014_screen_summary.json`
  - `iterations/iter_0014/executor_iteration_report.md`
  - `iterations/iter_0014/executor_next_steps.md`
  - `iterations/iter_0014/executor_hypothesis_screen.json`
- Decision:
  - Promote `H25` (diffusion manifold metric branch).
  - Keep `H26` as neutral pending independent prior augmentation (STRING).
  - Keep `H27` as inconclusive/mixed; recoverable via seeded/one-to-one-constrained GW variants.

### iter_0015 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `manifold_distance` (refinement): diffusion uplift under coexpression+degree matched stronger null (`H28`).
  - `cross_model_alignment` (rescue/new method): CCA-seeded one-to-one GW correspondence recovery (`H29`).
  - `topology_stability` (new method): triangle-thinness/hyperbolicity edge screen (`H30`).
- Key quantitative results:
  - H28 inconclusive: mean diffusion uplift stayed positive (`+0.00774` AUROC; positive rows `44/72`), but matched-null support dropped (`3/72` rows with `p<0.05`; domain matched Fisher p-values immune `0.0749`, lung `0.1878`, external-lung `0.99999`).
  - H29 negative: seeded one-to-one GW did not recover usable correspondences (mean top-1 `0.00833`, combined Fisher `p=0.1248`) and did not improve transfer utility (mean transfer AUROC `0.5008`, combined Fisher `p=0.4345`, mean delta vs H27 `-0.0178`).
  - H30 negative: thinness score underperformed baseline geometry (mean AUROC `0.4657` vs geodesic `0.5508`; significant rows `1/24`, significant domain-split groups `1/6`).
- Iteration artifacts:
  - `iterations/iter_0015/h28_diffusion_coexp_by_seed_layer_split.csv`
  - `iterations/iter_0015/h28_diffusion_coexp_domain_summary.csv`
  - `iterations/iter_0015/h28_diffusion_coexp_null_summary.csv`
  - `iterations/iter_0015/h29_seeded_gw_domain_summary.csv`
  - `iterations/iter_0015/h29_seeded_gw_null_summary.csv`
  - `iterations/iter_0015/h29_seeded_gw_map_quality.csv`
  - `iterations/iter_0015/h30_hyperbolicity_by_seed_layer_split.csv`
  - `iterations/iter_0015/h30_hyperbolicity_domain_summary.csv`
  - `iterations/iter_0015/h30_hyperbolicity_null_summary.csv`
  - `iterations/iter_0015/iter0015_screen_summary.json`
  - `iterations/iter_0015/executor_iteration_report.md`
  - `iterations/iter_0015/executor_next_steps.md`
  - `iterations/iter_0015/executor_hypothesis_screen.json`
- Decision:
  - Keep diffusion branch active but non-promoted until coexpression-adjusted incremental-value evidence is demonstrated.
  - Retire GW-based correspondence recovery as a primary direction after two controlled failures (`H27`, `H29`).
  - Mark current triangle-thinness variant negative and avoid immediate polishing of this exact formulation.

### iter_0016 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `manifold_distance` (refinement): diffusion incremental value after explicit covariate adjustment and stratified diffusion-feature nulls (`H31`).
  - `graph_topology` (new method): convexity-deficit + detour-ratio geometric screen against geodesic baseline (`H32`).
  - `cross_model_alignment` (new method): tri-domain cycle-consistent non-GW alignment (`H33`).
- Key quantitative results:
  - H31 neutral/mixed: mean incremental gain remained positive (`+0.00346` AUROC overall; mean log-loss gain `+0.00263`), with Fisher-significant support in `3/6` domain-split groups and strongest signal in immune (source `p=6.78e-08`, target `p=3.84e-02`).
  - H32 promising: convexity/detour combo improved over geodesic baseline (mean combo AUROC `0.5682`, mean delta `+0.01706`), with Fisher-significant deltas in `4/6` domain-split groups (strongest lung-source `p=4.00e-04`).
  - H33 inconclusive/mixed: cycle consistency improved (`cycle-return 0.6385 -> 0.6654`, delta `+0.0269`, `p=0.0062` vs random maps), but edge-transfer AUROC did not improve (mean delta vs independent maps `-4.55e-05`, significant domains `0/3`).
- Iteration artifacts:
  - `iterations/iter_0016/h31_diffusion_incremental_by_seed_layer_split.csv`
  - `iterations/iter_0016/h31_diffusion_incremental_domain_summary.csv`
  - `iterations/iter_0016/h31_diffusion_incremental_null_summary.csv`
  - `iterations/iter_0016/h32_convexity_detour_by_seed_layer_split.csv`
  - `iterations/iter_0016/h32_convexity_detour_domain_summary.csv`
  - `iterations/iter_0016/h32_convexity_detour_null_summary.csv`
  - `iterations/iter_0016/h33_cycle_consistent_alignment_domain_summary.csv`
  - `iterations/iter_0016/h33_cycle_consistent_alignment_map_quality.csv`
  - `iterations/iter_0016/h33_cycle_consistent_alignment_null_summary.csv`
  - `iterations/iter_0016/iter0016_screen_summary.json`
  - `iterations/iter_0016/executor_iteration_report.md`
  - `iterations/iter_0016/executor_next_steps.md`
  - `iterations/iter_0016/executor_hypothesis_screen.json`
- Decision:
  - Promote `H32` as the primary new branch for multi-seed robustness expansion.
  - Keep `H31` active but non-promoted pending stronger cross-domain robustness under additional confound control.
  - Keep `H33` inconclusive; continue only with anchor-regularized variants tied to downstream edge-transfer gains.

### iter_0017 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `graph_topology` (refinement): multiseed convexity+detour incremental value over geodesic+diffusion covariates (`H34`).
  - `intrinsic_dimensionality` (new method): local-linearity depth-breakpoint split-asymmetry screen (`H35`).
  - `cross_model_alignment` (rescue with major method change): anchor-regularized utility-optimized spectral alignment (`H36`).
- Key quantitative results:
  - `H34` neutral-positive: mean incremental delta AUROC `+0.00153` across 72 rows, with positive mean deltas in `6/6` domain-split groups but Fisher-significant support in `2/6` (immune target `p=3.84e-05`, lung target `p=1.05e-04`).
  - `H35` mixed/neutral: piecewise depth structure was significant in all domains for both splits, but split-specific breakpoint shift was significant in only `1/3` domains (external-lung source-target shift `+4.33` layers, `p=0.0465`).
  - `H36` inconclusive/mixed: held-out target AUROC improved strongly over baseline (`0.7753` vs `0.5745`, mean delta `+0.2008`, positive in `3/3` domains) and was significant vs label-permutation null in `3/3` domains (`p=0.00826` each), but random-anchor null was non-discriminative (`0/3` significant), leaving anchor-specific attribution unresolved.
- Iteration artifacts:
  - `iterations/iter_0017/h34_convexity_detour_multiseed_by_seed_layer_split.csv`
  - `iterations/iter_0017/h34_convexity_detour_multiseed_domain_summary.csv`
  - `iterations/iter_0017/h34_convexity_detour_multiseed_null_summary.csv`
  - `iterations/iter_0017/h35_linearity_breakpoint_by_seed_domain_split.csv`
  - `iterations/iter_0017/h35_linearity_breakpoint_summary.csv`
  - `iterations/iter_0017/h35_linearity_breakpoint_null_summary.csv`
  - `iterations/iter_0017/h36_anchor_spectral_alignment_domain_summary.csv`
  - `iterations/iter_0017/h36_anchor_spectral_alignment_map_quality.csv`
  - `iterations/iter_0017/h36_anchor_spectral_alignment_null_summary.csv`
  - `iterations/iter_0017/iter0017_screen_summary.json`
  - `iterations/iter_0017/executor_iteration_report.md`
  - `iterations/iter_0017/executor_next_steps.md`
  - `iterations/iter_0017/executor_hypothesis_screen.json`
- Decision:
  - Keep `H34` active as a low-effect but directionally consistent refinement branch; prioritize biologically anchored stratifications before promotion.
  - Keep `H35` active for focused replication in external-lung where split breakpoint asymmetry appeared.
  - Keep `H36` as rescue-only and inconclusive until anchor-specific null controls become discriminative.

### iter_0018 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `graph_topology` (refinement): consensus-tier concentration test for convexity/detour uplift (`H37`).
  - `intrinsic_dimensionality` (new method): TWO-NN / participation-ratio variance-skew mechanism screen (`H38`).
  - `persistent_homology` (new family): H1 feature-shuffle excess + coupling to geometry uplift (`H39`).
- Key quantitative results:
  - `H37` negative: mean high-tier-minus-low-tier uplift gap `-0.00801` (finite in `12/24` rows), row-level significance `0/24`, domain-split Fisher significance `0/6`.
  - `H38` neutral/mixed: mean `ΔR²(full-mean)=+0.35673` across `18` seed-split fits (positive in `18/18`), but only `1/18` row-level significant and `0/6` domain-split Fisher-significant.
  - `H39` inconclusive: mean H1 z-score `+0.34579` across `24` rows and positive mean z in `5/6` domain-splits, but `0/24` row-level significant and `0/6` domain-split Fisher-significant; global Spearman(`H1 z`, geometry delta) `+0.3157`.
- Iteration artifacts:
  - `iterations/iter_0018/h37_consensus_tier_geometry_by_seed_layer_split.csv`
  - `iterations/iter_0018/h37_consensus_tier_geometry_domain_summary.csv`
  - `iterations/iter_0018/h37_consensus_tier_geometry_null_summary.csv`
  - `iterations/iter_0018/h38_id_distribution_moments_by_seed_layer_split.csv`
  - `iterations/iter_0018/h38_id_distribution_moments_fit_by_seed_split.csv`
  - `iterations/iter_0018/h38_id_distribution_moments_domain_summary.csv`
  - `iterations/iter_0018/h38_id_distribution_moments_null_summary.csv`
  - `iterations/iter_0018/h39_ph_feature_shuffle_by_seed_layer_split.csv`
  - `iterations/iter_0018/h39_ph_feature_shuffle_domain_summary.csv`
  - `iterations/iter_0018/h39_ph_feature_shuffle_null_summary.csv`
  - `iterations/iter_0018/iter0018_screen_summary.json`
  - `iterations/iter_0018/executor_iteration_report.md`
  - `iterations/iter_0018/executor_next_steps.md`
  - `iterations/iter_0018/executor_hypothesis_screen.json`
- Decision:
  - Retire/deprioritize the current H37 concentration variant as negative.
  - Keep H38 as neutral pending out-of-sample validation and stronger null robustness.
  - Keep H39 as inconclusive; continue only with stronger topology-stability controls.

### iter_0019 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `module_structure` (major rescue/new method): continuous biological-support interaction model for geometric utility (`H40`).
  - `topology_stability` (new method, fallback): split-zigzag persistence proxy with split-swap/layer-permutation controls (`H41`).
  - `intrinsic_dimensionality` (rescue/new method): out-of-sample validation of ID moments from H38 (`H42`).
- Key quantitative results:
  - `H40` promising: across 72 rows (3 domains x 3 seeds x 2 splits x 4 layers), mean interaction coefficient was `+0.1317`; domain-split Fisher significance for interaction held in `4/6` groups, and top-vs-bottom support-decile uplift gap was positive in `4/6`.
  - `H41` inconclusive/partial: split-zigzag proxy showed directional positive mean delta AUROC (`+0.0115`) with `5/6` domain-splits positive, but `0/6` domain-splits were Fisher-significant under layer-order permutation controls.
  - `H42` negative: out-of-sample ID mechanism validation was unstable and mostly non-supportive (overall mean observed `ΔR²=-10.70`; only `4/12` domain-split-evaluation rows had `p<0.05`).
- Iteration artifacts:
  - `iterations/iter_0019/h40_support_interaction_by_seed_layer_split.csv`
  - `iterations/iter_0019/h40_support_interaction_domain_summary.csv`
  - `iterations/iter_0019/h40_support_interaction_null_summary.csv`
  - `iterations/iter_0019/h41_zigzag_persistence_by_seed_layer_split.csv`
  - `iterations/iter_0019/h41_zigzag_persistence_domain_summary.csv`
  - `iterations/iter_0019/h41_zigzag_persistence_null_summary.csv`
  - `iterations/iter_0019/h42_id_oos_by_seed_split.csv`
  - `iterations/iter_0019/h42_id_oos_domain_summary.csv`
  - `iterations/iter_0019/h42_id_oos_null_summary.csv`
  - `iterations/iter_0019/iter0019_screen_summary.json`
  - `iterations/iter_0019/executor_iteration_report.md`
  - `iterations/iter_0019/executor_next_steps.md`
  - `iterations/iter_0019/executor_hypothesis_screen.json`
- Decision:
  - Promote `H40` as the highest-upside active branch, with follow-up focused on true STRING priors and stricter out-of-domain calibration.
  - Keep `H41` as inconclusive and method-partial until true zigzag persistence tooling is available.
  - Mark `H42` as negative for broad OOS mechanism claims; do not promote H38 without robust holdout performance.

### iter_0020 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `module_structure` (refinement with major method change): STRING + ontology support interaction (`H43`).
  - `topology_stability` (rescue with major method change): true split-zigzag persistence using `dionysus` (`H44`).
  - `intrinsic_dimensionality` (carry-over refinement): robust OOS ID/local-linearity validation (`H45`).
- Key quantitative results:
  - `H43` promising: mean interaction coefficient `+0.23253` across 24 rows; domain-split Fisher-significant interaction in `4/6` groups; all domain-splits directionally positive (`6/6`). Mean AUROC delta remained near zero (`-1.84e-05`).
  - `H44` promising: true zigzag H1 total lifetime exceeded target-set permutation nulls in `12/12` rows; mean observed-minus-null delta `+234.78`; Fisher-significant in `3/3` domains.
  - `H45` inconclusive/mixed: winsorized robust OOS delta was positive (mean `+10.8848`, permutation-significant `6/12`) but failed block sign-bootstrap (`0/12`); trimmed metric was unstable/negative (mean `-65.4611`, with non-finite leave-layer-out rows).
- Iteration artifacts:
  - `iterations/iter_0020/h43_support_interaction_ontology_by_seed_layer_split.csv`
  - `iterations/iter_0020/h43_support_interaction_ontology_domain_summary.csv`
  - `iterations/iter_0020/h43_support_interaction_ontology_null_summary.csv`
  - `iterations/iter_0020/h43_string_network_api.tsv`
  - `iterations/iter_0020/h44_true_zigzag_by_seed_layer_split.csv`
  - `iterations/iter_0020/h44_true_zigzag_domain_summary.csv`
  - `iterations/iter_0020/h44_true_zigzag_null_summary.csv`
  - `iterations/iter_0020/h45_id_oos_robust_by_seed_split.csv`
  - `iterations/iter_0020/h45_id_oos_robust_domain_summary.csv`
  - `iterations/iter_0020/h45_id_oos_robust_null_summary.csv`
  - `iterations/iter_0020/iter0020_screen_summary.json`
  - `iterations/iter_0020/executor_iteration_report.md`
  - `iterations/iter_0020/executor_next_steps.md`
  - `iterations/iter_0020/executor_hypothesis_screen.json`
- Decision:
  - Keep `H43` as the top biologically anchored active branch, but require held-out transfer gains before strong promotion.
  - Promote `H44` as a viable topology-stability lead now that true zigzag tooling is operational.
  - Keep `H45` non-promoted/inconclusive and avoid further spend without a materially changed robust OOS formulation.

### iter_0021 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `topology_stability` (refinement): support-weighted zigzag utility-coupling test (`H46`).
  - `persistent_homology` (new method): bifiltration-like cycle-rank score vs distance-only ablation (`H47`).
  - `cross_model_alignment` (rescue/new method): cross-model top-k motif-overlap enrichment (`H48`).
- Key quantitative results:
  - `H46` negative: weighted zigzag failed its utility-coupling objective (`weighted>unweighted` in `0/36` rows; domain weighted-better count `1/3`; leave-domain-out `R2` weighted `-0.5186` vs unweighted `-1.0049`).
  - `H47` promising: mean `ΔAUROC(bif - distance)=+0.00566`, positive in `24/24` rows, row-level `p<0.05` in `13/24`, and domain-split Fisher significance in `6/6`.
  - `H48` inconclusive/mixed: motif-overlap enrichment concentrated in immune only (`4/18` significant rows; domain Fisher-significant `1/3`; lung/external-lung null).
- Iteration artifacts:
  - `iterations/iter_0021/h46_weighted_zigzag_by_seed_layer_split.csv`
  - `iterations/iter_0021/h46_weighted_zigzag_domain_summary.csv`
  - `iterations/iter_0021/h46_weighted_zigzag_null_summary.csv`
  - `iterations/iter_0021/h47_bifiltration_by_domain_layer_split.csv`
  - `iterations/iter_0021/h47_bifiltration_domain_summary.csv`
  - `iterations/iter_0021/h47_bifiltration_null_summary.csv`
  - `iterations/iter_0021/h48_cross_model_motif_overlap_by_domain_layer.csv`
  - `iterations/iter_0021/h48_cross_model_motif_overlap_summary.csv`
  - `iterations/iter_0021/h48_cross_model_motif_overlap_null_summary.csv`
  - `iterations/iter_0021/iter0021_screen_summary.json`
  - `iterations/iter_0021/executor_iteration_report.md`
  - `iterations/iter_0021/executor_next_steps.md`
  - `iterations/iter_0021/executor_hypothesis_screen.json`
- Decision:
  - Promote `H47` to multi-seed robustness follow-up.
  - Retire/deprioritize the current weighted-zigzag formulation tested in `H46`.
  - Keep `H48` as a narrow immune-specific rescue lead only; do not generalize cross-domain yet.

### iter_0022 (current)
- Executed breadth-oriented screening with three hypotheses/families:
  - `persistent_homology` (refinement): multiseed bifiltration robustness + utility coupling screen (`H49`).
  - `topology_stability` (new method): directed/signed topology pilot with dual null controls (`H50`).
  - `cross_model_alignment` (rescue/new method): expanded anti-sparsity cross-model motif fingerprint (`H51`).
- Key quantitative results:
  - `H49` inconclusive/mixed: robust discrimination signal with mean `delta_AUROC=+0.00599` (positive `69/72`, `p<0.05` in `48/72`, Fisher-significant domain-splits `6/6`), but weak utility linkage (`1/3` domains positive coupling, layer-placebo significant `0/3`).
  - `H50` promising: directed/signed topology improved over distance-only baseline (mean `delta_AUROC=+0.01585`, positive `11/12`, domain-split Fisher-significant `6/6`).
  - `H51` neutral/mixed: degree-null motif enrichment broadened (Fisher-significant summary rows `5/6`, at least one significant variant in `3/3` domains), but module-shuffle control failed (`0/18` module rows significant).
- Iteration artifacts:
  - `iterations/iter_0022/h49_bifiltration_multiseed_by_seed_layer_split.csv`
  - `iterations/iter_0022/h49_bifiltration_multiseed_domain_summary.csv`
  - `iterations/iter_0022/h49_bifiltration_multiseed_null_summary.csv`
  - `iterations/iter_0022/h50_directed_signed_topology_by_domain_layer_split.csv`
  - `iterations/iter_0022/h50_directed_signed_topology_domain_summary.csv`
  - `iterations/iter_0022/h50_directed_signed_topology_null_summary.csv`
  - `iterations/iter_0022/h51_cross_model_motif_fingerprint_by_domain_layer_k.csv`
  - `iterations/iter_0022/h51_cross_model_motif_fingerprint_summary.csv`
  - `iterations/iter_0022/h51_cross_model_motif_fingerprint_null_summary.csv`
  - `iterations/iter_0022/iter0022_screen_summary.json`
  - `iterations/iter_0022/executor_iteration_report.md`
  - `iterations/iter_0022/executor_next_steps.md`
  - `iterations/iter_0022/executor_hypothesis_screen.json`
- Decision:
  - Promote `H50` to multiseed replication.
  - Keep `H49` active for discrimination, but do not promote utility-coupling claims without stronger placebo-calibrated support.
  - Retire the current `H51` motif-overlap formulation unless re-opened with a materially changed utility/topology-transfer objective.

## Progress Update (iter_0023)
- Implemented and executed `iterations/iter_0023/run_iter0023_screen.py` in `subproject40-topology`.
- Completed the planned 3-slot packet (`H52/H53/H54`) with machine-readable artifacts and explicit null controls.
- Generated machine artifacts:
  - `iterations/iter_0023/h52_directed_signed_multiseed_by_seed_layer_split.csv`
  - `iterations/iter_0023/h52_directed_signed_multiseed_domain_summary.csv`
  - `iterations/iter_0023/h52_directed_signed_multiseed_null_summary.csv`
  - `iterations/iter_0023/h53_directed_path_homology_by_domain_layer_split.csv`
  - `iterations/iter_0023/h53_directed_path_homology_domain_summary.csv`
  - `iterations/iter_0023/h53_directed_path_homology_null_summary.csv`
  - `iterations/iter_0023/h54_linearity_rupture_by_seed_layer_split.csv`
  - `iterations/iter_0023/h54_linearity_rupture_domain_summary.csv`
  - `iterations/iter_0023/h54_linearity_rupture_null_summary.csv`
  - `iterations/iter_0023/iter0023_screen_summary.json`
  - `iterations/iter_0023/iter0023_metric_digest.json`
- Wrote mandatory iteration files:
  - `iterations/iter_0023/executor_iteration_report.md`
  - `iterations/iter_0023/executor_next_steps.md`
  - `iterations/iter_0023/executor_hypothesis_screen.json`
- Updated cumulative assets:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added `ITERATION UPDATE: iter_0023`)
  - `paper/autoloop_research_paper.pdf` (compiled with `latexmk -pdf`)

## Decisions (iter_0023)
- `H52` is **promising**: mean `delta_AUROC=+0.01461`, positive rows `58/72`, Fisher-significant domain-splits `6/6`.
- `H53` is **inconclusive**: mean `delta_AUROC=+0.00276`, Fisher-significant domain-splits `0/6`.
- `H54` is **negative**: mean `delta_AUROC=-0.04527`, positive rows `20/72`, positive mean domain-splits `1/6`.

## Blockers (iter_0023)
- No hard blockers.
- Null-resolution caveat: low permutation budgets produce coarse row-level p-value floors; aggregate Fisher results remained informative for `H52`.

## Next Step (iter_0023)
- Promote `H52` with higher permutation budget and targeted diagnosis of the `lung/source_disjoint` failure slice.
- Do not expand `H53` without a materially changed objective.
- Retire current `H54` rupture-index formulation.

### iter_0024 (current)
- Executed breadth-oriented 3-slot screening with one carry-over refinement and two materially changed/new methods:
  - `H55` (`topology_stability`, refinement): directed/signed high-permutation replication + failure-slice diagnostics.
  - `H56` (`topology_stability`, major rescue): densified directed path-homology v2 with utility-transfer endpoint.
  - `H57` (`manifold_distance`, new method): geodesic anisotropy-tail broad screen.
- Key quantitative results:
  - `H55` promising: mean `delta_AUROC=+0.01169`, positive rows `25/36`, domain-split Fisher-significant `6/6`; two source-disjoint failure slices stayed negative (`lung=-0.00315`, `external_lung=-0.00440`). Diagnostics linked uplift to margin concentration (`margin_iqr` vs delta corr `+0.451`) with lower margin spread and weaker sign balance in failure slices.
  - `H56` negative for objective: discrimination improved directionally (mean `delta_AUROC=+0.00757`, positive `11/12`) but utility-transfer failed (`mean F1 lift=0.0000`, significant transfer rows `0/12`), so the rescue gate was not met.
  - `H57` negative for keep gate: mean `delta_AUROC=-0.01779`, positive rows `25/72`, positive mean domain-splits `3/6` only (best `external_lung/target=+0.03418`, worst `immune/source=-0.08534`).
- Iteration artifacts:
  - `iterations/iter_0024/h55_directed_signed_highperm_by_seed_layer_split.csv`
  - `iterations/iter_0024/h55_directed_signed_highperm_domain_summary.csv`
  - `iterations/iter_0024/h55_directed_signed_highperm_null_summary.csv`
  - `iterations/iter_0024/h55_directed_signed_failure_slice_diagnostics.csv`
  - `iterations/iter_0024/h56_path_homology_v2_by_domain_layer_split.csv`
  - `iterations/iter_0024/h56_path_homology_v2_utility_transfer_summary.csv`
  - `iterations/iter_0024/h56_path_homology_v2_null_summary.csv`
  - `iterations/iter_0024/h57_geodesic_anisotropy_by_seed_layer_split.csv`
  - `iterations/iter_0024/h57_geodesic_anisotropy_domain_summary.csv`
  - `iterations/iter_0024/h57_geodesic_anisotropy_null_summary.csv`
  - `iterations/iter_0024/iter0024_screen_summary.json`
  - `iterations/iter_0024/iter0024_metric_digest.json`
  - `iterations/iter_0024/executor_iteration_report.md`
  - `iterations/iter_0024/executor_next_steps.md`
  - `iterations/iter_0024/executor_hypothesis_screen.json`
- Decision:
  - Promote `H55` as the active lead with targeted failure-slice rescue.
  - Retire current `H56` utility-transfer path-homology formulation.
  - Do not promote current `H57` anisotropy-tail formulation.

### iter_0025 (current)
- Executed breadth-oriented 3-slot screening with one carry-over refinement and two materially changed/new methods:
  - `H58` (`topology_stability`, refinement): biologically weighted directed/signed failure-slice rescue.
  - `H59` (`cross_model_alignment`, major-change rescue): topology-signature transfer pilot across scGPT/Geneformer.
  - `H60` (`intrinsic_dimensionality`, new method): endpoint ID-jump multiseed broad-screen.
- Key quantitative results:
  - `H58` negative for rescue objective: mean `delta_AUROC(weighted-distance)=+0.01137` and positive rows `25/36`, but mean `delta_AUROC(weighted-unweighted)=-0.00052`; source-disjoint failures remained negative (`lung=-0.00193`, `external_lung=-0.00433`).
  - `H59` inconclusive: mean transfer delta `+0.02404` with positive rows `10/12`, but null support failed (`0/12` rows with `p_best<0.05`; domain Fisher-significant `0/3`).
  - `H60` negative: mean `delta_AUROC(combined_id_jump-geodesic)=-0.00435`, positive rows `31/72`, positive mean domain-splits `3/6`, Fisher-significant domain-splits `1/6` (negative-direction immune/source slice).
- Iteration artifacts:
  - `iterations/iter_0025/h58_weighted_directed_signed_by_seed_layer_split.csv`
  - `iterations/iter_0025/h58_weighted_directed_signed_domain_summary.csv`
  - `iterations/iter_0025/h58_weighted_directed_signed_null_summary.csv`
  - `iterations/iter_0025/h58_weighted_directed_signed_failure_slice_summary.csv`
  - `iterations/iter_0025/h59_cross_model_topology_signature_transfer_by_domain_layer.csv`
  - `iterations/iter_0025/h59_cross_model_topology_signature_transfer_summary.csv`
  - `iterations/iter_0025/h59_cross_model_topology_signature_transfer_null_summary.csv`
  - `iterations/iter_0025/h60_id_jump_by_seed_layer_split.csv`
  - `iterations/iter_0025/h60_id_jump_domain_summary.csv`
  - `iterations/iter_0025/h60_id_jump_null_summary.csv`
  - `iterations/iter_0025/iter0025_screen_summary.json`
  - `iterations/iter_0025/iter0025_metric_digest.json`
  - `iterations/iter_0025/executor_iteration_report.md`
  - `iterations/iter_0025/executor_next_steps.md`
  - `iterations/iter_0025/executor_hypothesis_screen.json`
- Decision:
  - Retire the current `H58` weighting-tweak rescue formulation.
  - Keep `H59` as inconclusive pilot only; do not expand without materially stronger transfer objective/null calibration.
  - Retire standalone `H60` endpoint ID-jump formulation.

### iter_0026 (current)
- Executed breadth-oriented 3-slot screening with one new family plus two materially changed rescues:
  - `H61` (`graph_topology`, new family): curvature-assortativity surrogate screen.
  - `H62` (`cross_model_alignment`, major-change rescue): biologically anchored contrastive alignment with null-gap objective.
  - `H63` (`intrinsic_dimensionality`, major-change rescue): layer-transition ID-gradient screen.
- Key quantitative results:
  - `H61` negative: mean `delta_AUROC(topology-distance)=-0.00719`, positive mean domain-splits `2/6`, Fisher-significant domain-splits `1/6` (`lung/target` only).
  - `H62` inconclusive: mean transfer delta `+0.04757`, but mean `null_gap_q95=-0.12923`; robust support concentrated in immune (`domain Fisher p=0.00116`) with non-robust null-gap in lung/external-lung.
  - `H63` negative: mean `delta_AUROC(transition-ID-geodesic)=-0.02061`, positive mean domain-splits `1/6`, and all transition aggregates remained negative (`0->3=-0.01843`, `3->7=-0.02018`, `7->11=-0.02324`).
- Iteration artifacts:
  - `iterations/iter_0026/h61_graph_curvature_by_seed_layer_split.csv`
  - `iterations/iter_0026/h61_graph_curvature_domain_summary.csv`
  - `iterations/iter_0026/h61_graph_curvature_null_summary.csv`
  - `iterations/iter_0026/h62_anchor_alignment_by_domain_layer_split.csv`
  - `iterations/iter_0026/h62_anchor_alignment_domain_summary.csv`
  - `iterations/iter_0026/h62_anchor_alignment_null_summary.csv`
  - `iterations/iter_0026/h63_transition_id_gradient_by_seed_transition_split.csv`
  - `iterations/iter_0026/h63_transition_id_gradient_domain_summary.csv`
  - `iterations/iter_0026/h63_transition_id_gradient_null_summary.csv`
  - `iterations/iter_0026/iter0026_screen_summary.json`
  - `iterations/iter_0026/executor_iteration_report.md`
  - `iterations/iter_0026/executor_next_steps.md`
  - `iterations/iter_0026/executor_hypothesis_screen.json`
- Decision:
  - Retire current undirected curvature-surrogate graph-topology variant (`H61`).
  - Keep `H62` as inconclusive; at most one immune-focused rescue is justified before retirement.
  - Retire transition-based intrinsic-dimensionality AUROC-lift formulation (`H63`).

### iter_0027 (current)
- Executed breadth-oriented 3-slot screening packet:
  - `persistent_homology` (`H64`): support-margin two-axis filtration surrogate vs directed baseline.
  - `cross_model_alignment` (`H65`): major-change cross-model topology codebook transport rescue.
  - `intrinsic_dimensionality` (`H66`): interaction-only ID features with directed support/margin.
- Key quantitative results:
  - H64 negative: mean `delta_AUROC(two-axis - baseline) = -0.03184`; source-disjoint failure slices remained negative (`lung=-0.02633`, `external_lung=-0.02928`); positive mean domain-splits `0/6`.
  - H65 mixed-to-negative: mean `null_gap_q95 = +0.13671` (immune mean `+0.16157`, non-immune positive null-gap `4/4`), but mean `delta_AUROC(transfer - baseline) = -0.10204` with positive mean domain-splits `0/6`.
  - H66 negative: mean `delta_AUROC(interaction - baseline) = -0.13176`; positive mean domain-splits `0/6`; Fisher-significant domain-splits `0/6`.
- Iteration artifacts:
  - `iterations/iter_0027/h64_support_margin_two_axis_by_seed_layer_split.csv`
  - `iterations/iter_0027/h64_support_margin_two_axis_domain_summary.csv`
  - `iterations/iter_0027/h64_support_margin_two_axis_null_summary.csv`
  - `iterations/iter_0027/h65_codebook_transport_by_domain_layer_split.csv`
  - `iterations/iter_0027/h65_codebook_transport_domain_summary.csv`
  - `iterations/iter_0027/h65_codebook_transport_null_summary.csv`
  - `iterations/iter_0027/h66_id_interaction_by_seed_transition_split.csv`
  - `iterations/iter_0027/h66_id_interaction_domain_summary.csv`
  - `iterations/iter_0027/h66_id_interaction_null_summary.csv`
  - `iterations/iter_0027/iter0027_screen_summary.json`
  - `iterations/iter_0027/executor_iteration_report.md`
  - `iterations/iter_0027/executor_next_steps.md`
  - `iterations/iter_0027/executor_hypothesis_screen.json`
- Decision:
  - Mark `H64`, `H65`, and `H66` as negative for promotion in this loop.
  - Retire this cycle’s AUROC-lift form of two-axis persistence, codebook-transport alignment utility endpoint, and ID-interaction endpoint.

### iter_0028 (current)
- Executed breadth-oriented 3-slot packet aligned to brainstormer picks `N329/N338/N335`:
  - `H67` (`persistent_homology`): rank-based multiparameter persistence surface (major-change rescue).
  - `H68` (`cross_model_alignment`): cycle-consistent utility-regularized mapping (major-change rescue).
  - `H69` (`manifold_distance`): multiscale geodesic triangle-defect spectrum (cheap broad-screen).
- Key quantitative results:
  - `H67` negative: mean `delta_AUROC(rank_surface-baseline)=-0.03048`; positive mean domain-splits `0/6`; source-disjoint failure slices remained negative (`lung=-0.03243`, `external_lung=-0.03453`).
  - `H68` negative: mean `delta_AUROC(transfer-baseline)=-0.30464`, mean `null_gap_q95=-0.03847`, mean mapped-to-sc cosine `+0.00560`; immune and non-immune keep gates both failed.
  - `H69` promising: mean `delta_AUROC(triangle_defect-baseline)=+0.02617`, positive rows `30/36`, positive mean domain-splits `6/6`; row-level significance limited by coarse null budget (`p_best` floor `0.111`).
- Iteration artifacts:
  - `iterations/iter_0028/h67_rank_surface_by_seed_layer_split.csv`
  - `iterations/iter_0028/h67_rank_surface_domain_summary.csv`
  - `iterations/iter_0028/h67_rank_surface_null_summary.csv`
  - `iterations/iter_0028/h68_cycle_utility_ot_by_domain_layer_split.csv`
  - `iterations/iter_0028/h68_cycle_utility_ot_domain_summary.csv`
  - `iterations/iter_0028/h68_cycle_utility_ot_null_summary.csv`
  - `iterations/iter_0028/h69_triangle_defect_by_seed_layer_split.csv`
  - `iterations/iter_0028/h69_triangle_defect_domain_summary.csv`
  - `iterations/iter_0028/h69_triangle_defect_null_summary.csv`
  - `iterations/iter_0028/iter0028_screen_summary.json`
  - `iterations/iter_0028/executor_iteration_report.md`
  - `iterations/iter_0028/executor_next_steps.md`
  - `iterations/iter_0028/executor_hypothesis_screen.json`
- Decision:
  - Retire current `H67` and `H68` utility endpoint forms for this loop.
  - Keep `H69` active for one robustness expansion with higher null resolution.

### iter_0029 (current)
- Executed breadth-oriented 3-slot packet aligned to brainstormer picks `N343/N350/N355`:
  - `H70` (`manifold_distance`, refinement): hard-null robustness expansion of `H69` with higher permutation budget.
  - `H71` (`cross_model_alignment`, major-change rescue): topology-signature distillation transfer pilot.
  - `H72` (`topology_stability`, new method): edge trajectory motif class screen.
- Key quantitative results:
  - `H70` promising: mean `delta_AUROC=+0.02637` (std `0.02687`), positive rows `29/36`, positive mean domain-splits `6/6`, and mean matched-random-third `null_gap_q95=+0.01010` with positive mean random-gap in `6/6` domain-splits.
  - `H71` negative: mean `delta_AUROC(transfer-baseline)=-0.42758`, positive rows `0/12`, mean `null_gap_q95=-0.14795`, and mean mapped-to-sc cosine `+0.00634`.
  - `H72` inconclusive: mean `delta_AUROC(motif-baseline)=+0.00008`, positive rows `4/6`, but `p_best<0.05` in `0/6` and enrichment `p<0.05` in `0/6`.
- Iteration artifacts:
  - `iterations/iter_0029/h70_triangle_defect_robust_by_seed_layer_split.csv`
  - `iterations/iter_0029/h70_triangle_defect_robust_domain_summary.csv`
  - `iterations/iter_0029/h70_triangle_defect_robust_null_summary.csv`
  - `iterations/iter_0029/h71_topology_signature_distill_by_domain_layer_split.csv`
  - `iterations/iter_0029/h71_topology_signature_distill_domain_summary.csv`
  - `iterations/iter_0029/h71_topology_signature_distill_null_summary.csv`
  - `iterations/iter_0029/h72_edge_trajectory_motif_by_domain_split.csv`
  - `iterations/iter_0029/h72_edge_trajectory_motif_domain_summary.csv`
  - `iterations/iter_0029/h72_edge_trajectory_motif_null_summary.csv`
  - `iterations/iter_0029/iter0029_screen_summary.json`
  - `iterations/iter_0029/executor_iteration_report.md`
  - `iterations/iter_0029/executor_next_steps.md`
  - `iterations/iter_0029/executor_hypothesis_screen.json`
- Decision:
  - Promote `H70` as the active lead for confirmatory stress tests with stronger biological controls.
  - Retire the current `H71` cross-model transfer endpoint after repeated utility-negative outcomes.
  - Keep `H72` as inconclusive pilot only; no major budget without stronger null/seed support.

### iter_0030 (current)
- Executed breadth-oriented 3-slot packet aligned to brainstormer picks `N368/N365/N361`:
  - `H73` (`module_structure`, refinement): support-concordance anchoring of the `H70` triangle-defect branch.
  - `H74` (`cross_model_alignment`, major-change rescue-once): relational spectral alignment pilot.
  - `H75` (`manifold_distance`, new method): geodesic curvature-acceleration broad screen.
- Key quantitative results:
  - `H73` neutral/mixed: mean `delta_AUROC(triangle-baseline)=+0.02498`, but support interaction failed (`mean interaction=-0.00032`, interaction-positive rows `19/36`, mean interaction `null_gap_q95=-0.08390`, null-surviving interaction mean domain-splits `0/6`).
  - `H74` negative: mean `delta_AUROC(transfer-baseline)=+0.01136` but mean `null_gap_q95=-0.09881`; immune gate failed (`immune mean delta=-0.16020`, `immune mean null_gap=-0.14682`), with positive null-gap in only `2/12` rows.
  - `H75` inconclusive: mean `delta_AUROC(curvature-baseline)=+0.00210`, positive rows `3/6`, `p_best<0.05` in `2/6`, but `null_gap_q95` remained negative in all domain-splits (`0/6` positive).
- Iteration artifacts:
  - `iterations/iter_0030/h73_support_concordance_by_seed_layer_split.csv`
  - `iterations/iter_0030/h73_support_concordance_domain_summary.csv`
  - `iterations/iter_0030/h73_support_concordance_null_summary.csv`
  - `iterations/iter_0030/h74_relational_spectral_alignment_by_domain_layer_split.csv`
  - `iterations/iter_0030/h74_relational_spectral_alignment_domain_summary.csv`
  - `iterations/iter_0030/h74_relational_spectral_alignment_null_summary.csv`
  - `iterations/iter_0030/h75_curvature_acceleration_by_domain_split.csv`
  - `iterations/iter_0030/h75_curvature_acceleration_domain_summary.csv`
  - `iterations/iter_0030/h75_curvature_acceleration_null_summary.csv`
  - `iterations/iter_0030/iter0030_screen_summary.json`
  - `iterations/iter_0030/executor_iteration_report.md`
  - `iterations/iter_0030/executor_next_steps.md`
  - `iterations/iter_0030/executor_hypothesis_screen.json`
- Decision:
  - Keep only the core `H70` geometry branch active; retire the tested `H73` support-interaction formulation.
  - Retire cross-model edge-utility transfer endpoint again after `H74` rescue failure.
  - Keep `H75` as a low-budget inconclusive probe pending materially changed curvature parameterization.

## Progress Update (iter_0031)
- Implemented and ran `iterations/iter_0031/run_iter0031_screen.py` in `subproject40-topology`.
- Completed a 3-slot breadth packet mapped to brainstormer priorities:
  - `H76` coexpression-aware support-concordance interaction v2 (`N382`),
  - `H77` cross-model relational rank agreement endpoint (`N379`),
  - `H78` geodesic detour-elasticity neighborhood-scale screen (`N376`).
- Generated machine artifacts:
  - `iterations/iter_0031/h76_coexpression_support_interaction_by_seed_layer_split.csv`
  - `iterations/iter_0031/h76_coexpression_support_interaction_domain_summary.csv`
  - `iterations/iter_0031/h76_coexpression_support_interaction_null_summary.csv`
  - `iterations/iter_0031/h77_relational_rank_agreement_by_domain_layer_split.csv`
  - `iterations/iter_0031/h77_relational_rank_agreement_domain_summary.csv`
  - `iterations/iter_0031/h77_relational_rank_agreement_null_summary.csv`
  - `iterations/iter_0031/h78_geodesic_detour_elasticity_by_domain_split_layer.csv`
  - `iterations/iter_0031/h78_geodesic_detour_elasticity_domain_summary.csv`
  - `iterations/iter_0031/h78_geodesic_detour_elasticity_null_summary.csv`
  - `iterations/iter_0031/iter0031_screen_summary.json`
- Wrote required iteration outputs:
  - `iterations/iter_0031/executor_iteration_report.md`
  - `iterations/iter_0031/executor_next_steps.md`
  - `iterations/iter_0031/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0031`)
  - `paper/autoloop_research_paper.pdf` (compiled with `latexmk -pdf -interaction=nonstopmode -halt-on-error`)

## Decisions (iter_0031)
- `H76` is inconclusive: geometric lift remains positive (`mean delta_AUROC=+0.02323`) but interaction objective is weak (`mean interaction=+0.00041`) and null-surviving interaction appears in only `1/6` domain-splits.
- `H77` is negative: non-edge cross-model rank endpoint is effectively zero (`mean delta_spearman=+5.65e-06`, `mean delta_topk_overlap=+3.56e-05`) with uniformly negative null-gap (`0/12` rows positive).
- `H78` is inconclusive: directional signal is present (`mean delta_AUROC=+0.00193`, positive mean domain-splits `4/6`) but strict null-gap robustness fails (`0/6` domain-splits positive mean null-gap).

## Blockers (iter_0031)
- No hard data/runtime blocker.
- Method-level blocker: cross-model alignment remains null-gap negative even after switching away from edge-transfer endpoint.

## Next Step (iter_0031)
- Retire current `H77` formulation.
- Keep `H78` only as a low-budget probe with materially changed perturbation mechanics.
- Continue H70-lineage biological anchoring only with stronger immune/source-targeted designs and strict null-gap gates.

## Progress Update (iter_0032)
- Implemented and ran `iterations/iter_0032/run_iter0032_screen.py` in `subproject40-topology`.
- Completed a 3-slot breadth packet mapped to brainstormer priorities:
  - `H79` TF-module conditioned support calibration (`N395`),
  - `H80` pathway-centroid cross-model alignment (`N392`),
  - `H81` neighbor-dropout detour elasticity v2 (`N389`).
- Generated machine artifacts:
  - `iterations/iter_0032/h79_tf_module_conditioned_by_seed_layer_split.csv`
  - `iterations/iter_0032/h79_tf_module_conditioned_domain_summary.csv`
  - `iterations/iter_0032/h79_tf_module_conditioned_null_summary.csv`
  - `iterations/iter_0032/h80_pathway_centroid_alignment_by_domain_layer.csv`
  - `iterations/iter_0032/h80_pathway_centroid_alignment_domain_summary.csv`
  - `iterations/iter_0032/h80_pathway_centroid_alignment_null_summary.csv`
  - `iterations/iter_0032/h81_neighbor_dropout_detour_elasticity_by_domain_split_layer.csv`
  - `iterations/iter_0032/h81_neighbor_dropout_detour_elasticity_domain_summary.csv`
  - `iterations/iter_0032/h81_neighbor_dropout_detour_elasticity_null_summary.csv`
  - `iterations/iter_0032/iter0032_screen_summary.json`
- Wrote required iteration outputs:
  - `iterations/iter_0032/executor_iteration_report.md`
  - `iterations/iter_0032/executor_next_steps.md`
  - `iterations/iter_0032/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0032`)
  - `paper/autoloop_research_paper.pdf` (compiled with `latexmk -pdf -interaction=nonstopmode -halt-on-error`)

## Decisions (iter_0032)
- `H79` is inconclusive: module-conditioned calibration increased utility (`mean delta_AUROC=+0.03458`, immune/source `+0.03198`) but robustness did not clear (`mean null-gap > 0` in only `1/6` domain-splits; global interaction mean negative).
- `H80` is negative for promotion: pathway-centroid similarity was directional (`mean Spearman=+0.15032`) but null-gap remained negative in all domains (`0/3` positive domain means).
- `H81` is negative: neighbor-dropout elasticity reduced performance (`mean delta_AUROC=-0.01199`, positive rows `1/24`, positive null-gap domain-splits `0/6`).

## Blockers (iter_0032)
- No hard runtime or data blocker.
- Method-level blocker persists for cross-model alignment endpoints: null-gap robustness remains negative despite major endpoint change.

## Next Step (iter_0032)
- Retire `H81` in its current global-dropout utility form.
- Hold/retire `H80` for utility claims unless re-opened with a materially different pathway-trajectory invariance objective.
- If `H79` is revisited, constrain to a depth-conditional redesign with pre-registered null-gap gate (`>=2/6` positive domain-splits).

## Progress Update (iter_0033)
- Implemented and ran `iterations/iter_0033/run_iter0033_screen.py` in `subproject40-topology`.
- Completed a 3-slot breadth packet mapped to brainstormer priorities:
  - `H82` local witness-cycle persistence on H70 hotspots (`N399`),
  - `H83` cross-model pathway trajectory invariance (`N407`),
  - `H84` shortcut-bridge competition index broad screen (`N412`).
- Generated machine artifacts:
  - `iterations/iter_0033/h82_local_witness_cycle_by_seed_layer_split.csv`
  - `iterations/iter_0033/h82_local_witness_cycle_domain_summary.csv`
  - `iterations/iter_0033/h82_local_witness_cycle_null_summary.csv`
  - `iterations/iter_0033/h83_pathway_trajectory_invariance_by_domain.csv`
  - `iterations/iter_0033/h83_pathway_trajectory_invariance_domain_summary.csv`
  - `iterations/iter_0033/h83_pathway_trajectory_invariance_null_summary.csv`
  - `iterations/iter_0033/h84_shortcut_bridge_competition_by_domain_split_layer.csv`
  - `iterations/iter_0033/h84_shortcut_bridge_competition_domain_summary.csv`
  - `iterations/iter_0033/h84_shortcut_bridge_competition_null_summary.csv`
  - `iterations/iter_0033/iter0033_screen_summary.json`
- Wrote required iteration outputs:
  - `iterations/iter_0033/executor_iteration_report.md`
  - `iterations/iter_0033/executor_next_steps.md`
  - `iterations/iter_0033/executor_hypothesis_screen.json`

## Decisions (iter_0033)
- `H82` (`persistent_homology`, major-change rescue): **promising**. Mean `delta_AUROC=+0.01595`, positive mean domain-splits `6/6`, and positive mean null-gap domain-splits `4/6`.
- `H83` (`cross_model_alignment`, changed invariance endpoint): **negative**. Mean trajectory Spearman `-0.07043`, positive null-gap domains `0/3`.
- `H84` (`graph_topology`, cheap broad screen): **negative**. Mean `delta_AUROC=-0.02803`, positive mean/null-gap domain-splits `0/6` and `0/6`.

## Blockers (iter_0033)
- No runtime or data blocker.
- Method-level blocker persists for cross-model objectives: current trajectory-invariance formulation remained null-gap negative across all domains.

## Next Step (iter_0033)
- Promote `H82` for one confirmatory pass with stricter controls (explicit endpoint-swap null and alternate hotspot thresholding).
- Retire current `H83` and `H84` formulations unless materially changed.
- Keep next packet breadth-focused with one `H82` confirmation slot plus two orthogonal novelty slots.

## Progress Update (iter_0034)
- Implemented and ran `iterations/iter_0034/run_iter0034_screen.py` in `subproject40-topology`.
- Completed a 3-slot breadth packet mapped to brainstormer priorities:
  - `H85` dual-filtration local witness persistence (`N420`, refinement from `H82`),
  - `H86` cross-model barcode OT depth alignment (`N429`, major-change rescue),
  - `H87` sparse descriptor blend breadth screen (`N433`, cheap orthogonal scan).
- Generated machine artifacts:
  - `iterations/iter_0034/h85_dual_filtration_witness_by_seed_layer_split.csv`
  - `iterations/iter_0034/h85_dual_filtration_witness_domain_summary.csv`
  - `iterations/iter_0034/h85_dual_filtration_witness_null_summary.csv`
  - `iterations/iter_0034/h86_barcode_ot_depth_alignment_by_domain.csv`
  - `iterations/iter_0034/h86_barcode_ot_depth_alignment_domain_summary.csv`
  - `iterations/iter_0034/h86_barcode_ot_depth_alignment_null_summary.csv`
  - `iterations/iter_0034/h87_sparse_descriptor_blend_by_domain_split_layer.csv`
  - `iterations/iter_0034/h87_sparse_descriptor_blend_domain_summary.csv`
  - `iterations/iter_0034/h87_sparse_descriptor_blend_null_summary.csv`
  - `iterations/iter_0034/iter0034_screen_summary.json`
- Wrote required iteration outputs:
  - `iterations/iter_0034/executor_iteration_report.md`
  - `iterations/iter_0034/executor_next_steps.md`
  - `iterations/iter_0034/executor_hypothesis_screen.json`

## Decisions (iter_0034)
- `H85` is **neutral**: directional lift is present (`mean delta_AUROC=+0.00360`, positive mean domain-splits `5/6`) but robustness is short of gate (`positive mean null-gap domain-splits=3/6`, gate `>=4/6`).
- `H86` is **negative**: cross-model barcode OT rescue failed null controls in every pilot domain (`positive null-gap domains=0/3`).
- `H87` is **promising**: strong broad-screen effect (`mean delta_AUROC=+0.08035`, positive rows `24/24`, positive mean domain-splits `6/6`, positive mean null-gap domain-splits `4/6`).

## Blockers (iter_0034)
- No data/runtime blocker.
- Method-level blocker remains for cross-model alignment: major-change OT formulation did not clear null-gap criteria.

## Next Step (iter_0034)
- Promote `H87` to multiseed robustness validation with higher null resolution.
- Keep `H85` for one targeted repair pass on failing domain-splits.
- Retire `H86` in current form unless reopened with a materially different topology anchor.

## Progress Update (iter_0035)
- Implemented and ran `iterations/iter_0035/run_iter0035_screen.py` in `subproject40-topology`.
- Completed a 3-slot breadth packet aligned to the prior roadmap:
  - `H88` multiseed sparse-descriptor consensus robustness (`N448`, refinement),
  - `H89` local linearity phase-boundary screen (`N441`, new method),
  - `H90` perturbation topology-stability screen (`N438`, new method).
- Generated machine artifacts:
  - `iterations/iter_0035/h88_multiseed_sparse_descriptor_by_seed_split_layer.csv`
  - `iterations/iter_0035/h88_multiseed_sparse_descriptor_domain_summary.csv`
  - `iterations/iter_0035/h88_multiseed_sparse_descriptor_null_summary.csv`
  - `iterations/iter_0035/h88_multiseed_sparse_descriptor_stability.csv`
  - `iterations/iter_0035/h89_phase_boundary_by_domain_split_layer.csv`
  - `iterations/iter_0035/h89_phase_boundary_domain_summary.csv`
  - `iterations/iter_0035/h89_phase_boundary_null_summary.csv`
  - `iterations/iter_0035/h90_topology_stability_by_domain_split_layer.csv`
  - `iterations/iter_0035/h90_topology_stability_domain_summary.csv`
  - `iterations/iter_0035/h90_topology_stability_null_summary.csv`
  - `iterations/iter_0035/iter0035_screen_summary.json`
- Wrote required iteration outputs:
  - `iterations/iter_0035/executor_iteration_report.md`
  - `iterations/iter_0035/executor_next_steps.md`
  - `iterations/iter_0035/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0035`)
  - `paper/autoloop_research_paper.pdf` (compiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error`)

## Decisions (iter_0035)
- `H88` (`split_robustness`, refinement): **promising**.
  - Mean `delta_AUROC=+0.07603`, positive rows `72/72`, positive mean domain-splits `6/6`, and positive mean null-gap domain-splits `5/6`.
  - Descriptor-core stability is moderate (`mean nonzero-set Jaccard=0.49263`), with weakest robustness slice at `immune/source_disjoint` (`mean null-gap=-0.00264`).
- `H89` (`intrinsic_dimensionality`, new method): **negative**.
  - Directional lift (`mean delta_AUROC=+0.01676`) did not survive controls (`positive mean null-gap domain-splits=0/6`).
- `H90` (`topology_stability`, new method): **negative**.
  - Small directional lift (`mean delta_AUROC=+0.00449`) and weak positive stability trend (`mean stability_pos_minus_neg=+0.00793`) still failed robustness (`positive mean null-gap domain-splits=0/6`).

## Blockers (iter_0035)
- No data/runtime blockers.
- Non-blocking sklearn deprecation warnings appeared during logistic fitting (`penalty` argument), but all artifacts were generated successfully.

## Next Step (iter_0035)
- Keep only one carry-over slot for `H88` with a targeted `immune/source_disjoint` rescue and higher null budget.
- Retire `H89` and `H90` in their tested additive utility forms.
- Preserve breadth by filling the remaining slots with materially novel hypotheses outside retired standalone intrinsic-dimension/additive-stability utility directions.

## Progress Update (iter_0036)
- Implemented and ran `iterations/iter_0036/run_iter0036_screen.py` in `subproject40-topology`.
- Completed a 3-slot breadth packet aligned to prior roadmap:
  - `H91` stability-selected sparse-descriptor consensus (`N449`, carry-over refinement),
  - `H92` scale-space lifetime trajectory descriptors (`N452`, new method),
  - `H93` confidence/sign-weighted filtration rescue (`N458`, new method).
- Generated machine artifacts:
  - `iterations/iter_0036/h91_stability_selected_sparse_descriptor_by_seed_split_layer.csv`
  - `iterations/iter_0036/h91_stability_selected_sparse_descriptor_domain_summary.csv`
  - `iterations/iter_0036/h91_stability_selected_sparse_descriptor_null_summary.csv`
  - `iterations/iter_0036/h91_stability_selected_sparse_descriptor_stability.csv`
  - `iterations/iter_0036/h92_scale_space_lifetime_by_domain_split_layer.csv`
  - `iterations/iter_0036/h92_scale_space_lifetime_domain_summary.csv`
  - `iterations/iter_0036/h92_scale_space_lifetime_null_summary.csv`
  - `iterations/iter_0036/h93_confidence_sign_weighted_filtration_by_domain_split_layer.csv`
  - `iterations/iter_0036/h93_confidence_sign_weighted_filtration_domain_summary.csv`
  - `iterations/iter_0036/h93_confidence_sign_weighted_filtration_null_summary.csv`
  - `iterations/iter_0036/iter0036_screen_summary.json`
- Wrote required iteration outputs:
  - `iterations/iter_0036/executor_iteration_report.md`
  - `iterations/iter_0036/executor_next_steps.md`
  - `iterations/iter_0036/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0036`)
  - `paper/autoloop_research_paper.pdf` (compiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error`)

## Decisions (iter_0036)
- `H91` (`split_robustness`, refinement): **promising**.
  - Mean `delta_AUROC=+0.07424`, positive rows `72/72`, positive mean domain-splits `6/6`, positive mean null-gap domain-splits `6/6`.
  - Descriptor-core stability reached target (`mean nonzero-set Jaccard=0.65046`, sign agreement `1.0`).
- `H92` (`topology_stability`, new method): **negative**.
  - Small directional lift (`mean delta_AUROC=+0.00386`) collapsed under controls (`positive mean null-gap domain-splits=0/6`).
- `H93` (`persistent_homology`, new method): **promising**.
  - Strong uplift (`mean delta_AUROC=+0.08443`, positive rows `12/12`) with positive mean null-gap in all domain-splits (`6/6`).

## Blockers (iter_0036)
- No data/runtime blockers.
- Non-blocking execution note: logistic deprecation warnings were suppressed during the main run to keep output parseable.

## Next Step (iter_0036)
- Promote `H91` and `H93` to higher-null-resolution confirmation runs (and multiseed replication for `H93`).
- Retire the exact additive `H92` trajectory formulation and keep it only as a diagnostic feature family.
- Keep next packet breadth-first with at most one carry-over slot and at least one materially novel branch.

### iter_0037 (current)
- Executed breadth-oriented 3-slot screening with one refinement and two materially changed methods:
  - `H94` (`persistent_homology`, refinement): GO-ontology-stratified weighted filtration vs global weighted baseline.
  - `H95` (`graph_topology`, new method): bridge-curvature descriptor blend with degree-preserving edge-swap null.
  - `H96` (`cross_model_alignment`, rescue-once major change): cross-model GO-module topology-rank concordance.
- Key quantitative results:
  - H94 negative: mean `delta_auc_ontology_weighted_minus_global_weighted = -0.00933`, positive mean domain-splits `0/6`, positive mean null-gap domain-splits `0/6`.
  - H95 inconclusive: mean `delta_auc_graph_bridge_curvature_minus_h70 = +0.07710` with positive rows `24/24`, but positive mean null-gap domain-splits `0/6`.
  - H96 negative: mean module Spearman `-0.00555`, mean null-gap(q95 Spearman) `-0.21467`, positive null-gap domains `0/3`.
- Iteration artifacts:
  - `iterations/iter_0037/h94_ontology_stratified_weighted_filtration_by_seed_split_layer.csv`
  - `iterations/iter_0037/h94_ontology_stratified_weighted_filtration_domain_summary.csv`
  - `iterations/iter_0037/h94_ontology_stratified_weighted_filtration_null_summary.csv`
  - `iterations/iter_0037/h95_graph_bridge_curvature_by_domain_split_layer.csv`
  - `iterations/iter_0037/h95_graph_bridge_curvature_domain_summary.csv`
  - `iterations/iter_0037/h95_graph_bridge_curvature_null_summary.csv`
  - `iterations/iter_0037/h96_cross_model_module_topology_by_domain_layer.csv`
  - `iterations/iter_0037/h96_cross_model_module_topology_domain_summary.csv`
  - `iterations/iter_0037/h96_cross_model_module_topology_null_summary.csv`
  - `iterations/iter_0037/iter0037_screen_summary.json`
  - `iterations/iter_0037/executor_iteration_report.md`
  - `iterations/iter_0037/executor_next_steps.md`
  - `iterations/iter_0037/executor_hypothesis_screen.json`
- Decision:
  - Retire the tested GO-stratified additive refinement (`H94`) and keep global weighted filtration as the active persistent-homology baseline.
  - Keep `H95` as a one-pass inconclusive branch pending a higher-null-resolution rerun.
  - Keep cross-model alignment retired after this rescue-once failure (`H96`).

### iter_0038 (current)
- Executed breadth-oriented 3-slot screening with one carry-over rescue and two materially changed methods:
  - `H97` (`graph_topology`, refinement): calibrated bridge-curvature blend with stricter structure-matched rewiring nulls.
  - `H98` (`intrinsic_dimensionality`, new method): multi-radius ID heterogeneity entropy blend.
  - `H99` (`cross_model_alignment`, structural-reset new method): cross-model GO-module role-graph concordance.
- Key quantitative results:
  - H97 directional but non-robust: mean `delta_auc_graph_bridge_calibrated_minus_h70 = +0.07852`, positive rows `24/24`, positive mean domain-splits `6/6`, but positive mean null-gap domain-splits `0/6`.
  - H98 negative: mean `delta_auc_id_entropy_minus_h70 = -0.00773`, positive rows `5/24`, positive mean domain-splits `1/6`, positive mean null-gap domain-splits `0/6`.
  - H99 negative for rescue gate: mean `module_role_graph_concordance = +0.03934`, but mean null-gap-to-q95 was negative (`-0.02497`) and positive null-gap domains `0/3`.
- Iteration artifacts:
  - `iterations/iter_0038/h97_graph_bridge_calibrated_by_domain_split_layer.csv`
  - `iterations/iter_0038/h97_graph_bridge_calibrated_domain_summary.csv`
  - `iterations/iter_0038/h97_graph_bridge_calibrated_null_summary.csv`
  - `iterations/iter_0038/h98_id_entropy_by_domain_split_layer.csv`
  - `iterations/iter_0038/h98_id_entropy_domain_summary.csv`
  - `iterations/iter_0038/h98_id_entropy_null_summary.csv`
  - `iterations/iter_0038/h99_cross_model_role_graph_by_domain_layer.csv`
  - `iterations/iter_0038/h99_cross_model_role_graph_domain_summary.csv`
  - `iterations/iter_0038/h99_cross_model_role_graph_null_summary.csv`
  - `iterations/iter_0038/iter0038_screen_summary.json`
  - `iterations/iter_0038/executor_iteration_report.md`
  - `iterations/iter_0038/executor_next_steps.md`
  - `iterations/iter_0038/executor_hypothesis_screen.json`
- Decision:
  - Retire the tested additive bridge-curvature utility lineage (`H95/H97`) for promotion claims due repeated `0/6` null-gap domain-split support under adequate controls.
  - Retire the tested standalone/additive ID-entropy formulation (`H98`) as negative.
  - Keep cross-model alignment retired after another structural-reset endpoint failed null robustness (`H99`, `0/3` domains).

### iter_0039 (current)
- Executed breadth-oriented 3-slot packet with materially changed methods:
  - `H100` (`persistent_homology`, new method): relative persistence contrast vs matched background complexes.
  - `H101` (`persistent_homology`, new method): persistence derivative-spectrum additive screen.
  - `H102` (`cross_model_alignment`, rescue-once major change): GO-module OT alignment with monotone depth warp.
- Implemented and ran:
  - `iterations/iter_0039/run_iter0039_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0039/h100_relative_persistence_contrast_by_domain_split_layer.csv`
  - `iterations/iter_0039/h100_relative_persistence_contrast_domain_summary.csv`
  - `iterations/iter_0039/h100_relative_persistence_contrast_null_summary.csv`
  - `iterations/iter_0039/h101_persistence_derivative_spectrum_by_domain_split_layer.csv`
  - `iterations/iter_0039/h101_persistence_derivative_spectrum_domain_summary.csv`
  - `iterations/iter_0039/h101_persistence_derivative_spectrum_null_summary.csv`
  - `iterations/iter_0039/h102_ot_monotone_depth_warp_by_domain.csv`
  - `iterations/iter_0039/h102_ot_monotone_depth_warp_domain_summary.csv`
  - `iterations/iter_0039/h102_ot_monotone_depth_warp_null_summary.csv`
  - `iterations/iter_0039/iter0039_screen_summary.json`
- Wrote required iteration outputs:
  - `iterations/iter_0039/executor_iteration_report.md`
  - `iterations/iter_0039/executor_next_steps.md`
  - `iterations/iter_0039/executor_hypothesis_screen.json`
- Key quantitative results:
  - `H100` negative: mean `delta_auc_relative_ph_minus_h93 = -0.00188`; positive mean null-gap domain-splits `0/6`.
  - `H101` inconclusive: mean `delta_auc_persistence_derivative_minus_h70 = +0.00621`; positive mean null-gap domain-splits `0/6`.
  - `H102` negative rescue: mean `module_persistence_ot_concordance = +0.57065`, but mean null-gap(q95) `-0.09697` and positive null-gap domains `0/3`.
- Decision:
  - Retire the tested `H100` formulation (under baseline and non-robust).
  - Keep `H101` only as a single rescue-once candidate if materially changed (interaction-only use).
  - Re-retire cross-model branch after `H102` failed fast-fail gate (`0/3` domains).

## Executor Update (iter_0040)
- Implemented and executed `iterations/iter_0040/run_iter0040_screen.py` in `subproject40-topology` with a 3-hypothesis breadth packet (`H103/H104/H105`).
- Produced machine-readable outputs:
  - `iterations/iter_0040/h103_interaction_derivative_rescue_by_domain_split_layer.csv`
  - `iterations/iter_0040/h103_interaction_derivative_rescue_domain_summary.csv`
  - `iterations/iter_0040/h103_interaction_derivative_rescue_null_summary.csv`
  - `iterations/iter_0040/h104_depth_motif_grammar_by_domain_split.csv`
  - `iterations/iter_0040/h104_depth_motif_grammar_domain_summary.csv`
  - `iterations/iter_0040/h104_depth_motif_grammar_null_summary.csv`
  - `iterations/iter_0040/h105_string_conditioned_null_calibration_by_domain_split_layer.csv`
  - `iterations/iter_0040/h105_string_conditioned_null_calibration_domain_summary.csv`
  - `iterations/iter_0040/h105_string_conditioned_null_calibration_null_summary.csv`
  - `iterations/iter_0040/iter0040_screen_summary.json`
- Wrote required iteration outputs:
  - `iterations/iter_0040/executor_iteration_report.md`
  - `iterations/iter_0040/executor_next_steps.md`
  - `iterations/iter_0040/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0040`)
  - `paper/autoloop_research_paper.pdf` (compiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error`)

## Decisions (iter_0040)
- `H103` (`persistent_homology`, `N508` interaction-only rescue): **negative**.
  - Mean `delta_auc_interaction_derivative_minus_h91_h93 = -0.00304`; positive mean null-gap domain-splits `0/6`.
- `H104` (`manifold_distance`, `N520` depth motif grammar): **negative**.
  - Mean `delta_auc_motif_grammar_minus_h70 = -0.00908`; positive mean null-gap domain-splits `0/6`.
- `H105` (`null_sensitivity`, `N519` conditioned calibration): **negative** for rescue objective.
  - Mean conditioned-minus-unconditioned null-gap `-0.05125`; positive conditioned-gain domain-splits `0/6`.

## Next Step (iter_0040)
- Retire tested `N508/N520/N519` forms and shift next packet to non-retired branches with strict null-gap gates (one `module_structure`, one `manifold_distance`, one `split_robustness` slot).

### iter_0041 (current)
- Executed breadth-oriented 3-slot packet aligned to brainstormer picks:
  - `H106` (`persistent_homology`, `N538`): STRING triad-closure weighted filtration rescue.
  - `H107` (`topology_stability`, `N537`): finite-state descriptor motif screen with second-order sequence model.
  - `H108` (`cross_model_alignment`, `N531`): cross-model perturbation-response rank alignment.
- Key quantitative results:
  - H106 negative: mean `delta_auc_string_triad_weighted_minus_h93 = -0.00118`; positive mean null-gap domain-splits `0/6`.
  - H107 inconclusive: mean `delta_auc_dfa_motif_minus_h70 = +0.04563` but positive mean null-gap domain-splits `0/6`.
  - H108 promising/mixed: mean `module_response_rank_spearman = +0.73159`; positive domain null-gap count `2/3` (external_lung and lung positive, immune negative).
- Iteration artifacts:
  - `iterations/iter_0041/h106_string_triad_weighted_filtration_by_domain_split_layer.csv`
  - `iterations/iter_0041/h106_string_triad_weighted_filtration_domain_summary.csv`
  - `iterations/iter_0041/h106_string_triad_weighted_filtration_null_summary.csv`
  - `iterations/iter_0041/h107_finite_state_descriptor_motif_by_domain_split.csv`
  - `iterations/iter_0041/h107_finite_state_descriptor_motif_domain_summary.csv`
  - `iterations/iter_0041/h107_finite_state_descriptor_motif_null_summary.csv`
  - `iterations/iter_0041/h108_cross_model_perturbation_response_by_domain.csv`
  - `iterations/iter_0041/h108_cross_model_perturbation_response_domain_summary.csv`
  - `iterations/iter_0041/h108_cross_model_perturbation_response_null_summary.csv`
  - `iterations/iter_0041/iter0041_screen_summary.json`
  - `iterations/iter_0041/executor_iteration_report.md`
  - `iterations/iter_0041/executor_next_steps.md`
  - `iterations/iter_0041/executor_hypothesis_screen.json`
- Decision:
  - Retire `H106/N538` exact formulation (fail-fast gate triggered: `0/6` positive null-gap domain-splits).
  - Keep `H107/N537` as non-promoted/inconclusive due robustness failure.
  - Promote `H108/N531` to multi-seed robustness follow-up with immune-domain failure analysis.

### iter_0042 (current)
- Executed breadth-oriented 3-slot packet from the prior brainstormer brief:
  - `H109` (`cross_model_alignment`, `N546`): multi-seed cross-model perturbation Jacobian alignment rescue.
  - `H110` (`topology_stability`, `N539`): perturbation persistence vineyards.
  - `H111` (`topology_stability`, `N551`): biologically anchored finite-state grammar.
- Implemented and ran:
  - `iterations/iter_0042/run_iter0042_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0042/h109_cross_model_jacobian_alignment_by_seed_domain.csv`
  - `iterations/iter_0042/h109_cross_model_jacobian_alignment_domain_summary.csv`
  - `iterations/iter_0042/h109_cross_model_jacobian_alignment_null_summary.csv`
  - `iterations/iter_0042/h110_persistence_vineyard_by_domain_split_layer.csv`
  - `iterations/iter_0042/h110_persistence_vineyard_domain_summary.csv`
  - `iterations/iter_0042/h110_persistence_vineyard_null_summary.csv`
  - `iterations/iter_0042/h111_bio_anchored_fsm_by_domain_split.csv`
  - `iterations/iter_0042/h111_bio_anchored_fsm_domain_summary.csv`
  - `iterations/iter_0042/h111_bio_anchored_fsm_null_summary.csv`
  - `iterations/iter_0042/iter0042_screen_summary.json`
  - `iterations/iter_0042/executor_iteration_report.md`
  - `iterations/iter_0042/executor_next_steps.md`
  - `iterations/iter_0042/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0042`)
  - `paper/autoloop_research_paper.pdf` (compiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error`)
- Key quantitative results:
  - `H109`: mean response Spearman `+0.79114` and mean Jacobian cosine `+0.52211`, but positive response null-gap rows only `2/9` and positive Jacobian null-gap rows `0/9`; immune null-gap remained negative for all seeds.
  - `H110`: mean `delta_auc_vineyard_features_minus_h93=+0.00091`; positive mean null-gap domain-splits `0/6`.
  - `H111`: mean `delta_auc_biofsm_minus_h70=+0.11202` with positive mean delta in `6/6` domain-splits, but positive mean null-gap only `1/6`.
- Decision:
  - Retire the tested `H109/N546` rescue endpoint (fail-fast immune condition triggered across all seeds).
  - Retire the tested `H110/N539` vineyards formulation (`0/6` positive mean null-gap domain-splits).
  - Keep `H111/N551` as inconclusive (directional but not robust) for at most one materially changed rescue.

### iter_0043 (current)
- Executed breadth-oriented 3-slot packet from the prior brainstormer brief:
  - `H112` (`topology_stability`, `N565`): semi-Markov biologically anchored grammar rescue.
  - `H113` (`persistent_homology`, `N552`): depth-transition zigzag long-bar mass screen.
  - `H114` (`intrinsic_dimensionality`, `N559`): intrinsic-dimension hysteresis broad screen.
- Implemented and ran:
  - `iterations/iter_0043/run_iter0043_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0043/h112_semimarkov_biogrammar_by_domain_split.csv`
  - `iterations/iter_0043/h112_semimarkov_biogrammar_domain_summary.csv`
  - `iterations/iter_0043/h112_semimarkov_biogrammar_null_summary.csv`
  - `iterations/iter_0043/h113_depth_zigzag_longbar_by_domain_split.csv`
  - `iterations/iter_0043/h113_depth_zigzag_longbar_domain_summary.csv`
  - `iterations/iter_0043/h113_depth_zigzag_longbar_null_summary.csv`
  - `iterations/iter_0043/h114_id_hysteresis_by_domain_split_layer.csv`
  - `iterations/iter_0043/h114_id_hysteresis_domain_summary.csv`
  - `iterations/iter_0043/h114_id_hysteresis_null_summary.csv`
  - `iterations/iter_0043/iter0043_screen_summary.json`
  - `iterations/iter_0043/executor_iteration_report.md`
  - `iterations/iter_0043/executor_next_steps.md`
  - `iterations/iter_0043/executor_hypothesis_screen.json`
- Key quantitative results:
  - `H112`: mean `delta_auc_semimarkov_minus_second_order=-0.03805`; positive mean null-gap domain-splits `0/6`.
  - `H113`: mean `delta_long_bar_mass_positive_minus_negative=-155.38889`; positive mean null-gap domain-splits `0/6`.
  - `H114`: mean `delta_auc_id_hysteresis_minus_h70=+0.00026`; positive mean null-gap domain-splits `0/6`.
- Decision:
  - Mark all three tested formulations (`H112/H113/H114`) as negative for promotion in this loop due failed null-gap robustness (and for `H112/H113`, failed directionality on the primary endpoint).

### iter_0044 (current)
- Executed breadth screen with two new hypotheses not used in `iter_0043`:
  - `H115` (`manifold_distance`, N558-style tangent-subspace acceleration across layers).
  - `H116` (`module_structure`, N563-style TRRUST sign-motif interaction).
- Implemented and ran:
  - `iterations/iter_0044/run_iter0044_screen.py`
  - `conda run -n subproject40-topology python iterations/iter_0044/run_iter0044_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0044/h115_tangent_acceleration_by_domain_split.csv`
  - `iterations/iter_0044/h115_tangent_acceleration_domain_summary.csv`
  - `iterations/iter_0044/h115_tangent_acceleration_null_summary.csv`
  - `iterations/iter_0044/h116_trrust_sign_motif_by_domain_split.csv`
  - `iterations/iter_0044/h116_trrust_sign_motif_domain_summary.csv`
  - `iterations/iter_0044/h116_trrust_sign_motif_null_summary.csv`
  - `iterations/iter_0044/iter0044_screen_summary.json`
- Required iteration docs written:
  - `iterations/iter_0044/executor_iteration_report.md`
  - `iterations/iter_0044/executor_next_steps.md`
  - `iterations/iter_0044/executor_hypothesis_screen.json`
- Key quantitative results:
  - `H115`: mean `delta_auc_vs_h70=-0.00622`; positive direction in `2/6` splits; mean null-gap `+0.00156`.
  - `H116`: mean `delta_auc_vs_h70=+0.07810`; positive direction in `6/6` splits; mean null-gap `+0.06989` (`6/6` null-pass splits).
- Decision:
  - Mark `H115` negative/retired in this formulation.
  - Mark `H116` promising; prioritize multi-seed replication with stricter motif decoy controls.

### iter_0045 (current)
- Executed breadth-oriented 3-slot packet with one refinement and two materially new methods:
  - `H118` (`module_structure`, refinement from `H116`): multiseed signed TRRUST motif x graph-community interaction over H70.
  - `H119` (`cross_model_alignment`, new method): disagreement-conditioned cross-model transfer utility.
  - `H120` (`manifold_distance`, new method): geodesic-path curvature-drift descriptors over H70.
- Implemented and ran:
  - `iterations/iter_0045/run_iter0045_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0045/h118_signed_motif_module_by_seed_domain_split.csv`
  - `iterations/iter_0045/h118_signed_motif_module_domain_summary.csv`
  - `iterations/iter_0045/h118_signed_motif_module_null_summary.csv`
  - `iterations/iter_0045/h119_disagreement_gated_transfer_by_domain_split.csv`
  - `iterations/iter_0045/h119_disagreement_gated_transfer_domain_summary.csv`
  - `iterations/iter_0045/h119_disagreement_gated_transfer_null_summary.csv`
  - `iterations/iter_0045/h120_geodesic_curvature_drift_by_domain_split_layer.csv`
  - `iterations/iter_0045/h120_geodesic_curvature_drift_domain_summary.csv`
  - `iterations/iter_0045/h120_geodesic_curvature_drift_null_summary.csv`
  - `iterations/iter_0045/iter0045_screen_summary.json`
  - `iterations/iter_0045/executor_iteration_report.md`
  - `iterations/iter_0045/executor_next_steps.md`
  - `iterations/iter_0045/executor_hypothesis_screen.json`
- Key quantitative results:
  - `H118`: mean `delta_vs_h70=+0.09885` (18 rows), positive mean null-gap domain-splits `3/6`.
  - `H119`: mean `delta_vs_h70=+0.00060` (6 rows), positive mean null-gap domain-splits `1/6`.
  - `H120`: mean `delta_vs_h70=+0.03854` (12 rows), positive mean null-gap domain-splits `3/6`.
- Decision:
  - `H118` marked promising but not yet fully robust (partial null-gap survival).
  - `H119` marked negative and retired in this endpoint form.
  - `H120` kept neutral for higher-null follow-up.

### iter_0046 (current)
- Executed breadth-oriented 3-slot packet aligned to prior brainstormer picks:
  - `H121` (`manifold_distance`, `N605`): directional geodesic asymmetry over H70.
  - `H122` (`cross_model_alignment`, `N609` major reset): cross-model landscape transport.
  - `H123` (`module_structure`, `N600` refinement): strict signed motif-community hardening with dual-axis split included when feasible.
- Implemented and ran:
  - `iterations/iter_0046/run_iter0046_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0046/h121_directional_geodesic_asymmetry_by_domain_split_layer.csv`
  - `iterations/iter_0046/h121_directional_geodesic_asymmetry_domain_summary.csv`
  - `iterations/iter_0046/h121_directional_geodesic_asymmetry_null_summary.csv`
  - `iterations/iter_0046/h122_landscape_transport_by_domain_split_layer.csv`
  - `iterations/iter_0046/h122_landscape_transport_domain_summary.csv`
  - `iterations/iter_0046/h122_landscape_transport_null_summary.csv`
  - `iterations/iter_0046/h123_signed_motif_module_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0046/h123_signed_motif_module_hardening_domain_summary.csv`
  - `iterations/iter_0046/h123_signed_motif_module_hardening_null_summary.csv`
  - `iterations/iter_0046/iter0046_screen_summary.json`
- Wrote required iteration artifacts:
  - `iterations/iter_0046/executor_iteration_report.md`
  - `iterations/iter_0046/executor_next_steps.md`
  - `iterations/iter_0046/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0046`)
  - `paper/autoloop_research_paper.pdf` (compiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error`)
- Key quantitative results:
  - `H121`: mean `delta_vs_h70=+0.03273` (`11/12` positive rows); positive mean null-gap domain-splits `2/6`.
  - `H122`: mean `transport_score_neg_mse=-100.54990`; positive mean null-gap domain-splits `0/6`.
  - `H123`: mean `delta_vs_h70=+0.09351` (`22/22` positive rows); positive mean null-gap domain-splits `8/8` observed.
- Decision:
  - Keep `H121` as neutral-positive rescue (directionally consistent, partial null support).
  - Retire `H122` endpoint as negative after major objective reset fails all null-gap checks.
  - Promote `H123` as the top active branch; strict nulls retained strong support, with one remaining coverage gap (`lung/dual_axis_disjoint`).

### iter_0047 (current)
- Executed breadth-oriented 3-slot packet from prior brainstormer guidance:
  - `H124` (`module_structure`, `N625` refinement): signed motif-community hardening with STRING conditioning.
  - `H125` (`cross_model_alignment`, `N622` major-reset exploration): anchor-constrained cycle-consistent cross-model alignment.
  - `H126` (`manifold_distance`, `N620` cheap broad screen): geodesic torsion and turning-angle asymmetry.
- Implemented and ran:
  - `iterations/iter_0047/run_iter0047_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0047/h124_signed_string_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0047/h124_signed_string_hardening_domain_summary.csv`
  - `iterations/iter_0047/h124_signed_string_hardening_null_summary.csv`
  - `iterations/iter_0047/h125_anchor_cycle_alignment_by_domain_split_layer.csv`
  - `iterations/iter_0047/h125_anchor_cycle_alignment_domain_split_summary.csv`
  - `iterations/iter_0047/h125_anchor_cycle_alignment_domain_summary.csv`
  - `iterations/iter_0047/h125_anchor_cycle_alignment_null_summary.csv`
  - `iterations/iter_0047/h126_geodesic_torsion_by_domain_split_layer.csv`
  - `iterations/iter_0047/h126_geodesic_torsion_domain_summary.csv`
  - `iterations/iter_0047/h126_geodesic_torsion_null_summary.csv`
  - `iterations/iter_0047/iter0047_screen_summary.json`
- Wrote required iteration artifacts:
  - `iterations/iter_0047/executor_iteration_report.md`
  - `iterations/iter_0047/executor_next_steps.md`
  - `iterations/iter_0047/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0047`)
  - `paper/autoloop_research_paper.pdf` (compiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error`)
- Key quantitative results:
  - `H124`: mean `delta_vs_h70=+0.13098` (27 rows), positive mean null-gap domain-splits `4/9`; lung dual-axis coverage restored (`3` rows) but mean null-gap remained `-0.00326`.
  - `H125`: mean `transfer_delta_auc_vs_h70=+0.09855` (12 rows), but positive domain null-gap count `0/3` and immune mean null-gap `-0.01894`.
  - `H126`: mean `delta_vs_h70=+0.04421` (12 rows), positive mean null-gap domain-splits `2/6` with one source-disjoint split null-positive.
- Decision:
  - Keep `H124` active as neutral refinement (directional strength, insufficient strict-null robustness for promotion).
  - Retire this `H125` endpoint as negative (fails pre-registered domain null-gap gate after objective reset).
  - Mark `H126` as promising for targeted hardening (fast-screen gate passed with non-zero null-surviving support).

### iter_0048 (current)
- Executed a breadth-oriented 3-slot packet with one refinement, one new family, and one materially changed rescue method:
  - `H127` (`module_structure`, `N641`-style refinement): signed motif-community + STRING hardening with explicit GO co-membership interactions and GO-membership permutation null.
  - `H128` (`graph_topology`, new family): graph curvature/community surrogate screen over residual kNN graphs.
  - `H129` (`manifold_distance`, `N634`-style changed method): multi-scale torsion spectrum (`k={8,12,16}`) with scale-order permutation null.
- Implemented and ran:
  - `iterations/iter_0048/run_iter0048_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0048/h127_signed_string_go_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0048/h127_signed_string_go_hardening_domain_summary.csv`
  - `iterations/iter_0048/h127_signed_string_go_hardening_null_summary.csv`
  - `iterations/iter_0048/h128_graph_topology_surrogate_by_domain_split_layer.csv`
  - `iterations/iter_0048/h128_graph_topology_surrogate_domain_summary.csv`
  - `iterations/iter_0048/h128_graph_topology_surrogate_null_summary.csv`
  - `iterations/iter_0048/h129_multiscale_torsion_by_domain_split_layer.csv`
  - `iterations/iter_0048/h129_multiscale_torsion_domain_summary.csv`
  - `iterations/iter_0048/h129_multiscale_torsion_null_summary.csv`
  - `iterations/iter_0048/iter0048_screen_summary.json`
- Wrote required iteration artifacts:
  - `iterations/iter_0048/executor_iteration_report.md`
  - `iterations/iter_0048/executor_next_steps.md`
  - `iterations/iter_0048/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0048`)
  - `paper/autoloop_research_paper.pdf` (compiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error`)
- Key quantitative results:
  - `H127`: mean `delta_vs_h70=+0.13222` (9 rows), positive mean delta domain-splits `9/9`, but positive mean null-gap domain-splits `2/9`; `lung/dual_axis_disjoint` mean null-gap `-0.00596`.
  - `H128`: mean `delta_vs_h70=+0.00753` (12 rows), positive mean delta domain-splits `4/6`, but positive mean null-gap domain-splits `0/6`.
  - `H129`: mean `delta_vs_h70=+0.02100` (12 rows), positive mean delta domain-splits `5/6`, but positive mean null-gap domain-splits `0/6`.
- Decision:
  - Keep `H127` as active/neutral (directionally strong, strict-null robustness still insufficient).
  - Mark `H128` as negative and retire this surrogate endpoint (`0/6` positive mean null-gap domain-splits).
  - Mark `H129` as negative in this formulation (`0/6` positive mean null-gap domain-splits); only revisit with major method changes.

### iter_0049 (current)
- Executed the planned 3-slot packet from prior brainstormer guidance:
  - `H130` (`module_structure`, `N656` refinement): continuous GO-semantic x STRING hardening on top of H127 lineage.
  - `H131` (`cross_model_alignment`, `N653` major-change rescue): chart/sheaf-style local alignment with cycle-consistency diagnostics.
  - `H132` (`manifold_distance`, `N650` cheap broad-screen): local chart-fracture descriptors on directed geodesic paths.
- Implemented and ran:
  - `iterations/iter_0049/run_iter0049_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0049/h130_semantic_go_string_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0049/h130_semantic_go_string_hardening_domain_summary.csv`
  - `iterations/iter_0049/h130_semantic_go_string_hardening_null_summary.csv`
  - `iterations/iter_0049/h131_chart_sheaf_alignment_by_domain_split_layer.csv`
  - `iterations/iter_0049/h131_chart_sheaf_alignment_domain_split_summary.csv`
  - `iterations/iter_0049/h131_chart_sheaf_alignment_domain_summary.csv`
  - `iterations/iter_0049/h131_chart_sheaf_alignment_null_summary.csv`
  - `iterations/iter_0049/h132_chart_fracture_by_domain_split_layer.csv`
  - `iterations/iter_0049/h132_chart_fracture_domain_summary.csv`
  - `iterations/iter_0049/h132_chart_fracture_null_summary.csv`
  - `iterations/iter_0049/iter0049_screen_summary.json`
- Wrote required iteration artifacts:
  - `iterations/iter_0049/executor_iteration_report.md`
  - `iterations/iter_0049/executor_next_steps.md`
  - `iterations/iter_0049/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0049`)
  - `paper/autoloop_research_paper.pdf` (compiled via `latexmk -pdf -interaction=nonstopmode -halt-on-error`)
- Key quantitative results:
  - `H130`: mean `delta_vs_h70=+0.13096` (27 rows), positive mean delta domain-splits `9/9`, but positive mean null-gap domain-splits `0/9`; `lung/dual_axis_disjoint` mean null-gap `-0.00541`, `immune/source_disjoint` mean null-gap `-0.01486`.
  - `H131`: mean `alignment_delta_vs_random=-0.00293` (12 rows), positive mean null-gap domains `0/3`; immune mean null-gap `-0.14339`.
  - `H132`: mean `delta_vs_h70=+0.01637` (12 rows), positive mean delta domain-splits `4/6`, but positive mean null-gap domain-splits `0/6`.
- Decision:
  - Keep `H130` as active-but-neutral (strong directionality, no strict-null survival).
  - Mark `H131` negative and retire this endpoint.
  - Mark `H132` negative and retire this endpoint in current formulation.

### iter_0050 (current)
- Executed a 3-slot breadth screen with two novel tests and one carry-over refinement:
  - `H133` (`persistent_homology`): rank-surface filtration topology surrogate.
  - `H134` (`intrinsic_dimensionality`): TWO-NN path-phase descriptor screen.
  - `H135` (`module_structure`, refinement): hard-slice semantic motif-community rerun of the `H130` lineage.
- Key quantitative results:
  - H133 negative: mean `delta_vs_h70 = -0.04611` over `6` rows; positive mean null-gap domain-splits `0/6`.
  - H134 negative/mixed: mean `delta_vs_h70 = +0.01132` over `12` rows, but positive mean null-gap domain-splits `0/6`.
  - H135 negative/mixed: mean `delta_vs_h70 = +0.13870` over `12` rows, but positive mean null-gap domain-splits `0/4`; hard slices remained negative (`lung/dual_axis_disjoint = -0.00502`, `immune/source_disjoint = -0.01473`).
- Iteration artifacts:
  - `iterations/iter_0050/h133_rank_surface_persistence_by_domain_split.csv`
  - `iterations/iter_0050/h133_rank_surface_persistence_domain_summary.csv`
  - `iterations/iter_0050/h133_rank_surface_persistence_null_summary.csv`
  - `iterations/iter_0050/h134_id_phase_descriptor_by_domain_split_layer.csv`
  - `iterations/iter_0050/h134_id_phase_descriptor_domain_summary.csv`
  - `iterations/iter_0050/h134_id_phase_descriptor_null_summary.csv`
  - `iterations/iter_0050/h135_hard_slice_semantic_refinement_by_seed_domain_split.csv`
  - `iterations/iter_0050/h135_hard_slice_semantic_refinement_domain_summary.csv`
  - `iterations/iter_0050/h135_hard_slice_semantic_refinement_null_summary.csv`
  - `iterations/iter_0050/iter0050_screen_summary.json`
  - `iterations/iter_0050/executor_iteration_report.md`
  - `iterations/iter_0050/executor_next_steps.md`
  - `iterations/iter_0050/executor_hypothesis_screen.json`
- Decision:
  - Retire the tested `H133` rank-surface topology surrogate.
  - Keep `H134` as negative in this formulation (directional-only; no null survival).
  - Retire the tested H130-style hard-slice semantic refinement endpoint (`H135`) after repeated strict-null failure on both designated hard slices.

### iter_0051 (current)
- Executed a breadth-oriented 3-slot packet aligned to the previous brainstorm shortlist:
  - `H136` (`manifold_distance`, `N680`): sectional anisotropy + tangent-orientation manifold screen.
  - `H137` (`cross_model_alignment`, `N684` major-change rescue): correspondence-free cross-model topology-descriptor alignment.
  - `H138` (`module_structure`, `N686`): ontology sheaf-obstruction hardening over signed motif-community backbone.
- Implemented and ran:
  - `iterations/iter_0051/run_iter0051_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0051/h136_sectional_anisotropy_by_domain_split_layer.csv`
  - `iterations/iter_0051/h136_sectional_anisotropy_domain_split_summary.csv`
  - `iterations/iter_0051/h136_sectional_anisotropy_null_summary.csv`
  - `iterations/iter_0051/h137_correspondence_free_alignment_by_domain_split_layer.csv`
  - `iterations/iter_0051/h137_correspondence_free_alignment_domain_split_summary.csv`
  - `iterations/iter_0051/h137_correspondence_free_alignment_domain_summary.csv`
  - `iterations/iter_0051/h137_correspondence_free_alignment_null_summary.csv`
  - `iterations/iter_0051/h138_ontology_sheaf_hardening_by_seed_domain_split.csv`
  - `iterations/iter_0051/h138_ontology_sheaf_hardening_domain_split_summary.csv`
  - `iterations/iter_0051/h138_ontology_sheaf_hardening_domain_summary.csv`
  - `iterations/iter_0051/h138_ontology_sheaf_hardening_null_summary.csv`
  - `iterations/iter_0051/iter0051_screen_summary.json`
- Wrote required iteration artifacts:
  - `iterations/iter_0051/executor_iteration_report.md`
  - `iterations/iter_0051/executor_next_steps.md`
  - `iterations/iter_0051/executor_hypothesis_screen.json`
- Key quantitative results:
  - `H136`: mean `delta_vs_h70=+0.02572` (12 rows), positive mean delta domain-splits `5/6`, positive mean null-gap domain-splits `2/6`.
  - `H137`: mean `alignment_delta_vs_random=+0.00194` (12 rows), positive domain mean null-gap `0/3`, immune mean null-gap `-0.09748`.
  - `H138`: mean `delta_vs_h70=+0.13381` (27 rows), positive mean delta domain-splits `9/9`, but positive mean null-gap domain-splits `0/9`; hard slices stayed negative (`immune/source=-0.00567`, `lung/dual_axis=-0.00476`).
- Decision:
  - Promote `H136` as the only branch passing its bounded keep gate this iteration.
  - Retire the tested `H137` correspondence-free cross-model endpoint.
  - Mark this additive `H138` ontology-sheaf endpoint as negative in current form.

### iter_0052 (current)
- Executed a breadth-oriented 3-slot packet with one carry-over and two novel tests:
  - `H139` (`manifold_distance`, refinement of `H136`): sectional anisotropy robustness expansion across seeds `42/43/44` and source/target/dual-axis disjoint splits.
  - `H140` (`topology_stability`, new family): neighborhood-size scaling (`k={8,12,16}`) with swap-control gain metric.
  - `H141` (`null_sensitivity`, new method): strict max-null fragility audit over this iteration's `H139` multi-null outputs.
- Implemented and ran:
  - `iterations/iter_0052/run_iter0052_screen.py`
- Generated machine artifacts:
  - `iterations/iter_0052/h139_sectional_seed_robustness_by_seed_domain_split.csv`
  - `iterations/iter_0052/h139_sectional_seed_robustness_domain_split_summary.csv`
  - `iterations/iter_0052/h139_sectional_seed_robustness_domain_summary.csv`
  - `iterations/iter_0052/h139_sectional_seed_robustness_null_summary.csv`
  - `iterations/iter_0052/h140_neighborhood_scaling_by_domain_split_k.csv`
  - `iterations/iter_0052/h140_neighborhood_scaling_domain_split_summary.csv`
  - `iterations/iter_0052/h140_neighborhood_scaling_domain_summary.csv`
  - `iterations/iter_0052/h141_strict_null_sensitivity_row_summary.csv`
  - `iterations/iter_0052/h141_strict_null_sensitivity_domain_split_summary.csv`
  - `iterations/iter_0052/h141_strict_null_sensitivity_domain_summary.csv`
  - `iterations/iter_0052/h141_strict_null_sensitivity_nullkind_summary.csv`
  - `iterations/iter_0052/iter0052_screen_summary.json`
- Wrote required iteration outputs:
  - `iterations/iter_0052/executor_iteration_report.md`
  - `iterations/iter_0052/executor_next_steps.md`
  - `iterations/iter_0052/executor_hypothesis_screen.json`
- Updated cumulative records:
  - `reports/autoloop_master_log.md`
  - `paper/autoloop_research_paper.tex` (added marker `ITERATION UPDATE: iter_0052`)
  - `paper/autoloop_research_paper.pdf` (recompiled with `latexmk -pdf -interaction=nonstopmode -halt-on-error`)
- Key quantitative results:
  - `H139`: mean `delta_vs_h70=+0.03135` (25 rows), positive mean null-gap domain-splits `6/9`, strict-positive rows `16/25`.
  - `H140`: mean `delta_gain_vs_swap=+0.03374` (24 rows), positive gain rows `19/24`, positive split gains `6/8`.
  - `H141`: mean `strict_margin=-0.00523` (25 rows), strict-positive rows `15/25`, strict-positive domain-splits `3/9`.
- Decision:
  - Promote `H139` as the strongest active branch with expanded multiseed support.
  - Keep `H140` as neutral-positive pending multiseed replication.
  - Keep `H141` as inconclusive/failure-map evidence; use it to target rescue on failing slices rather than broad tuning.
