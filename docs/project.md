Project in one line
Predict which companies will go bankrupt using financial ratios, by first clustering companies into subgroups and then training stacked models inside each subgroup, evaluated with the special Eq(1) accuracy and a <20% predicted‑bankrupt constraint on test.

What is already done
Loaded the train data (5807×97, 198 bankrupt vs 5609 non‑bankrupt).​

Built a 3.1 preprocessing pipeline:

Dropped Index and Bankrupt? as features.

Used RandomForest feature importance to select 40 best features.

Fitted a scaler and saved everything via joblib so we can reuse it later.​

Ran 3.2 clustering on the scaled 40‑dimensional features and fixed k = 7:

Final cluster stats:

Cluster 0: 728 rows, 132 bankrupt

Cluster 1: 701 rows, 0 bankrupt

Cluster 2: 2059 rows, 6 bankrupt

Cluster 3: 1 row, 1 bankrupt

Cluster 4: 2313 rows, 56 bankrupt

Cluster 5: 2 rows, 0 bankrupt

Cluster 6: 3 rows, 3 bankrupt

Saved train_with_clusters.csv and per‑cluster CSVs (cluster_0.csv, …, cluster_6.csv).​

Work assignment going forward
A (me)

3.1 and 3.2 are done.

3.3.2: build a full stacking model for cluster 0 (728 / 132), compute TT/TF and Eq(1) on the cluster‑0 train rows, and save cluster0_stacking.joblib.​

B

3.3.2: build a stacking model for cluster 4 (2313 / 56), with its own preprocessing, ≥3 base models, and a meta‑model, then compute TT/TF and Eq(1) and save cluster4_stacking_B.joblib.​

3.3.1: build the cluster‑ID classifier on train_with_clusters.csv using our 40‑feature preprocessing, save cluster_id_model.joblib, and report accuracy / important features.​

C

3.3.2: build a stacking model for cluster 2 (2059 / 6), focusing on handling heavy imbalance (can use imbalanced‑learn if needed), then compute TT/TF and Eq(1) and save cluster2_stacking_C.joblib.​

Section 4 co‑lead: help build the generalization notebook (test CSV → preprocessing → cluster‑ID model → correct subgroup model → enforce <20% bankrupt → submission CSV).​

D

3.3.2: also build an independent stacking model on cluster 2 (same data, different design), with its own TT/TF, Eq(1), and joblib cluster2_stacking_D.joblib, so D has a graded subgroup.​

Section 4 co‑lead with C: implement and test the full inference pipeline, including constant‑0 handling for clusters 1 and 5 and constant‑1 handling for tiny clusters 3 and 6, plus file‑format checks.​

Once all subgroup models and the cluster‑ID model are done, we’ll:

Fill Table 3 with TT, TF, and feature counts per modeled cluster (0, 2, 4, etc., plus Constant rows for 1 and 5).​

Finish the DOCX report (subgroup summaries + Table 3) and the video explaining the workflow and roles.

For B – Cluster 4 + 3.3.1
Main subgroup (3.3.2): cluster 4

Use cluster_4.csv (rows where cluster == 4 from train_with_clusters.csv).​

Build a full stacking model for bankruptcy prediction in this subgroup:

Define X_sub and y_sub (Bankrupt?) for cluster 4.​

Do any subgroup‑specific preprocessing and feature selection you want (can reuse A’s 40 features or choose a smaller subset just for this subgroup).​

Train at least three non‑parametric base models (e.g., RandomForest, GradientBoosting, XGBoost/LightGBM if allowed, k‑NN, decision tree, etc.).​

Train a meta‑model that takes base models’ outputs (e.g., predicted probabilities) and predicts Bankrupt?.​

Use cross‑validation inside the subgroup to tune and evaluate.​

On the final model, run predictions on the original cluster‑4 train rows and compute TT, TF and Eq(1) accuracy; these numbers go into Table 3.​

Save everything (subgroup preprocessing + fitted stacking model) as one joblib object, e.g. cluster4_stacking_B.joblib.​

Create an individual notebook B_Cluster4.ipynb showing preprocessing decisions, confusion matrix, TT/TF, Eq(1) accuracy, and the feature count used.​

Extra responsibility: 3.3.1 Cluster‑ID Prediction

Take train_with_clusters.csv and A’s preprocessing joblib (preprocess_for_clustering.joblib).​

Build a classification model to predict cluster from the same 40 features used for clustering:

X_cluster = transform_for_clustering(df) using A’s function / joblib.​

y_cluster = df["cluster"].

Train any supervised classifier (e.g., RandomForest, GradientBoosting, SVC, etc.) to predict cluster ID.​

Report accuracy (and maybe per‑cluster confusion) and identify important features for cluster prediction.​

Save the fitted model as cluster_id_model.joblib; this will be used by C and D in Section 4.​

Include this work in the team 3.1–3.3.1 notebook.​

For C – Cluster 2 + Section 4 co‑lead
Main subgroup (3.3.2): cluster 2

Use cluster_2.csv (rows where cluster == 2, 2059 rows with 6 bankrupt).​

Build a stacking model specialized for this heavily imbalanced subgroup:

Carefully design preprocessing (possibly oversampling with imbalanced-learn if you choose, which the spec allows as a last resort).​

Select a sensible subset of features (could start from A’s 40 and reduce further).​

Train ≥3 non‑parametric base models and a meta‑model, with cross‑validation focused on Eq(1) and catching rare bankrupts.​

Compute TT, TF, Eq(1) accuracy on the original cluster‑2 train data, record feature count, and save everything as cluster2_stacking_C.joblib.​

Provide an individual notebook C_Cluster2.ipynb with confusion matrix, TT/TF, and model details.​

Extra responsibility: Section 4 generalization (co‑owner with D)

Work with D to build the team generalization notebook, which will:

Load the test CSV once it is posted.​

Apply A’s preprocessing joblib (preprocess_for_clustering.joblib) to get the 40‑dimensional features.​

Use B’s cluster‑ID model (cluster_id_model.joblib) to assign each test company to a cluster.​

Route each test row to the correct subgroup stacking model:

cluster 0 → A’s joblib

cluster 2 → C’s joblib

cluster 4 → B’s joblib

clusters 1 and 5 → constant 0

clusters 3 and 6 → constant 1 (or handled as you decide and document).​

Enforce the “< 20 % predicted bankrupt” rule on the test set; if the raw model predictions exceed this, implement a post‑processing step to limit positives while prioritizing the most confident.​

Output Index,Bankrupt? CSV exactly in the required format.​

For D – Cluster 2 (second model) + Section 4 co‑lead
Main subgroup (3.3.2): cluster 2 (alternate model)

Also use cluster_2.csv and build an independent stacking model (different architecture / features / hyperparameters than C’s).​

Follow the same rules: ≥3 non‑parametric base models, one meta‑model, cross‑validation, TT/TF and Eq(1) on the original cluster‑2 train data, and a single joblib bundle, e.g. cluster2_stacking_D.joblib.​

Your individual grade for 3.3.2 will use your best subgroup model (here, your own cluster‑2 model).​

Provide an individual notebook D_Cluster2.ipynb documenting your variant of the stacking pipeline.​

Extra responsibility: Section 4 generalization (with C)

Co‑own the Section 4 pipeline with C:

Help design and code the inference flow: load joblibs, call transform_for_clustering, predict cluster IDs, dispatch to subgroup models, enforce the 20 % rule, and create the submission CSV.​

Add checks and logging to verify that the number of predicted bankrupts, index alignment, and file format all meet the spec before submission.

