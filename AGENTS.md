# Permanent Project Instructions

1. `PROJECT_BRIEF.md` is the source of truth.
2. Work one phase at a time.
3. Follow the exact phase order defined in the brief.
4. Keep reusable logic inside `src/`.
5. Use Google Colab as the main experiment environment.
6. Keep the Colab notebook as the main execution interface.
7. Do not commit the PaySim CSV to GitHub.
8. Load PaySim from a configurable Google Drive path.
9. `isFraud` is the target.
10. Drop `isFlaggedFraud`.
11. Prevent train/test leakage.
12. Split before SMOTE.
13. Apply SMOTE only to training data.
14. Never SMOTE validation or test data.
15. Never use adversarial test samples for training.
16. Adversarial hardening samples must originate only from training data.
17. Keep baseline and hardened models separate.
18. Do not overwrite baseline results.
19. Use fixed random seeds.
20. Keep code memory-aware for free Google Colab.
21. Support both `DEVELOPMENT_MODE` and `FULL_MODE`.
22. Primary metrics are Precision, Recall, F1-score, and PR-AUC.
23. Accuracy is supplementary only.
24. Fraud Recall is especially important.
25. Never fabricate experiment results.
26. All final metrics must come from actual execution.
27. Preserve feature names for SHAP and Streamlit.
28. Verify ART estimator/attack compatibility before implementation.
29. Keep adversarial feature changes realistic and constrained.
30. Clearly label simulated or conceptual components.
31. Real-time simulation must not be described as a live bank feed.
32. Concept drift must be described as simulated drift because PaySim is synthetic.
33. Do not rewrite working phases unnecessarily.
34. Make the smallest correct fix when debugging.
35. Keep code understandable and reasonably commented.
