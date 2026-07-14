# LAMP-Merge Full-Scope Interaction Hyperparameter Analysis

Experiment root: `outputs/lamp_merge_hparam_interaction_full_20260714_hparam_interaction_local5x5_rerun`.

Every grid point evaluates the full formal medical benchmark: five datasets, four vision backbones, three client counts, and three Dirichlet skew levels. A complete point therefore contains 180 raw cells and 60 client-average cells.

The diagnostic-prototype grid fixes a value of $\gamma$ for each curve and scans $s$. The prevalence-calibration grid fixes a value of $\tau$ for each curve and scans $\lambda$.

| Module | Curve | X value | Raw cells | Raw mean Acc | Client-average cells | Client-average mean Acc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| diagnostic prototype reconstruction | $\gamma=0.35$ | 15.00 | 180 | 0.6181 | 60 | 0.6181 |
| diagnostic prototype reconstruction | $\gamma=0.35$ | 17.50 | 180 | 0.6186 | 60 | 0.6186 |
| diagnostic prototype reconstruction | $\gamma=0.35$ | 20.00 | 180 | 0.6184 | 60 | 0.6184 |
| diagnostic prototype reconstruction | $\gamma=0.35$ | 22.50 | 180 | 0.6180 | 60 | 0.6180 |
| diagnostic prototype reconstruction | $\gamma=0.35$ | 25.00 | 180 | 0.6172 | 60 | 0.6172 |
| diagnostic prototype reconstruction | $\gamma=0.40$ | 15.00 | 180 | 0.6196 | 60 | 0.6196 |
| diagnostic prototype reconstruction | $\gamma=0.40$ | 17.50 | 180 | 0.6201 | 60 | 0.6201 |
| diagnostic prototype reconstruction | $\gamma=0.40$ | 20.00 | 180 | 0.6199 | 60 | 0.6199 |
| diagnostic prototype reconstruction | $\gamma=0.40$ | 22.50 | 180 | 0.6195 | 60 | 0.6195 |
| diagnostic prototype reconstruction | $\gamma=0.40$ | 25.00 | 180 | 0.6187 | 60 | 0.6187 |
| diagnostic prototype reconstruction | $\gamma=0.45$ | 15.00 | 180 | 0.6206 | 60 | 0.6206 |
| diagnostic prototype reconstruction | $\gamma=0.45$ | 17.50 | 180 | 0.6211 | 60 | 0.6211 |
| diagnostic prototype reconstruction | $\gamma=0.45$ | 20.00 | 180 | 0.6209 | 60 | 0.6209 |
| diagnostic prototype reconstruction | $\gamma=0.45$ | 22.50 | 180 | 0.6205 | 60 | 0.6205 |
| diagnostic prototype reconstruction | $\gamma=0.45$ | 25.00 | 180 | 0.6197 | 60 | 0.6197 |
| diagnostic prototype reconstruction | $\gamma=0.50$ | 15.00 | 180 | 0.6214 | 60 | 0.6214 |
| diagnostic prototype reconstruction | $\gamma=0.50$ | 17.50 | 180 | 0.6219 | 60 | 0.6219 |
| diagnostic prototype reconstruction | $\gamma=0.50$ | 20.00 | 180 | 0.6219 | 60 | 0.6219 |
| diagnostic prototype reconstruction | $\gamma=0.50$ | 22.50 | 180 | 0.6213 | 60 | 0.6213 |
| diagnostic prototype reconstruction | $\gamma=0.50$ | 25.00 | 180 | 0.6205 | 60 | 0.6205 |
| diagnostic prototype reconstruction | $\gamma=0.55$ | 15.00 | 180 | 0.6219 | 60 | 0.6219 |
| diagnostic prototype reconstruction | $\gamma=0.55$ | 17.50 | 180 | 0.6223 | 60 | 0.6223 |
| diagnostic prototype reconstruction | $\gamma=0.55$ | 20.00 | 180 | 0.6224 | 60 | 0.6224 |
| diagnostic prototype reconstruction | $\gamma=0.55$ | 22.50 | 180 | 0.6218 | 60 | 0.6218 |
| diagnostic prototype reconstruction | $\gamma=0.55$ | 25.00 | 180 | 0.6209 | 60 | 0.6209 |
| long-tail prevalence calibration | $\tau=1.50$ | 4.00 | 180 | 0.6182 | 60 | 0.6182 |
| long-tail prevalence calibration | $\tau=1.50$ | 4.50 | 180 | 0.6181 | 60 | 0.6181 |
| long-tail prevalence calibration | $\tau=1.50$ | 5.00 | 180 | 0.6178 | 60 | 0.6178 |
| long-tail prevalence calibration | $\tau=1.50$ | 5.50 | 180 | 0.6170 | 60 | 0.6170 |
| long-tail prevalence calibration | $\tau=1.50$ | 6.00 | 180 | 0.6154 | 60 | 0.6154 |
| long-tail prevalence calibration | $\tau=2.00$ | 4.00 | 180 | 0.6198 | 60 | 0.6198 |
| long-tail prevalence calibration | $\tau=2.00$ | 4.50 | 180 | 0.6205 | 60 | 0.6205 |
| long-tail prevalence calibration | $\tau=2.00$ | 5.00 | 180 | 0.6209 | 60 | 0.6209 |
| long-tail prevalence calibration | $\tau=2.00$ | 5.50 | 180 | 0.6211 | 60 | 0.6211 |
| long-tail prevalence calibration | $\tau=2.00$ | 6.00 | 180 | 0.6207 | 60 | 0.6207 |
| long-tail prevalence calibration | $\tau=2.50$ | 4.00 | 180 | 0.6198 | 60 | 0.6198 |
| long-tail prevalence calibration | $\tau=2.50$ | 4.50 | 180 | 0.6205 | 60 | 0.6205 |
| long-tail prevalence calibration | $\tau=2.50$ | 5.00 | 180 | 0.6209 | 60 | 0.6209 |
| long-tail prevalence calibration | $\tau=2.50$ | 5.50 | 180 | 0.6211 | 60 | 0.6211 |
| long-tail prevalence calibration | $\tau=2.50$ | 6.00 | 180 | 0.6207 | 60 | 0.6207 |
| long-tail prevalence calibration | $\tau=3.00$ | 4.00 | 180 | 0.6180 | 60 | 0.6180 |
| long-tail prevalence calibration | $\tau=3.00$ | 4.50 | 180 | 0.6193 | 60 | 0.6193 |
| long-tail prevalence calibration | $\tau=3.00$ | 5.00 | 180 | 0.6204 | 60 | 0.6204 |
| long-tail prevalence calibration | $\tau=3.00$ | 5.50 | 180 | 0.6215 | 60 | 0.6215 |
| long-tail prevalence calibration | $\tau=3.00$ | 6.00 | 180 | 0.6220 | 60 | 0.6220 |
| long-tail prevalence calibration | $\tau=3.50$ | 4.00 | 180 | 0.6181 | 60 | 0.6181 |
| long-tail prevalence calibration | $\tau=3.50$ | 4.50 | 180 | 0.6193 | 60 | 0.6193 |
| long-tail prevalence calibration | $\tau=3.50$ | 5.00 | 180 | 0.6204 | 60 | 0.6204 |
| long-tail prevalence calibration | $\tau=3.50$ | 5.50 | 180 | 0.6215 | 60 | 0.6215 |
| long-tail prevalence calibration | $\tau=3.50$ | 6.00 | 180 | 0.6220 | 60 | 0.6220 |
