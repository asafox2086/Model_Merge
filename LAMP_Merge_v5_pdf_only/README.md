# LAMP-Merge v5 PDF-only figure package

This package is built from the uploaded `LAMP_Merge.zip` v5 source.

- `v5.tex` and `appendix.tex` have been mechanically changed so `figures/*.png` references become `figures/*.pdf`.
- Every PNG figure from the uploaded package was converted to a real PDF using Pillow. No file was renamed by changing the suffix.
- Existing PDF-only figures, such as `intro.pdf` and `method.pdf`, were copied as PDF.
- The visual content of PNG-based figures is preserved by converting from the uploaded PNG itself, including figures intentionally kept in older ACC-only form.

## Converted from uploaded PNG

| Source PNG | Output PDF |
| --- | --- |
| `figures/01_module_ablation_accuracy.png` | `figures/01_module_ablation_accuracy.pdf` |
| `figures/02_internal_module_ablations.png` | `figures/02_internal_module_ablations.pdf` |
| `figures/03_baseline_2x2_ablation.png` | `figures/03_baseline_2x2_ablation.pdf` |
| `figures/04_ultrasound_class_distribution.png` | `figures/04_ultrasound_class_distribution.pdf` |
| `figures/05_diagnostic_reconstruction_hparams.png` | `figures/05_diagnostic_reconstruction_hparams.pdf` |
| `figures/05_hyperparameter_sensitivity.png` | `figures/05_hyperparameter_sensitivity.pdf` |
| `figures/06_prevalence_calibration_hparams.png` | `figures/06_prevalence_calibration_hparams.pdf` |
| `figures/07_tsne_output_probability.png` | `figures/07_tsne_output_probability.pdf` |
| `figures/08_prototype_geometry.png` | `figures/08_prototype_geometry.pdf` |
| `figures/appendix_sample_blood_0.png` | `figures/appendix_sample_blood_0.pdf` |
| `figures/appendix_sample_blood_1.png` | `figures/appendix_sample_blood_1.pdf` |
| `figures/appendix_sample_blood_2.png` | `figures/appendix_sample_blood_2.pdf` |
| `figures/appendix_sample_blood_3.png` | `figures/appendix_sample_blood_3.pdf` |
| `figures/appendix_sample_blood_4.png` | `figures/appendix_sample_blood_4.pdf` |
| `figures/appendix_sample_blood_5.png` | `figures/appendix_sample_blood_5.pdf` |
| `figures/appendix_sample_blood_6.png` | `figures/appendix_sample_blood_6.pdf` |
| `figures/appendix_sample_blood_7.png` | `figures/appendix_sample_blood_7.pdf` |
| `figures/appendix_sample_derma_0.png` | `figures/appendix_sample_derma_0.pdf` |
| `figures/appendix_sample_derma_1.png` | `figures/appendix_sample_derma_1.pdf` |
| `figures/appendix_sample_derma_2.png` | `figures/appendix_sample_derma_2.pdf` |
| `figures/appendix_sample_derma_3.png` | `figures/appendix_sample_derma_3.pdf` |
| `figures/appendix_sample_derma_4.png` | `figures/appendix_sample_derma_4.pdf` |
| `figures/appendix_sample_derma_5.png` | `figures/appendix_sample_derma_5.pdf` |
| `figures/appendix_sample_derma_6.png` | `figures/appendix_sample_derma_6.pdf` |
| `figures/appendix_sample_organ_c_0.png` | `figures/appendix_sample_organ_c_0.pdf` |
| `figures/appendix_sample_organ_c_1.png` | `figures/appendix_sample_organ_c_1.pdf` |
| `figures/appendix_sample_organ_c_10.png` | `figures/appendix_sample_organ_c_10.pdf` |
| `figures/appendix_sample_organ_c_2.png` | `figures/appendix_sample_organ_c_2.pdf` |
| `figures/appendix_sample_organ_c_3.png` | `figures/appendix_sample_organ_c_3.pdf` |
| `figures/appendix_sample_organ_c_4.png` | `figures/appendix_sample_organ_c_4.pdf` |
| `figures/appendix_sample_organ_c_5.png` | `figures/appendix_sample_organ_c_5.pdf` |
| `figures/appendix_sample_organ_c_6.png` | `figures/appendix_sample_organ_c_6.pdf` |
| `figures/appendix_sample_organ_c_7.png` | `figures/appendix_sample_organ_c_7.pdf` |
| `figures/appendix_sample_organ_c_8.png` | `figures/appendix_sample_organ_c_8.pdf` |
| `figures/appendix_sample_organ_c_9.png` | `figures/appendix_sample_organ_c_9.pdf` |
| `figures/appendix_sample_organ_s_0.png` | `figures/appendix_sample_organ_s_0.pdf` |
| `figures/appendix_sample_organ_s_1.png` | `figures/appendix_sample_organ_s_1.pdf` |
| `figures/appendix_sample_organ_s_10.png` | `figures/appendix_sample_organ_s_10.pdf` |
| `figures/appendix_sample_organ_s_2.png` | `figures/appendix_sample_organ_s_2.pdf` |
| `figures/appendix_sample_organ_s_3.png` | `figures/appendix_sample_organ_s_3.pdf` |
| `figures/appendix_sample_organ_s_4.png` | `figures/appendix_sample_organ_s_4.pdf` |
| `figures/appendix_sample_organ_s_5.png` | `figures/appendix_sample_organ_s_5.pdf` |
| `figures/appendix_sample_organ_s_6.png` | `figures/appendix_sample_organ_s_6.pdf` |
| `figures/appendix_sample_organ_s_7.png` | `figures/appendix_sample_organ_s_7.pdf` |
| `figures/appendix_sample_organ_s_8.png` | `figures/appendix_sample_organ_s_8.pdf` |
| `figures/appendix_sample_organ_s_9.png` | `figures/appendix_sample_organ_s_9.pdf` |
| `figures/appendix_sample_ultrasound_0.png` | `figures/appendix_sample_ultrasound_0.pdf` |
| `figures/appendix_sample_ultrasound_1.png` | `figures/appendix_sample_ultrasound_1.pdf` |
| `figures/appendix_sample_ultrasound_2.png` | `figures/appendix_sample_ultrasound_2.pdf` |
| `figures/appendix_sample_ultrasound_3.png` | `figures/appendix_sample_ultrasound_3.pdf` |
| `figures/appendix_sample_ultrasound_4.png` | `figures/appendix_sample_ultrasound_4.pdf` |
| `figures/appendix_sample_ultrasound_5.png` | `figures/appendix_sample_ultrasound_5.pdf` |
| `figures/appendix_sample_ultrasound_6.png` | `figures/appendix_sample_ultrasound_6.pdf` |
| `figures/appendix_sample_ultrasound_7.png` | `figures/appendix_sample_ultrasound_7.pdf` |
| `figures/compare_nl_me.png` | `figures/compare_nl_me.pdf` |
| `figures/dataset_examples.png` | `figures/dataset_examples.pdf` |
| `figures/sample_blood.png` | `figures/sample_blood.pdf` |
| `figures/sample_derma.png` | `figures/sample_derma.pdf` |
| `figures/sample_organ_c.png` | `figures/sample_organ_c.pdf` |
| `figures/sample_organ_s.png` | `figures/sample_organ_s.pdf` |
| `figures/sample_ultrasound.png` | `figures/sample_ultrasound.pdf` |
| `figures/vit_tiny_radar.png` | `figures/vit_tiny_radar.pdf` |

## Copied existing PDF

| Source PDF | Output PDF |
| --- | --- |
| `figures/intro.pdf` | `figures/intro.pdf` |
| `figures/method.pdf` | `figures/method.pdf` |
