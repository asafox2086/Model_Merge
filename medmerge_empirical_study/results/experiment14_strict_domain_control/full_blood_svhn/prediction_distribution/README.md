# Separate AVG Prediction Distributions

- [Medical BloodMNIST bar chart](blood_avg_prediction_distribution.png)
- [Natural SVHN bar chart](svhn_avg_prediction_distribution.png)
- [BloodMNIST counts and percentages](blood_avg_prediction_distribution.csv)
- [SVHN counts and percentages](svhn_avg_prediction_distribution.csv)

The chart uses the AVG model's full 3,421-image test prediction distribution in each domain. The two domains have the same eight class supports, client label partitions, client sample counts, ResNet configuration, and uniform checkpoint averaging.

BloodMNIST predicts Class 3 for 3,254 images (95.1%) and Class 0 for 167 (4.9%). SVHN predicts Class 6 for 2,586 images (75.6%), Class 5 for 802 (23.4%), Class 0 for 24 (0.7%), and Class 3 for 9 (0.3%). These plots measure prediction collapse, not feature collapse.
