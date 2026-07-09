# LAMP-Merge Full-Scope Hyperparameter Sensitivity

Experiment root: `/data2/liyapeng_grp/program/MedMNISTMerge/outputs/lamp_merge_hparam_full_20260708_overnight_hparam_tau5_client_stats`.

The sensitivity analysis is evaluated on the full medical-image grid: five datasets, four backbones, three client counts, and three beta values. Each value therefore has 180 raw cells and 60 client-average cells when complete.

## Overall

| Module | Parameter | Symbol | Value | Raw cells | Raw mean Acc | Client-average cells | Client-average mean Acc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | prototype head scale | s | 5.00 | 180 | 0.6042 | 60 | 0.6042 |
| M1 | prototype head scale | s | 7.00 | 180 | 0.6131 | 60 | 0.6131 |
| M1 | prototype head scale | s | 10.00 | 180 | 0.6190 | 60 | 0.6190 |
| M1 | prototype head scale | s | 12.00 | 180 | 0.6204 | 60 | 0.6204 |
| M1 | prototype head scale | s | 15.00 | 180 | 0.6206 | 60 | 0.6206 |
| M1 | prototype head scale | s | 17.00 | 180 | 0.6209 | 60 | 0.6209 |
| M1 | prototype head scale | s | 20.00 | 180 | 0.6209 | 60 | 0.6209 |
| M1 | prototype head scale | s | 22.00 | 180 | 0.6205 | 60 | 0.6205 |
| M1 | prototype head scale | s | 25.00 | 180 | 0.6197 | 60 | 0.6197 |
| M1 | prototype head scale | s | 27.00 | 180 | 0.6192 | 60 | 0.6192 |
| M1 | prototype head scale | s | 30.00 | 180 | 0.6181 | 60 | 0.6181 |
| M1 | prototype head scale | s | 32.00 | 180 | 0.6175 | 60 | 0.6175 |
| M1 | prototype head scale | s | 35.00 | 180 | 0.6168 | 60 | 0.6168 |
| M1 | prototype head scale | s | 37.00 | 180 | 0.6161 | 60 | 0.6161 |
| M1 | prototype head scale | s | 40.00 | 180 | 0.6154 | 60 | 0.6154 |
| M2 | long-tail calibration strength | lambda | 2.00 | 180 | 0.6131 | 60 | 0.6131 |
| M2 | long-tail calibration strength | lambda | 3.00 | 180 | 0.6173 | 60 | 0.6173 |
| M2 | long-tail calibration strength | lambda | 4.00 | 180 | 0.6198 | 60 | 0.6198 |
| M2 | long-tail calibration strength | lambda | 5.00 | 180 | 0.6209 | 60 | 0.6209 |
| M2 | long-tail calibration strength | lambda | 6.00 | 180 | 0.6207 | 60 | 0.6207 |
| M2 | long-tail calibration strength | lambda | 7.00 | 180 | 0.6206 | 60 | 0.6206 |
| M2 | long-tail calibration strength | lambda | 8.00 | 180 | 0.6203 | 60 | 0.6203 |
| M2 | long-tail calibration strength | lambda | 10.00 | 180 | 0.6190 | 60 | 0.6190 |

## Dataset-Level Client Average

| Module | Symbol | Value | Dataset | Cells | Mean Acc |
| --- | --- | --- | --- | --- | --- |
| M1 | s | 5.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 5.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M1 | s | 5.00 | dermamnist_224 | 12 | 0.6772 |
| M1 | s | 5.00 | organcmnist_224 | 12 | 0.5512 |
| M1 | s | 5.00 | organsmnist_224 | 12 | 0.5179 |
| M1 | s | 7.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 7.00 | chaoshengmnist_224 | 12 | 0.4576 |
| M1 | s | 7.00 | dermamnist_224 | 12 | 0.6706 |
| M1 | s | 7.00 | organcmnist_224 | 12 | 0.5827 |
| M1 | s | 7.00 | organsmnist_224 | 12 | 0.5374 |
| M1 | s | 10.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 10.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M1 | s | 10.00 | dermamnist_224 | 12 | 0.6615 |
| M1 | s | 10.00 | organcmnist_224 | 12 | 0.6050 |
| M1 | s | 10.00 | organsmnist_224 | 12 | 0.5539 |
| M1 | s | 12.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 12.00 | chaoshengmnist_224 | 12 | 0.4583 |
| M1 | s | 12.00 | dermamnist_224 | 12 | 0.6533 |
| M1 | s | 12.00 | organcmnist_224 | 12 | 0.6130 |
| M1 | s | 12.00 | organsmnist_224 | 12 | 0.5598 |
| M1 | s | 15.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 15.00 | chaoshengmnist_224 | 12 | 0.4576 |
| M1 | s | 15.00 | dermamnist_224 | 12 | 0.6417 |
| M1 | s | 15.00 | organcmnist_224 | 12 | 0.6206 |
| M1 | s | 15.00 | organsmnist_224 | 12 | 0.5655 |
| M1 | s | 17.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 17.00 | chaoshengmnist_224 | 12 | 0.4579 |
| M1 | s | 17.00 | dermamnist_224 | 12 | 0.6357 |
| M1 | s | 17.00 | organcmnist_224 | 12 | 0.6236 |
| M1 | s | 17.00 | organsmnist_224 | 12 | 0.5699 |
| M1 | s | 20.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 20.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M1 | s | 20.00 | dermamnist_224 | 12 | 0.6286 |
| M1 | s | 20.00 | organcmnist_224 | 12 | 0.6270 |
| M1 | s | 20.00 | organsmnist_224 | 12 | 0.5741 |
| M1 | s | 22.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 22.00 | chaoshengmnist_224 | 12 | 0.4575 |
| M1 | s | 22.00 | dermamnist_224 | 12 | 0.6231 |
| M1 | s | 22.00 | organcmnist_224 | 12 | 0.6280 |
| M1 | s | 22.00 | organsmnist_224 | 12 | 0.5764 |
| M1 | s | 25.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 25.00 | chaoshengmnist_224 | 12 | 0.4577 |
| M1 | s | 25.00 | dermamnist_224 | 12 | 0.6155 |
| M1 | s | 25.00 | organcmnist_224 | 12 | 0.6291 |
| M1 | s | 25.00 | organsmnist_224 | 12 | 0.5786 |
| M1 | s | 27.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 27.00 | chaoshengmnist_224 | 12 | 0.4582 |
| M1 | s | 27.00 | dermamnist_224 | 12 | 0.6110 |
| M1 | s | 27.00 | organcmnist_224 | 12 | 0.6298 |
| M1 | s | 27.00 | organsmnist_224 | 12 | 0.5794 |
| M1 | s | 30.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 30.00 | chaoshengmnist_224 | 12 | 0.4576 |
| M1 | s | 30.00 | dermamnist_224 | 12 | 0.6044 |
| M1 | s | 30.00 | organcmnist_224 | 12 | 0.6306 |
| M1 | s | 30.00 | organsmnist_224 | 12 | 0.5805 |
| M1 | s | 32.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 32.00 | chaoshengmnist_224 | 12 | 0.4578 |
| M1 | s | 32.00 | dermamnist_224 | 12 | 0.6007 |
| M1 | s | 32.00 | organcmnist_224 | 12 | 0.6305 |
| M1 | s | 32.00 | organsmnist_224 | 12 | 0.5808 |
| M1 | s | 35.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 35.00 | chaoshengmnist_224 | 12 | 0.4580 |
| M1 | s | 35.00 | dermamnist_224 | 12 | 0.5960 |
| M1 | s | 35.00 | organcmnist_224 | 12 | 0.6311 |
| M1 | s | 35.00 | organsmnist_224 | 12 | 0.5813 |
| M1 | s | 37.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 37.00 | chaoshengmnist_224 | 12 | 0.4578 |
| M1 | s | 37.00 | dermamnist_224 | 12 | 0.5929 |
| M1 | s | 37.00 | organcmnist_224 | 12 | 0.6311 |
| M1 | s | 37.00 | organsmnist_224 | 12 | 0.5814 |
| M1 | s | 40.00 | bloodmnist_224 | 12 | 0.8174 |
| M1 | s | 40.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M1 | s | 40.00 | dermamnist_224 | 12 | 0.5889 |
| M1 | s | 40.00 | organcmnist_224 | 12 | 0.6317 |
| M1 | s | 40.00 | organsmnist_224 | 12 | 0.5814 |
| M2 | lambda | 2.00 | bloodmnist_224 | 12 | 0.8174 |
| M2 | lambda | 2.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M2 | lambda | 2.00 | dermamnist_224 | 12 | 0.5782 |
| M2 | lambda | 2.00 | organcmnist_224 | 12 | 0.6312 |
| M2 | lambda | 2.00 | organsmnist_224 | 12 | 0.5813 |
| M2 | lambda | 3.00 | bloodmnist_224 | 12 | 0.8174 |
| M2 | lambda | 3.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M2 | lambda | 3.00 | dermamnist_224 | 12 | 0.5991 |
| M2 | lambda | 3.00 | organcmnist_224 | 12 | 0.6310 |
| M2 | lambda | 3.00 | organsmnist_224 | 12 | 0.5813 |
| M2 | lambda | 4.00 | bloodmnist_224 | 12 | 0.8174 |
| M2 | lambda | 4.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M2 | lambda | 4.00 | dermamnist_224 | 12 | 0.6168 |
| M2 | lambda | 4.00 | organcmnist_224 | 12 | 0.6291 |
| M2 | lambda | 4.00 | organsmnist_224 | 12 | 0.5782 |
| M2 | lambda | 5.00 | bloodmnist_224 | 12 | 0.8174 |
| M2 | lambda | 5.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M2 | lambda | 5.00 | dermamnist_224 | 12 | 0.6286 |
| M2 | lambda | 5.00 | organcmnist_224 | 12 | 0.6270 |
| M2 | lambda | 5.00 | organsmnist_224 | 12 | 0.5741 |
| M2 | lambda | 6.00 | bloodmnist_224 | 12 | 0.8174 |
| M2 | lambda | 6.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M2 | lambda | 6.00 | dermamnist_224 | 12 | 0.6366 |
| M2 | lambda | 6.00 | organcmnist_224 | 12 | 0.6230 |
| M2 | lambda | 6.00 | organsmnist_224 | 12 | 0.5692 |
| M2 | lambda | 7.00 | bloodmnist_224 | 12 | 0.8174 |
| M2 | lambda | 7.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M2 | lambda | 7.00 | dermamnist_224 | 12 | 0.6452 |
| M2 | lambda | 7.00 | organcmnist_224 | 12 | 0.6190 |
| M2 | lambda | 7.00 | organsmnist_224 | 12 | 0.5642 |
| M2 | lambda | 8.00 | bloodmnist_224 | 12 | 0.8174 |
| M2 | lambda | 8.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M2 | lambda | 8.00 | dermamnist_224 | 12 | 0.6516 |
| M2 | lambda | 8.00 | organcmnist_224 | 12 | 0.6146 |
| M2 | lambda | 8.00 | organsmnist_224 | 12 | 0.5605 |
| M2 | lambda | 10.00 | bloodmnist_224 | 12 | 0.8174 |
| M2 | lambda | 10.00 | chaoshengmnist_224 | 12 | 0.4574 |
| M2 | lambda | 10.00 | dermamnist_224 | 12 | 0.6615 |
| M2 | lambda | 10.00 | organcmnist_224 | 12 | 0.6050 |
| M2 | lambda | 10.00 | organsmnist_224 | 12 | 0.5539 |
