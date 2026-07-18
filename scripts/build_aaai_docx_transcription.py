#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "My_merge_ret" / "aaai_docx_transcription_work"
BLOCKS_JSON = WORK / "blocks.json"
MEDIA_DIR = WORK / "media"
OUT = ROOT / "My_merge_ret" / "aaai_lamp_merge_docx_tex"


TRANSLATIONS = {
    1: "AAAI Paper Outline",
    2: "We formulate two research questions:",
    3: "Question 1: In multi-center medical settings with partial local class coverage, post-hoc parameter merging can aggregate several non-degenerate client predictors into a globally collapsed model whose prediction distribution is highly concentrated.",
    4: "Question 2: Under long-tailed medical class priors, majority-class predictive collapse can produce an apparent gain in Accuracy, and the attainable accuracy is directly determined by the highest-frequency class prior.",
    5: "These questions lead to two empirical observations:",
    6: "Observation 1: Aggregation-induced Predictive Collapse.",
    7: "To characterize this phenomenon, we first measure the predicted class distribution on the test set:",
    9: "Here, $N$ denotes the number of test images, $c$ denotes a diagnostic class, and $q(c)$ denotes the fraction of test images predicted as class $c$. We further define the prediction-collapse strength as:",
    11: "$\\rho$ is the largest class proportion in the predicted distribution; the closer $\\rho$ is to 1, the closer the model is to a single-class predictor.",
    12: "Specifically, in multi-center medical image tasks, each client usually observes only a subset of the global diagnostic space, so local models inevitably carry class bias. However, this bias does not mean that the client models have already degenerated into single-class predictors: before merging, local models can still respond effectively to multiple diagnostic classes. The degeneration mainly occurs during post-hoc merging. Existing parameter-level merging methods operate on whole models through averaging, pruning, or sign aggregation, and lack an explicit description of which diagnostic classes each client can discriminate reliably. As a result, originally local and controllable class bias is superposed and amplified in the global parameter space, eventually forming single-class or few-class predictive collapse in the merged model.",
    13: "For example, on OrgansMNIST-224 with ResNet, 3 clients, and $\\beta=0.01$, the three clients have $\\rho$ values of 0.7683, 0.3161, and 0.3686, respectively. This indicates that local models have different degrees of class bias but have not all collapsed to a single output class. Under the same setting, the Breadcrumbs merged model reaches $\\rho=0.9738$, with predictions almost concentrated on one diagnostic class. This result shows that predictive collapse is not simply inherited from already failed client models, but is further induced by a class-agnostic merging process.",
    14: "This observation directly motivates M1: post-hoc merging should not only mix classifier heads in parameter space, but should explicitly preserve the class-level diagnostic knowledge that remains effective on clients. Therefore, M1 requires clients to upload feature prototypes for each diagnostic class, using diagnostic prototypes as structured carriers of local discriminative ability.",
    15: "Observation 2: Collapse Benefit under Long-tailed Priors.",
    16: "Medical image classification usually inherits real clinical prevalence or sampling prevalence, so common diseases, common organs, or high-frequency diagnostic classes may occupy a large proportion of the test set. Let the highest-frequency class proportion in the test set be:",
    18: "If a model completely collapses to this highest-frequency class, then it has no real multi-class diagnostic ability but can still obtain the following accuracy:",
    20: "Therefore, when the highest-frequency class proportion is large, a single-class predictor can obtain high accuracy without multi-class diagnostic ability. In other words, on long-tailed medical data, Accuracy may misinterpret majority-class collapse as effective merging. This phenomenon explains why some baselines can still obtain high Accuracy despite having highly concentrated prediction distributions.",
    21: "This observation motivates M2: eliminating collapse is not equivalent to forcing the prediction distribution to become uniform, because long-tailed priors in medical data have real statistical meaning; at the same time, the model must not degenerate into a majority-class predictor. M2 therefore does not change the diagnostic prototypes reconstructed by M1. Instead, when prevalence counts uploaded by clients indicate a strong dominant class, M2 adds a bounded centered log-prior to the classification scores. This preserves the independent discriminative direction of each diagnostic class while allowing real high-frequency classes to receive necessary Accuracy calibration.",
    22: "Related Work",
    23: "Model Merging",
    24: "General post-hoc model merging aims to integrate already trained checkpoints, thereby avoiding the cost of joint training or retraining. Existing methods can be categorized according to the variable being merged. The first class directly interpolates in weight space, such as weight averaging and Model Soups; these methods can perform well when multiple models remain within compatible parameter basins. The second class operates in parameter-delta space, for example Task Arithmetic represents fine-tuned models as parameter displacements relative to a shared initialization and linearly combines these displacements. The third class attempts to remove parameter interference before averaging; for example, Git Re-Basin reduces permutation mismatch through neuron permutation alignment, while TIES-Merging and the DARE family alleviate update conflicts through pruning small updates, sign voting, or randomly dropping and rescaling task vectors. The fourth class further introduces geometric, curvature, or robust statistical assumptions, such as Fisher merging, RegMean, Model Stock, Breadcrumbs, Iso-C, FreeMerge, and RobustMerge.",
    25: "These methods form important baselines in this paper, but their merging variables do not model medical diagnostic classes. They operate on global parameters, task vectors, curvature estimates, permutation structures, or update signs, rather than on whether client $i$ has reliable evidence for diagnostic class $c$. In multi-center medical image tasks, client training sets usually contain both missing classes and long-tailed distributions. When many client-class pairs satisfy $n_{i,c}=0$, class-agnostic parameter merging mixes a reliable class direction from one client with missing or conflicting directions from other clients. Consequently, the merged model may retain majority-class directions while suppressing rare or locally missing classes, leading to a globally collapsed model with a highly concentrated prediction distribution.",
    26: "Therefore, the key reason why general merging methods fail in our setting is not that their parameter-merging techniques are intrinsically weak, but that their merging variables are mismatched with the core structure of multi-center medical data. Medical image merging needs to explicitly represent class-level diagnostic availability, namely that each client has different evidence strength for different diagnostic classes. M1 in LAMP-Merge is designed precisely for this gap: it does not directly average classifier heads, but lets clients upload class prototypes and class support counts, and reconstructs a global diagnostic prototype classifier class by class on the server.",
    27: "Medical Model Merging",
    28: "Medical fusion methods usually exploit structures specific to clinical data. For example, mTAN uses continuous-time attention for irregularly sampled time series, Raindrop uses graph structures to model relationships among physiological sensors, and MedFuse fuses chest X-ray images with clinical time series. These works show that medical data are not merely application scenarios for general fusion algorithms; medical data have irregular temporal structure, modality heterogeneity, clinical measurement mechanisms, and disease-prevalence distributions.",
    29: "However, these medical fusion methods usually fuse patient-level input modalities or clinical representations, rather than independently trained medical image classifier checkpoints. Their training processes often require access to patient-level samples, timestamps, multimodal records, EHR, or server-side validation feedback. When the server cannot view clients' raw medical images, per-sample features, per-sample logits, or per-sample predictions, these methods cannot be directly applied to the one-shot model-merging setting studied in this paper.",
    30: "Model-merging algorithms that do not leak raw images, such as MedMerge (https://ieeexplore.ieee.org/abstract/document/11382027/), then ...",
    31: "LAMP-Merge confines medical structure to uploadable class-level statistics: clients locally compute diagnostic class prototypes and prevalence counts, and the server constructs the model only from these aggregate statistics. Therefore, it does not need access to the source images, preserving patient privacy.",
    32: "Federated Learning",
    33: "Federated learning and decentralized learning are important directions in privacy-sensitive medical AI. FedAvg and its variants train a global model through multiple communication rounds: the server sends a model, clients perform local optimization, then upload updates for server aggregation. Swarm Learning removes the traditional central parameter server and uses decentralized coordination for multi-party training. Gossip Learning further propagates models asynchronously in a peer-to-peer manner, allowing participants to gradually obtain collaborative benefits through local training and model exchange. These methods are suitable when clients can remain online for a long time, communicate repeatedly, and continue training from intermediate global models. However, the problem studied in this paper is not collaborative training, but single-shot asynchronous merging after training is complete. In realistic medical deployment, hospitals may finish training at different times and may not be able to afford repeated model distribution, training, and upload protocols. Therefore, in our setting, the server does not send a sequence of global models to clients, and clients do not continue optimization after observing the server's merged result. Thus, federated learning and decentralized learning solve multi-round training-protocol problems, while LAMP-Merge solves the one-shot post-hoc merging problem. LAMP-Merge only requires each client to upload its model and class-level statistics once after local training, and the server generates a unified global diagnostic model without establishing a multi-round feedback loop.",
    34: "Method Design",
    35: "Module 1: Prototype Reconstruction Module",
    36: "To address the problem that post-hoc parameter merging can degenerate into a globally collapsed model with a highly concentrated prediction distribution in medical model merging, we propose the prototype reconstruction module, which aims to mitigate collapse by redefining weights according to the data distribution.",
    37: "LAMP-Merge first reconstructs a class-wise diagnostic head in the shared reference feature space. Let the fixed reference backbone be instantiated from the shared model configuration before local training, and let $T(\\cdot)$ be the evaluation transform.",
    38: "For each diagnostic class $c$, client $i$ computes the corresponding",
    39: "reference-space class prototype.",
    41: "Here, the term denotes the support size of this class. If $n_{i,c}=0$, then the client uploads no support data for this class and provides no prototype evidence.",
    42: "The server converts support counts into evidence weights:",
    43: "where",
    44: "is the default evidence exponent. The normalized reliability of client $i$ for class $c$ is",
    46: "Then the diagnostic prototype is constructed as",
    48: "Finally, LAMP-Merge synthesizes a cosine prototype classifier:",
    50: "where $s=18.75$ is the default head scale.",
    51: "This module does not average incompatible classifier heads. Instead, it preserves the classes for which each client has evidence support and reconstructs a global multi-class diagnostic head.",
    52: "Module 2: Long-tail Prevalence Calibration Module",
    53: "Module 2 should correspond to Question 2.",
    54: "Reference: To address the collapse benefit caused by long-tailed distributions in medical model merging, we propose the long-tail prevalence calibration module, which aims to mitigate this issue by adding class-prior knowledge.",
    55: "Each client also uploads the prevalence",
    56: "count value for each class. The server then estimates the global class prior distribution.",
    58: "We measure the imbalance of the dominant class as follows:",
    59: "where $C$ is the total number of classes.",
    60: "M2 is activated only when $r>\\tau$ (default $\\tau=2.5$). Once activated, it adds a centered log-prior bias term.",
    62: "The default maximum calibration strength is $\\lambda=4.25$. The final score is",
    64: "Experimental Design",
    65: "We evaluate LAMP-Merge in a single-shot asynchronous medical model-merging scenario. Each client trains locally on its private and class-biased data partition. After local training is complete, each client uploads one checkpoint and the class-level statistics required by LAMP-Merge. The server performs only one merging operation and does not run an iterative server-client optimization process.",
    66: "/*",
    67: "Medical images",
    68: "+ multiple trained client checkpoints",
    69: "+ the server does not view raw data / validation data / proxy data",
    70: "+ candidate models are not sent back to clients for scoring",
    71: "+ single-shot post-hoc model merging",
    72: "Clients compute statistics locally, and the server only merges statistics and checkpoints. The design emphasis is placed on the client side, using minimal privacy exposure to obtain the best possible performance.",
    73: "Reason: within the limited data of each client, it is difficult to find a method that effectively combines the information provided by these clients. Whether using incremental fusion, BN estimation, or other methods, privacy protection leads to insufficient data, insufficient evaluation coverage, and final performance that can even be worse than avg.",
    74: "*/",
    75: "Experimental Design Matrix:",
    76: "This paper focuses on the medical model-merging scenario.",
    77: "Basic Training Settings",
    78: "Datasets: We use five medical image classification datasets",
    79: "with resolution 224. All tasks are single-label multi-class diagnosis or organ-recognition tasks.",
    80: "We evaluate four vision backbones: ResNet, ConvNeXt, ViT-Tiny, and Swin-Tiny. CLIP-ViT-B/32 and other vision-language backbones are outside the experimental scope of this paper. For each dataset-backbone combination, we construct non-IID client partitions with the number of clients $K\\in\\{3,5,7\\}$ and $\\beta\\in\\{0,0.01,0.1\\}$. All experiments use seed 42. This benchmark contains 180 raw test-accuracy cells and 60 client-average cells; the client-average metric averages the three $\\beta$ values under a fixed client number, reducing sensitivity to a single partition skewness.",
    81: "Backbone Settings",
    82: "The shared reference feature space is the feature coordinate system produced by the shared reference backbone $\\phi_0$ and preprocessing transform $T$.",
    83: "We compare against twelve post-hoc merging baselines: avg, ties, dare\\_linear, dare\\_ties, regmean, fisher, breadcrumb, model\\_stock, from, iso\\_c, free\\_merge, and robustmerge. The main comparison criterion is that a cell is counted as successful if the LAMP-Merge value is greater than or equal to the strongest non-LAMP baseline in that cell.",
    84: "All experiments were run on a server with two Intel(R) Xeon(R) Silver 4210 CPUs @ 2.20GHz, 40 logical CPU threads in total, and four NVIDIA GeForce RTX 2080 Ti GPUs with 11264 MiB memory each. The NVIDIA driver version is 535.154.05. The operating system is Ubuntu 20.04.1 LTS. The software environment is the conda environment \\texttt{MM}, using Python 3.10.20, PyTorch 2.6.0+\\texttt{cu118}, torchvision 0.21.0+\\texttt{cu118}, NumPy 2.2.6, and CUDA runtime 11.8.",
    85: "Experimental Result Tables",
    86: "Small",
    87: "ResNet",
    88: "Client Average",
    90: "ConvNeXt",
    92: "ViT-Tiny",
    93: "Client Average",
    95: "Swin-Tiny",
    96: "Client Average",
    98: "Ablation Study",
    99: "Internal Ablation",
    100: "Prototype Information:",
    103: "Class Statistical Information",
    105: "Note: No prevalence calibration should be deleted later; this is an inter-module ablation.",
    108: "Diagnostic prototype alternatives are all significantly lower than the formal method, showing that the performance gain cannot be explained by an additional classifier head, random directions, or class counts alone. No prevalence calibration and Uniform prevalence prior both remove the effective long-tail prior term and match M1 only on the full grid. The additive smoothed prior differs from the formal method by only about $6.1\\times10^{-5}$ in mean Accuracy, showing that M2 is stable to mild count perturbations. In contrast, avg+M2 obtains only 0.2918 mean Accuracy and a collapse ratio of 0.9551, demonstrating that prevalence calibration cannot replace diagnostic prototype reconstruction.",
    109: "Inter-module Ablation",
    110: "The inter-module ablation follows the formal main-table protocol, covering 5 datasets, 4 vision backbones, 3 client counts, and 3 Dirichlet $\\beta$ values, for a total of 180 raw cells and 60 client-average cells. CLIP-ViT-B/32 is excluded. This table directly compares complete LAMP-Merge, M1 only with diagnostic prototype reconstruction retained, avg+M2 which attaches M2 to ordinary averaging, and ordinary parameter averaging avg.",
    112: "The dataset-level inter-module ablation is as follows:",
    114: "Hyperparameter Analysis",
    115: "The hyperparameter analysis summarizes the value range, best point, worst point, and fluctuation amplitude of each internal factor, showing that the formal setting does not rely on an accidental single-point gain.",
    116: "Full-scope results show that LAMP-Merge has a stable plateau for both hyperparameters, rather than relying on single-point tuning. For M1, $s=20$ obtains the highest client-average mean Accuracy of 0.6209; within $s\\in[12,22]$, the mean Accuracy remains between 0.6204 and 0.6209, indicating that the diagnostic prototype head only needs a moderate logit scale to work stably. For M2, $\\lambda=5$ obtains the highest client-average mean Accuracy of 0.6209; within $\\lambda\\in[4,8]$, the mean Accuracy remains between 0.6198 and 0.6209, indicating that the benefit of long-tail calibration comes from a bounded medical prevalence prior rather than unboundedly following the majority class.",
    117: "Additional Metric Diagnostic Experiment",
    118: "To rule out the possibility that LAMP-Merge only exploits majority-class Accuracy, we additionally construct a full-scope prediction-distribution diagnostic experiment. This experiment covers 5 formal medical datasets, 4 backbones (resnet, convnext, vit\\_t, and swin\\_tiny), $K\\in\\{3,5,7\\}$, and $\\beta\\in\\{0,0.01,0.1\\}$ with seed 42, giving 180 cases per non-client method. The measured objects include aggregate rows for individual clients, general model-merging baselines, and LAMP-Merge.",
    119: "In addition to Accuracy, this experiment reports the following new metrics:",
    121: "Here, BA is balanced accuracy, i.e., macro recall, used to measure whether all diagnostic classes are recalled. Let $q(c)$ denote the proportion of predictions assigned to class $c$ on the test set, and let $p(c)$ denote the true class proportion in the test set. Then prediction-collapse strength and prediction-distribution deviation are defined as:",
    123: "The closer $\\rho$ is to 1, the closer the model is to a single-class predictor; the smaller $TV(q,p)$ is, the closer the predicted class distribution is to the true diagnostic distribution. The effective number of predicted classes is defined as:",
    125: "A larger value indicates that the model actually uses more diagnostic classes. If a method only collapses to the majority class, it usually shows high Accuracy but low BA, Macro F1, and $C_{\\mathrm{eff}}$, while $\\rho$ and TV are high.",
    128: "M1-only has higher BA, Macro-F1, and $C_{\\mathrm{eff}}$, indicating that diagnostic prototypes directly restore multi-class discrimination. LAMP-Merge has higher overall Accuracy and lower Pred-True TV, indicating that M2 calibrates the prediction distribution to the real medical prevalence. These are not contradictory: M1 optimizes class-balanced discrimination, while M2 optimizes overall risk consistent with the true test distribution under strong long-tail conditions.",
    129: "Dataset-level results:",
    131: "best generic denotes the strongest fixed general merging baseline selected by full-scope mean Accuracy on each dataset. It is used only for analysis and does not participate in LAMP-Merge model selection.",
    133: "Prediction Distribution Visualization",
    134: "The following figure aggregates predicted class proportions over 36 full-scope cases on Derma. The true proportion of test-set class 5 is 0.67; the predicted proportion of M1-only is 0.37, while LAMP-Merge reaches 0.68 after M2 calibration. In contrast, avg+M2 predicts class 5 for every sample, and the collapse strength of the strongest general baseline free\\_merge is 0.9522. Thus, M2 provides bounded long-tail calibration only after M1 has restored a multi-class diagnostic representation; it cannot rescue an already collapsed average model.",
    135: "The remaining datasets are as follows:",
    139: "t-SNE",
    140: "t-SNE is used only for paper analysis and does not participate in server-side merging or model selection. For each (dataset, backbone), the analysis places the test features of that backbone and the class prototypes of 5 methods under all 9 (K,b) settings into the same t-SNE fitting. The five subplots share the same two-dimensional coordinate system, so prototype positions are comparable across methods. The 20 figures cover 5 datasets and 4 backbones.",
    142: "The remaining figures are not included.",
    143: "In-depth Analysis Experiment",
    144: "This experiment uses only the class prototypes, class support counts, and global prototypes constructed from these statistics uploaded by clients. It evaluates whether the formal method restores multi-class diagnostic ability because the prototypes carry stable class semantic directions, or merely because extra parameters or statistics are added.",
    145: "The Accuracy ablation contains 11 settings; this section lists only the 8 settings that change prototype geometry. No prevalence calibration, Uniform prevalence prior, and Smoothed prevalence prior only change the M2 class bias $b_c$ and do not change the global prototype $p_c$, client weight $\\alpha_{i,c}$, or client class prototype $\\mu_{i,c}$. Therefore, their geometric metrics completely overlap with the corresponding prototype construction. Repeating them in the geometry figure would not provide new geometric evidence.",
    148: "This group of results supports the core mechanism of LAMP-Merge: the formal method does not maximize a single geometric metric, but forms a stable combination among diagnostic class separability, cross-client semantic consistency, global-prototype alignment with client evidence, and evidence allocation driven by class support counts. Specifically, LAMP-Merge obtains a mean Pairwise distance of 0.1115 and a mean Nearest-class distance of 0.0355 on the full grid, showing that each diagnostic class has an independent discriminative direction in the shared reference feature space. Meanwhile, Prototype consistency reaches 0.8751 and Prototype-client alignment reaches 0.9419, showing that these class directions preserve same-class semantic consistency across clients and do not deviate from the real client-uploaded class evidence after server aggregation. Evidence entropy is 0.2124, indicating that the formal method does not simply assign equal contributions to all clients, but gives higher weights to more reliable client evidence according to class support counts.",
    149: "Therefore, the fact that some ablation settings exceed LAMP-Merge on individual geometric metrics is not a counterexample. Excessively large Pairwise distance or Nearest-class distance only indicates that directions are far apart, but does not guarantee that these directions correspond to true diagnostic semantics. Prototype consistency or Prototype-client alignment close to 1 may also come from class-agnostic directions or constructive consistency, and does not imply the existence of effective class-discriminative boundaries. Higher Evidence entropy means more uniform evidence allocation, but in medical long-tail and client-class-missing scenarios, uniform allocation weakens reliable class evidence from high-support clients. In other words, these geometric quantities should be used as joint diagnostics rather than independent optimization targets. The advantage of LAMP-Merge is that its geometric structure is consistent with the final Accuracy ablation: shared reference-space class prototypes provide stable diagnostic semantic directions, class support counts perform reliability weighting, and the long-tail prior only gives bounded calibration to the final scores.",
    150: "Additional Theoretical Analysis",
}


SECTION_IDS = {2, 22, 34, 64, 85, 98, 114, 117, 143, 150}
SUBSECTION_IDS = {6, 15, 23, 27, 32, 35, 52, 75, 86, 98, 99, 109, 133, 139}
SUBSUBSECTION_IDS = {77, 81, 87, 90, 92, 95, 100, 103}


def contains_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def latex_escape(text: str) -> str:
    text = text.replace("\\texttt{", "@@TEXTTT_OPEN@@")
    text = text.replace("}", "@@RBRACE@@")
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = "".join(replacements.get(ch, ch) for ch in text)
    out = out.replace("@@TEXTTT_OPEN@@", r"\texttt{")
    out = out.replace("@@RBRACE@@", "}")
    return out


def comment_original(text: str) -> list[str]:
    if not text:
        return []
    lines = []
    for line in text.splitlines() or [text]:
        lines.append("% Original: " + line)
    return lines


def table_cell(cell: str) -> str:
    mapping = {
        "设置": "Setting",
        "方法": "Method",
        "method": "Method",
        "数据集": "Dataset",
        "BA ↑": r"BA $\uparrow$",
        "Macro-F1 ↑": r"Macro-F1 $\uparrow$",
        "ρ ↓": r"$\rho \downarrow$",
        "Ceff ↑": r"$C_{\mathrm{eff}} \uparrow$",
        "Pred-True TV ↓": r"Pred-True TV $\downarrow$",
    }
    if cell in mapping:
        return mapping[cell]
    return latex_escape(cell)


def make_table(rows: list[list[str]], idx: int) -> str:
    if not rows:
        return ""
    max_cols = max(len(row) for row in rows)
    padded = [row + [""] * (max_cols - len(row)) for row in rows]
    env = "table*" if max_cols > 7 else "table"
    width = r"\textwidth" if env == "table*" else r"\linewidth"
    align = "l" + "c" * (max_cols - 1)
    lines = []
    lines.append(r"\begin{" + env + r"}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2pt}")
    lines.append(r"\resizebox{" + width + r"}{!}{%")
    lines.append(r"\begin{tabular}{" + align + r"}")
    lines.append(r"\toprule")
    for r_idx, row in enumerate(padded):
        converted = [table_cell(cell) for cell in row]
        lines.append(" & ".join(converted) + r" \\")
        if r_idx == 0 or (idx in {89, 91, 94, 97} and r_idx == 1):
            lines.append(r"\midrule")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"}")
    lines.append(r"\end{" + env + r"}")
    return "\n".join(lines)


def include_image(name: str) -> str:
    path = MEDIA_DIR / name
    size = path.stat().st_size if path.exists() else 0
    # Formula images in the docx are small PNGs. Keep them inline and centered.
    m = re.match(r"image(\d+)\.png", name)
    image_id = int(m.group(1)) if m else 999
    if image_id <= 15 or 21 <= image_id <= 24:
        return "\n".join([
            r"\begin{center}",
            rf"\includegraphics[width=0.62\linewidth,keepaspectratio]{{figures/{name}}}",
            r"\end{center}",
        ])
    if size > 300000 or image_id in {19, 20, 25, 26, 32, 34}:
        return "\n".join([
            r"\begin{figure*}[t]",
            r"\centering",
            rf"\includegraphics[width=\textwidth,keepaspectratio]{{figures/{name}}}",
            r"\end{figure*}",
        ])
    return "\n".join([
        r"\begin{figure}[t]",
        r"\centering",
        rf"\includegraphics[width=\linewidth,keepaspectratio]{{figures/{name}}}",
        r"\end{figure}",
    ])


def paragraph_command(idx: int, text: str) -> str:
    visible = TRANSLATIONS.get(idx, text)
    if contains_cjk(visible):
        visible = "[English translation missing; see original comment.]"
    if idx in SECTION_IDS:
        return r"\section{" + visible + "}"
    if idx in SUBSECTION_IDS:
        return r"\subsection{" + visible + "}"
    if idx in SUBSUBSECTION_IDS:
        return r"\subsubsection{" + visible + "}"
    # Preserve comment-like design notes as visible block quotes.
    if idx in set(range(66, 75)):
        return r"\begin{quote}\small " + visible + r"\end{quote}"
    return visible + "\n"


def build_tex(blocks: list[dict]) -> str:
    lines: list[str] = []
    lines.extend([
        r"\documentclass[letterpaper]{article}",
        r"\usepackage[submission]{aaai2026}",
        r"\usepackage[utf8]{inputenc}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{adjustbox}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{url}",
        "",
        r"\begin{document}",
        "",
    ])
    for idx, block in enumerate(blocks, start=1):
        lines.append("")
        lines.append(f"% --- DOCX block {idx:03d} ---")
        if block["type"] == "p":
            lines.extend(comment_original(block.get("text", "")))
            text = block.get("text", "").strip()
            if text:
                lines.append(paragraph_command(idx, text))
            for image in block.get("images", []):
                name = image.get("file", "")
                if name:
                    lines.append("% Original image: " + name)
                    lines.append(include_image(name))
        else:
            rows = [[cell.get("text", "") for cell in row] for row in block.get("rows", [])]
            for row in rows:
                lines.append("% Original table row: " + " | ".join(row))
            lines.append(make_table(rows, idx))
    lines.extend([
        "",
        r"\end{document}",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    blocks = json.loads(BLOCKS_JSON.read_text(encoding="utf-8"))
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    for src in MEDIA_DIR.glob("*.png"):
        shutil.copy2(src, OUT / "figures" / src.name)
    template_dir = ROOT / "My_merge_ret" / "aaai26_lamp"
    for name in ["aaai2026.sty", "aaai2026.bst"]:
        src = template_dir / name
        if src.exists():
            shutil.copy2(src, OUT / name)
    (OUT / "main.tex").write_text(build_tex(blocks), encoding="utf-8")
    (OUT / "README.md").write_text(
        "# LAMP-Merge AAAI LaTeX Transcription\n\n"
        "This folder is generated from `AAAI论文思路.docx`.\n\n"
        "Build with:\n\n"
        "```bash\n"
        "pdflatex main.tex\n"
        "pdflatex main.tex\n"
        "```\n\n"
        "The Chinese source text is preserved as LaTeX comments beginning with `% Original:`.\n",
        encoding="utf-8",
    )
    (OUT / "Makefile").write_text(
        "all:\n\tpdflatex main.tex\n\tpdflatex main.tex\n\nclean:\n\trm -f *.aux *.log *.out *.toc *.bbl *.blg\n",
        encoding="utf-8",
    )
    zip_path = OUT.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    subprocess.run(["zip", "-qr", str(zip_path), OUT.name], cwd=OUT.parent, check=True)
    print(OUT)
    print(zip_path)


if __name__ == "__main__":
    main()
