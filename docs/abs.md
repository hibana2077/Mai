# Thesis Title (Abbreviation = Mai)

**Mai: Masked Affinity Interaction kernel for partially observable video interactions**
(Chinese literal translation: **Mai: Masked Affinity Interaction Kernel for Partially Observable Video Interactions**)

---

# Core Research Question

In video understanding (action recognition, HOI, human-human/human-object interaction), the real world often leads to **partial observability** due to occlusion, viewpoint, and sensing limitations. How can we stably model **cross-temporal interaction structures** while **only seeing fragmented representations**, and have **provable positive semi-definite kernels (PSD) and consistency**, while being able to **train at scale**?

Related background: **Conditional probability embedding / mean embedding** in RKHS can perform non-parametric inference and temporal state representation (PSR/HMM/POMDP in RKHS) without explicit density estimation, providing us with a rigorous probability-calculation framework. ([arXiv][1], [Gatsby][2])

---

# Research Objectives

1.  Design a video kernel function **Mai-kernel** that is **friendly to partial observability and interaction structures**, while:

    *   Being expressive for **object-object interaction** and **cross-temporal alignment**;
    *   **Maintaining provable PSD** under "visible/invisible" masks;
    *   Being able to be **linearized and scaled** to large data using methods such as **random features / Nyström / FALKON**. ([BPB][3], [Wikipedia][4], [NeurIPS Papers][5], [arXiv][6])
2.  Establish **differentiable approximations** (using soft-DTW/Global Alignment-like kernels for temporal alignment) to support end-to-end training. ([ICML][7], [arXiv][8])
3.  Validate on **video interaction tasks** (action, HOI, social interaction prediction), especially for **partially visible scenarios**. See below for dataset planning. ([arXiv][9], [GitHub][10], [ego4d-data.org][11])

---

# Contributions and Innovations (leaning towards mathematical theory, with feasibility checked)

**Mai-kernel: Masked Affinity Interaction kernel**
Given two videos $V, V'$. At each time $t$, there is a set of objects $\mathcal{O}_t = \{z_{t,i}\}$ and a visibility mask $m_{t,i} \in \{0, 1\}$. Define base kernels:

*   Object representation kernel $\kappa_Z(z, z')$: Standard PSD kernels such as RBF/linear, which can be approximated by random features. ([NeurIPS Papers][5])
*   Temporal alignment kernel $\kappa_T(t, s)$: Adopting the **Global Alignment (GAK) / soft-DTW** family, known to be PSD or differentiable smooth approximations under appropriate conditions. ([ICML][7], [arXiv][8])
*   Mask kernel $\kappa_M(m, m') = \langle m, m' \rangle$ (linear kernel, 0/1 indicator, naturally PSD).

Define **interaction pairs** $(i, j)$ ($i \neq j$) for pairwise relationships at time $t$, Mai-kernel:

$$
K_{\text{Mai}}(V,V')=\!\sum_{t,s}\!\kappa_T(t,s)\!\!\sum_{(i\neq j)\in\mathcal{O}_t}\sum_{(u\neq v)\in\mathcal{O}'_s}
\big[\kappa_Z(z_{t,i},z'_{s,u})\kappa_Z(z_{t,j},z'_{s,v})\big]\,
\kappa_M(m_{t,i},m'_{s,u})\,\kappa_M(m_{t,j},m'_{s,v}).
$$

This is the Haussler R-convolution idea (summing over "parts" and then taking the **product**), combining temporal alignment and mask gating. **Product and summation preserve PSD** (Schur product theorem & kernel closure), so $K_{\text{Mai}}$ is PSD. ([BPB][3], [University College London Department of Computer Science][12], [Wikipedia][4])

**Feasibility Check**:

*   If interaction is turned off (only single objects are left), Mai degenerates into a time-aligned set kernel/GAK, which can be analyzed with existing theory; if the mask is turned off, it degenerates into a many-to-many interaction kernel; both are known PSD categories of sums/products. ([ICML][7], [BPB][3])
*   The overall computation graph can be linearized using **Random Fourier Features / Nyström / FALKON**, with known statistical generalization and computational bounds. ([NeurIPS Papers][5], [arXiv][6])

---

# Mathematical Theory Derivation and Proof (Summary)

**Theorem 1 (PSD)**
If $\kappa_Z, \kappa_T, \kappa_M$ are all PSD kernels, then $K_{\text{Mai}}$ is PSD.
*Proof key points*: The product kernel of $\kappa_Z\kappa_Z$ is PSD; multiplying by $\kappa_M\kappa_M$ is still PSD (Schur product theorem). Summing over all interaction pairs and temporal alignments (R-convolution) still preserves PSD. ([Wikipedia][4], [BPB][3])

**Theorem 2 (Expressiveness / Generality)**
If $\kappa_Z$ takes sufficient eigenvalues (such as Gaussian RBF) and $\kappa_T$ takes GAK/soft-DTW, then $K_{\text{Mai}}$ can approximate any continuous interaction similarity function (on appropriate compact subsets).
*Proof key points*: Universality of RBF + closure of R-convolution in composite structures; temporal alignment kernels preserve the approximation of variable-length sequences. ([BPB][3], [ICML][7])

**Theorem 3 (Random Feature Consistency)**
Approximating $\kappa_Z$ with $D$-dimensional RFF, and the temporal kernel part with Nyström approximation, then $\sup_{V,V'}|K_{\text{Mai}}-\tilde K_{\text{Mai}}|=O_p(D^{-1/2})$ and with $O(\sqrt{n}\log n)$ level of features under sample size $n$, an $O(n^{-1/2})$ learning bound can be achieved.
*Proof key points*: Inherit the approximation and learning bounds of Rahimi-Recht and Rudi-Rosasco, and then use the triangle inequality to pass to Mai composed of sums/products. ([People @ EECS][13], [arXiv][6])

**Theorem 4 (Generalization Bound for Kernel Ridge Regression/Classification)**
Using $K_{\text{Mai}}$ for kernel ridge regression or SVM, if FALKON / Nyström scaling is adopted, then **optimal statistical accuracy** level can be achieved in $O(n\sqrt{n})$ time and near $O(n)$ memory.
*Proof key points*: Directly cite FALKON and random feature generalization results. ([arXiv][14], [papers.neurips.cc][15])

**Note (Connection with Partial Observability)**
The mask gating of Mai is equivalent to selecting visible subsets for **conditional distribution embedding** in RKHS, which is compatible with the operations of kernel Bayes, PSR-in-RKHS / POMDP-in-RKHS, ensuring consistent updating and comparison under occlusion/missingness. ([Gatsby][2], [arXiv][16])

---

# Expected Datasets (covering interaction, temporal reasoning, occlusion/partial visibility)

*   **Something-Something V2**: Object interaction + strong temporal dependencies. ([Papers with Code][17])
*   **EPIC-KITCHENS-100 (including VISOR)**: Egocentric perspective, natural occlusion and hand-object interaction; VISOR provides pixel-level annotations. ([Papers with Code][18], [EPIC Kitchens][19])
*   **AVA / AVA-Kinetics**: Spatio-temporal localization and interaction scenes of atomic actions. ([Google Research][20], [arXiv][9])
*   **VidHOI**: Video HOI benchmark (explicit interaction annotation). ([GitHub][10])
*   **Ego4D (Social / Episodic Memory)**: Social interaction, long duration and frequent occlusion. ([ego4d-data.org][11])

---

# Experimental Design and Evaluation

**Tasks**

1.  Action classification (SSv2) 
2.  Temporal localization (AVA) 
3.  Video HOI detection (VidHOI) 
4.  Egocentric interaction segmentation/retrieval (EPIC-KITCHENS + VISOR, Ego4D).

**Metrics**
Top-1/Top-5, mAP (HOI/AVA), IoU/F1 (segmentation/localization), and gains on **occlusion subsets**.

**Comparison Groups and Ablation**

*   No mask gating (remove $\kappa_M$);
*   No interaction (only do single object aggregation);
*   No temporal alignment ($\kappa_T$ replaced with δ);
*   RFF/Nyström/FALKON different approximation and complexity scans;
*   Compare or combine with **VideoMAE**-like self-supervision as pre-processing (see below). ([arXiv][21])

**Combination with Self-Supervision (Optional)**
Use **"Masked Kernel Alignment" (MKA)** as pretext: randomly mask tubes, requiring the kernel alignment induced by Mai for visible fragments to be maximized (connected to CKA/Kernel Alignment theory). ([NeurIPS Papers][22], [arXiv][23])

---

# Successful Submission Plan (Main Line)

1.  **Method Chapter**: Mai definition + **PSD proof** + approximation and complexity analysis (citing FALKON/RFF/Nyström). ([arXiv][14], [People @ EECS][13])
2.  **Implementation**: Detection/tracking extracts object features (existing backbones can be used), Mai is used to calculate the similarity between videos or as the kernel of kernel SVM/kernel Ridge/kernel GP; HOI tasks adopt kernelized structure learning or use Mai as the kernel of the relationship graph.
3.  **Results**: On SSv2/AVA/VidHOI/EPIC-KITCHENS (including VISOR) and Ego4D subtasks, especially report the robustness curve when the **occlusion ratio increases**. ([Papers with Code][17], [CVF Open Access][24], [GitHub][10], [EPIC Kitchens][19], [ego4d-data.org][11])
4.  **Interpretability**: Analyze the stability of representations after adding masks with kernel alignment (CKA). ([Proceedings of Machine Learning Research][25])

# Failure Backup Plan (Secondary Line)

*   If end-to-end kernel-based final classification has limited improvement, switch to **"Mai as pretext / regularizer"**: after VideoMAE pre-training, use **MKA loss** to fine-tune to enhance interaction and occlusion robustness; or insert Mai as a **retrieval/re-weighting module** into the existing Transformer pipeline. ([arXiv][21])
*   If the cost of the temporal kernel is too high, use **low-rank approximation
# Differences from Existing Research

*   **Handles the "interaction × partial observability × temporal alignment" trilemma from the kernel design level**, rather than relying solely on deep architectures and data augmentation; Mai is synthesized with **mask gating + R-convolution of interaction pairs + temporal alignment kernel**, and is **strictly PSD** (proven). ([BPB][3], [Wikipedia][4])
*   Unlike VideoMAE/contrastive learning, Mai provides **provable kernelized objectives** and concise scalable approximation strategies (RFF/Nyström/FALKON), with **statistical-computational dual guarantees**. ([arXiv][21])
*   Inherits from the **observable state embedding** of PSR/POMDP-in-RKHS, but directly faces **video interaction** and **explicitly models occlusion**. ([arXiv][16])

---

# Supplement: Connection with Classical Theories (Convenient for Writing Related Work)

*   **Conditional mean embedding / Kernel Bayes / PSR-in-RKHS** provide non-parametric temporal reasoning toolboxes. ([Gatsby][2], [arXiv][27])
*   **Operator-valued kernels / Vector-valued RKHS** can be extended to multi-task output (such as simultaneously predicting actions and relationships). ([PubMed][28], [Journal of Machine Learning Research][29])
*   **Koopman × RKHS / kEDMD** can be used as a further dynamical system perspective (optional appendix). ([NeurIPS Papers][30], [arXiv][31])

---

## Reference Points (Can be placed in the introduction/method proof of the paper)

*   Kernel mean/conditional embeddings review and tools. ([nowpublishers.com][32])
*   RKHS representation of PSR/HMM/POMDP. ([arXiv][27])
*   R-convolution kernel and kernel closure (Schur product). ([BPB][3], [Wikipedia][4])
*   Temporal alignment kernel (GAK/soft-DTW). ([ICML][7], [arXiv][26])
*   Large-scale kernel approximation and generalization bounds (RFF/Nyström/FALKON). ([People @ EECS][13], [Proceedings of Machine Learning Research][33], [arXiv][14])
*   Self-supervised alignment (CKA/Alignment) can be used as an auxiliary objective. ([Proceedings of Machine Learning Research][25], [NeurIPS Papers][22])
*   Video/interaction datasets. ([Papers with Code][17], [EPIC Kitchens][19], [Google Research][20], [arXiv][9], [GitHub][10], [ego4d-data.org][11])

---

[1]: https://arxiv.org/abs/1605.09522?utm_source=chatgpt.com "Kernel Mean Embedding of Distributions: A Review and Beyond"
[2]: https://www.gatsby.ucl.ac.uk/~gretton/papers/SonFukGre13.pdf?utm_source=chatgpt.com "Kernel Embeddings of Conditional Distributions"
[3]: https://bpb-us-e1.wpmucdn.com/sites.ucsc.edu/dist/4/821/files/2018/12/Convolution-Kernels.pdf?utm_source=chatgpt.com "Convolution Kernels on Discrete Structures UCSC-CRL-99-10"
[4]: https://en.wikipedia.org/wiki/Schur_product_theorem?utm_source=chatgpt.com "Schur product theorem"
[5]: https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines?utm_source=chatgpt.com "Random Features for Large-Scale Kernel Machines"
[6]: https://arxiv.org/abs/1602.04474?utm_source=chatgpt.com "Generalization Properties of Learning with Random Features"
[7]: https://icml.cc/2011/papers/489_icmlpaper.pdf?utm_source=chatgpt.com "Fast Global Alignment Kernels"
[8]: https://arxiv.org/abs/cs/0610033?utm_source=chatgpt.com "A kernel for time series based on global alignments"
[9]: https://arxiv.org/abs/2005.00214?utm_source=chatgpt.com "The AVA-Kinetics Localized Human Actions Video Dataset"
[10]: https://github.com/coldmanck/VidHOI?utm_source=chatgpt.com "coldmanck/VidHOI: Official implementation of \"ST-HOI"
[11]: https://ego4d-data.org/docs/benchmarks/overview/?utm_source=chatgpt.com "Benchmarks Overview"
[12]: https://www0.cs.ucl.ac.uk/staff/m.pontil/reading/haussler.pdf?utm_source=chatgpt.com "Convolution Kernels on Discrete Structures"
[13]: https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf?utm_source=chatgpt.com "Random Features for Large-Scale Kernel Machines"
[14]: https://arxiv.org/pdf/1705.10958?utm_source=chatgpt.com "FALKON: An Optimal Large Scale Kernel Method"
[15]: https://papers.neurips.cc/paper/6978-falkon-an-optimal-large-scale-kernel-method.pdf?utm_source=chatgpt.com "FALKON: An Optimal Large Scale Kernel Method"
[16]: https://arxiv.org/abs/1210.4887?utm_source=chatgpt.com "Hilbert Space Embeddings of POMDPs"
[17]: https://paperswithcode.com/dataset/something-something-v2?utm_source=chatgpt.com "Something-Something V2 Dataset"
[18]: https://paperswithcode.com/dataset/epic-kitchens-100?utm_source=chatgpt.com "EPIC-KITCHENS-100 Dataset"
[19]: https://epic-kitchens.github.io/VISOR/?utm_source=chatgpt.com "VISOR annotations - EPIC-KITCHENS Dataset"
[20]: https://research.google.com/ava/?utm_source=chatgpt.com "AVA: A Video Dataset of Atomic Visual Action"
[21]: https://arxiv.org/abs/2203.12602?utm_source=chatgpt.com "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training"
[22]: https://papers.nips.cc/paper/1946-on-kernel-target-alignment?utm_source=chatgpt.com "On Kernel-Target Alignment"
[23]: https://arxiv.org/abs/1905.00414?utm_source=chatgpt.com "Similarity of Neural Network Representations Revisited"
[24]: https://openaccess.thecvf.com/content_cvpr_2018/papers/Gu_AVA_A_Video_CVPR_2018_paper.pdf?utm_source=chatgpt.com "AVA: A Video Dataset of Spatio-Temporally Localized ..."
[25]: https://proceedings.mlr.press/v97/kornblith19a/kornblith19a.pdf?utm_source=chatgpt.com "Similarity of Neural Network Representations Revisited"
[26]: https://arxiv.org/abs/1703.01541?utm_source=chatgpt.com "Soft-DTW: a Differentiable Loss Function for Time-Series"
[27]: https://arxiv.org/abs/1309.6819?utm_source=chatgpt.com "Hilbert Space Embeddings of Predictive State Representations"
[28]: https://pubmed.ncbi.nlm.nih.gov/15563752/?utm_source=chatgpt.com "On learning vector-valued functions"
[29]: https://jmlr.org/papers/volume17/11-315/11-315.pdf?utm_source=chatgpt.com "Operator-valued Kernels for Learning from Functional ..."
[30]: https://papers.nips.cc/paper/6583-dynamic-mode-decomposition-with-reproducing-kernels-for-koopman-spectral-analysis?utm_source=chatgpt.com "Dynamic Mode Decomposition with Reproducing Kernels ..."
[31]: https://arxiv.org/abs/2403.18809?utm_source=chatgpt.com "$L^\infty$-error bounds for approximations of the Koopman operator by kernel extended dynamic mode decomposition"
[32]: https://www.nowpublishers.com/article/DownloadSummary/MAL-060?utm_source=chatgpt.com "Kernel Mean Embedding of Distributions: A Review and ..."
[33]: https://proceedings.mlr.press/v162/chatalic22a/chatalic22a.pdf?utm_source=chatgpt.com "Nyström Kernel Mean Embeddings"
