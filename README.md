# Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning

* **Title**: **[Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning](https://arxiv.org/pdf/2303.15230.pdf)**
* **Authors**: [Siteng Huang](https://kyonhuang.top/), Biao Gong, Yutong Feng, Yiliang Lv, [Donglin Wang](https://milab.westlake.edu.cn/)
* **Institutes**: Westlake University, Alibaba Group, Zhejiang University
* **More details**: [[arXiv]](https://arxiv.org/abs/2303.15230) | [[homepage]](https://kyonhuang.top/publication/Troika)

The code will be available upon acceptance and approval.

## Overview

![](https://kyonhuang.top/files/Troika/Troika-overview.png)

With a particular focus on the universality of the solution, in this work, we propose a novel **Multi-Path paradigm** for VLM-based CZSL models that establishes three identification branches to jointly model the state, object, and composition. The presented **Troika** is an outstanding implementation that aligns the branch-specific prompt representations with decomposed visual features. To calibrate the bias between semantically similar multi-modal representations, we further devise a **Cross-Modal Traction module** into Troika that shifts the prompt representation towards the current visual content. Experiments show that on the closed-world setting, Troika exceeds the current state-of-the-art methods by up to **+7.4%** HM and **+5.7%** AUC. And on the more challenging open-world setting, Troika still surpasses the best CLIP-based method by up to **+3.8%** HM and **+2.7%** AUC.

## Results

### Main Results

The following results are obtained with a pre-trained CLIP (ViT-L/14). More experimental results can be found in the paper.

![](https://kyonhuang.top/files/Troika/Troika-SOTA.png)

### Qualitative Results

![](https://kyonhuang.top/files/Troika/Troika-qualitative-results.png)

## Citation

If you find this work useful in your research, please cite our paper:

```
@article{Huang2023Troika,
    title={Troika: Multi-Path Cross-Modal Traction for Compositional Zero-Shot Learning},
    author={Siteng Huang and Biao Gong and Yutong Feng and Yiliang Lv and Donglin Wang},
    journal={arXiv preprint arXiv:2303.15230},
    year={2023}
}
```

## Acknowledgement

Our code references the following projects:

* [DFSP](https://github.com/Forest-art/DFSP)