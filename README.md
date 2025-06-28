# FSVS-Net: A Few-shot Semi-Supervised Vessel Segmentation Network for Multiple Organs Based on Feature Distillation and Bidirectional Weighted Fusion
<div align="center">
  <img src="https://github.com/user-attachments/assets/d7280999-0752-42e1-b914-493098219fad" alt="FSVS-Net Architecture" width="800"/>
</div>

This repository contains the implementation, dataset, and documentation for the paper:
```paper
FSVS-Net: A Few-shot Semi-Supervised Vessel Segmentation Network for Multiple Organs Based on Feature Distillation and Bidirectional Weighted Fusion
```

Authors: Yuqun Yang, Jichen Xu, Mengyuan Xu, et al.

Published in: Information Fusion, 2025

Link:
[https://www.sciencedirect.com/science/article/pii/S1566253525003549](https://www.sciencedirect.com/science/article/pii/S1566253525003549)

## Introduction

FSVS-Net is a few-shot semi-supervised vessel segmentation network with feature distillation and bidirectional weighted fusion. Key innovations include: 

- A feature distillation module to improve the accuracy of vessel representation,.
- Bidirectional weighted fusion strategy for leveraging the relationships between adjacent slices.
- Evaluation on three datasets for hepatic vessels, pulmonary vessels, and renal arteries segmentation.


This repository provides:
- The source code for training and evaluation.
- The labeled datasets.

## Datasets

Related datasets are being orgnized, and are expected to be available in July (originally planned for June, but delayed due to the complexity of the dataset preparation).

## License

This project is licensed under the MIT License.

## Acknowledgements

This code is based on [Mem3D](https://github.com/lingorX/Mem3D) and [STCN](https://github.com/hkchengrex/STCN).

## Contact

For any questions or issues, feel free to reach out to
- Yuqun Yang: yqunyang@163.com
- Jichen Xu: branrafa95@gmail.com
- Bo Wang: bow@hust.edu.cn

If you find this work helpful in your research or projects, please consider citing the following paper:
```bibtex
@article{yang2025fsvs,
  title={FSVS-Net: A few-shot semi-supervised vessel segmentation network for multiple organs based on feature distillation and bidirectional weighted fusion},
  author={Yang, Yuqun and Xu, Jichen and Xu, Mengyuan and Tang, Xu and Wang, Bo and Shu, Kechen and You, Zheng},
  journal={Information Fusion},
  pages={103281},
  year={2025},
  publisher={Elsevier}
}
