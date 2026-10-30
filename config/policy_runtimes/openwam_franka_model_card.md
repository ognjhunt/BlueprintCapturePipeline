---
license: apache-2.0
library_name: openwam
pipeline_tag: robotics
tags:
  - robotics
  - world-action-model
  - openwam
---

# OpenWAM-Alpha-Real-Single-Arm-Franka

OpenWAM-α applies the recipe selected by the OpenWAM study: a Wan2.2-TI2V-5B video backbone with its native Wan-VAE encoder, a Dual-System Joint Self-Attention architecture, and a mutual attention mask. It is pretrained on 518.5M frames (about 6,400 hours), mixing robot data (70%: 40% real-world, 30% simulation) with egocentric human data (30%).

Finetuned for single-arm manipulation on the **Franka-Research-3** platform, covering six real-world tabletop tasks.

- Paper: https://arxiv.org/abs/2609.07398
- Code: https://github.com/OpenWAM-Official/OpenWAM
- Project page: https://openwam-official.github.io/

## Citation

```bibtex
@article{wang2026openwam,
  title   = {OpenWAM: An Open, Modular Exploration Towards Systematic World-Action Model Pretraining},
  author  = {Yuran Wang and Siqiao Huang and Mingleyang Li and Chenhao Zhang and Jiaqi Liang and Weiyang Jin and Yue Chen and Xuemin Chi and Donghao Zhou and Qize Yu and Yu-Kai Wang and Yuhan Rui and Shenzhe Yao and Zhen Yuan and Zhenhao Shen and Kefei Zhu and Zijie Zhu and Ning Gao and Xiaowei Chi and Guanqi He and Shanghang Zhang and Hao Dong and Lin Shao and Hang Zhao},
  year    = {2026},
  journal = {arXiv preprint arXiv: 2609.07398}
}
```
