<h1 align="center">Hybrid Layout Control for Diffusion Transformer: Fewer Annotations, Superior Aesthetics</h1>

<div style="font-size: 1.25rem; text-align: center;">
  <a href="https://kemingwu.github.io/" target="_blank">Keming Wu</a><sup>1*</sup>,
  <a href="https://www.microsoft.com/en-us/research/" target="_blank">Junwen Chen</a><sup>3*</sup>,
  <a href="https://www.microsoft.com/en-us/research/" target="_blank">Zhanhao Liang</a><sup>2*</sup>,
  <a href="https://www.microsoft.com/en-us/research/" target="_blank">Yinuo Wang</a><sup>1*</sup>,
  <a href="https://www.microsoft.com/en-us/research/" target="_blank">Ji Li</a><sup>5</sup>,
  <a href="https://scholar.google.com/citations?user=NeCCx-kAAAAJ&hl=en" target="_blank">Chao Zhang</a><sup>4</sup>,
  <a href="https://binwangthss.github.io/" target="_blank">Bin Wang</a><sup>1</sup>,
  <a href="https://www.microsoft.com/en-us/research/people/yuyua/" target="_blank">Yuhui Yuan</a><sup>6*</sup>
</div>

<div style="font-size: 1rem; text-align: center; margin-top: 0.5rem;">
  <sup>1</sup>Tsinghua University &emsp;
  <sup>2</sup>The Australian National University &emsp;
  <sup>3</sup>The University of Electro-Communications Tokyo &emsp;
  <sup>4</sup>Peking University &emsp;
  <sup>5</sup>Microsoft Research &emsp;
  <sup>6</sup>Canva Research
</div>

<div style="font-size: 0.9rem; text-align: center; margin-top: 0.3rem;">
  <sup>*</sup>Work done at Microsoft Research Asia
</div>

<h3 align="center">🌟ICCV 2025🌟</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2503.20672"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://hybrid-layout-msra.github.io/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/wukeming11/HybridLayout'><img src='https://img.shields.io/badge/Model-Huggingface-yellow?logo=huggingface&logoColor=yellow' alt='Model'></a>

<table>
  <tr>
    <td><img src="assets\instance_diff_example.jpg" alt="gif5" width="150"></td>
    <td><img src="assets\dog_ours.jpg" alt="gif5" width="150"></td>
    <td><img src="assets\sa_11143675_style7.jpg" alt="gif1" width="150"></td>
    <td><img src="assets\sa_11143664_style2.jpg" alt="gif2" width="150"></td>
    <td><img src="assets\sa_11143662_style2.jpg" width="150"></td>
  </tr>
</table>

<table>
  <tr>
     <td><img src="assets\sa_11143631_style6.jpg" alt="gif4" width="150"></td>
    <td><img src="assets\sa_11143576_style4.jpg" alt="gif1" width="150"></td>
    <td><img src="assets\sa_11143554_style7.jpg" alt="gif2" width="150"></td>
    <td><img src="assets\sa_11143536_style8.jpg" alt="gif3" width="150"></td>
    <td><img src="assets\sa_11143524_style7.jpg" alt="gif4" width="150"></td>
  </tr>
</table>


## :fire: News

- [2025/7/20] Repository is initialized.
- [2025/6/26] 🎉🎉🎉 HybridLayout is accepted by **ICCV 2025**! 🎉🎉🎉


<!-- TODO List -->
## 🚧 TODO List
- [ ] Release inference code and pretrained model
- [ ] Release training code


## Table of Contents
- [Environment Setup](#environment-setup)
- [Testing](#testing-hybridlayout)

## Environment Setup

### 1. Create Conda Environment
```bash
conda create -n hybrid_layout python=3.10 -y
conda activate hybrid_layout
```

### 2. Install Dependencies 
```bash
git clone https://github.com/KemingWu/HybridLayout.git
cd HybridLayout
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0  pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install diffusers==0.31.0 transformers==4.44.0 accelerate==0.34.2 peft==0.12.0 datasets==2.20.0 prodigyopt
pip install wandb==0.17.7 einops==0.8.0 sentencepiece==0.2.0 mmengine==0.10.4
pip install braceexpand==0.1.7 webdataset==0.2.100
pip3 install -U xformers --index-url https://download.pytorch.org/whl/cu118
```

### 3. Login to Hugging Face
```bash
huggingface-cli login
```

## ✨ Quick Start  
Use inference.py to simply have a try:
```
python inference.py
```

## :mailbox_with_mail: Citation
If you find this code useful in your research, please consider citing:

```

```
