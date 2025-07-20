<h1 align="center">Hybrid Layout Control for Diffusion Transformer: Fewer Annotations, Superior Aesthetics</h1>
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
- [2025/6/26] <span style="color: red; font-weight: bold;">🎉🎉🎉 HybridLayout is accepted by <span style="color: red; font-weight: bold;">ICCV 2025</span>!</span>



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
