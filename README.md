<h1 align="center">Hybrid Layout Control for Diffusion Transformer: Fewer Annotations, Superior Aesthetics</h1>
<h3 align="center">🌟ICCV 2025🌟</h3>
<p align="center">
  <a href="https://arxiv.org/abs/2503.20672"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
  <a href='https://hybrid-layout-msra.github.io/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
  <a href='https://huggingface.co/wukeming11/HybridLayout'><img src='https://img.shields.io/badge/Model-Huggingface-yellow?logo=huggingface&logoColor=yellow' alt='Model'></a>

<table>
  <tr>
    <td><img src="instance_diff_example.jpg" alt="gif5" width="150"></td>
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

<span style="font-size: 16px; font-weight: 600;">This repository supports article-level visual text rendering of business content (infographics and slides) based on ultra-dense layouts

<!-- Features -->
## 🌟 Features
- **Long context length**: Supports ultra-dense layouts with 50+ layers and article-level descriptive prompts with more than 1000 tokens, and can generate high-quality business content with up to 2240*896 resolution.
- **Powerful visual text rendering**: Supports article-level visual text rendering in ten different languages and maintains high spelling accuracy.
- **Image generation diversity and flexibility**: Supports layer-wise detail refinement through layout conditional CFG.


<!-- TODO List -->
## 🚧 TODO List
- [x] Release inference code and pretrained model
- [ ] Release training code


## Table of Contents
- [Environment Setup](#environment-setup)
- [Testing](#testing-bizgen)

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
```

### 3. Login to Hugging Face
```bash
huggingface-cli login
```

## Quick Start
Use inference.py to simply have a try:
```
python inference.py
```

## Testing BizGen

### 1. Download Checkpoints

Create a path `bizgen/checkpoints` and download the following [checkpoints](https://huggingface.co/PYY2001/BizGen) into this path.

| Name | Description|
|----------|-------------|
| `byt5` | ByT5 model checkpoint |
| `lora_infographic` | Unet LoRA weights and finetuned ByT5 mapper checkpoint for infographic |
| `lora_slides` | Unet LoRA weights and finetuned ByT5 mapper checkpoint for slides |
| `spo` | Post-trained SDXL checkpoint (for aesthetic improvement) |

The downloaded checkpoints should be organized as follows:
```
checkpoints/
├── byt5/
│   ├── base.pt
│   └── byt5_model.pt
├── lora/
|   ├── infographic/
|   |   ├──byt5_mapper.pt
|   |   └──unet_lora.pt
|   └── slides/
|       ├──byt5_mapper.pt
|       └──unet_lora.pt
└── spo
```

### 2. Run the testing Script
```bash
python inference.py \
--ckpt_dir checkpoints/lora/infographic \
--output_dir infographic \
--sample_list meta/infographics.json 
```


## :mailbox_with_mail: Citation
If you find this code useful in your research, please consider citing:

```

```
