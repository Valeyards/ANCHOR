# ANCHOR

## Disease-adapted Pathology Foundation Model for Neuroblastoma Characterization

## Overview
ANCHOR (Automated Neuroblastoma Characterization in Histopathology, Onco-biomarkers, and Risk) model is a unified foundation model trained on H&E stained whole-slide histopathology images to perform 12 clinically relevant tasks related to neuroblastoma diagnosis, biomarker prediction, and outcome.
It adapts the pathology foundation model Virchow2 through LoRA-based self-supervised learning to capture disease-specific histology representations for rare pediatric embryonal tumors. 
The model was developed using more than 15 million image patches from a multi-center neuroblastoma dataset and evaluated across 12 downstream tasks spanning:
- morphological diagnosis, including subtype, MKI and Shimada classification.

- molecular and protein marker prediction, including *MYCN* amplification, ALK/c-MYC expression and 1p36/11q23 deletion.

- treatment-related toxicity prediction, including liver injury and myelosuppression.

- survival outcome prediction, including overall survival and progression-free survival.

Additionally, in a randomized cross-reader study, ANCHOR demonstrated its potential for assisting pathologists in improving diagnostic accuracy, efficiency, and consistency.

## Installation
To install ANCHOR, please follow the instructions below:
1. Clone the repository:
```bash
git clone https://github.com/Valeyards/ANCHOR.git
cd ANCHOR
```
2. Create a conda environment and install the dependencies:
```bash
conda create -n anchor python=3.11
conda activate anchor
pip install -r requirements.txt
```

## Generate Tiles from Whole Slide Images
Preprocess the slides following [CLAM](https://github.com/mahmoodlab/CLAM), [TRIDENT](https://github.com/mahmoodlab/TRIDENT), [PREPATH](https://github.com/birkhoffkiki/PrePATH), or any other tile extraction method with foreground tissue segmentation and stitching.

## Use ANCHOR to extract patch-level features
1. First, request access to the model weights from the [Huggingface model page](https://huggingface.co/arvinyw/ANCHOR). 

2. Initialize ANCHOR model:
    ```python
    import torch
    import timm
    from timm.layers import SwiGLUPacked

    model = timm.create_model(
        "hf-hub:arvinyw/ANCHOR",
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    model.eval()
    ```
    or you can configure it mannually:
    ```python
    with open("config.json"), encoding="utf-8") as f:
        cfg = json.load(f)

    model_args = dict(cfg["model_args"])

    model = timm.create_model(
        cfg["architecture"],
        pretrained=False,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
        **model_args,
    )
    weights = torch.load("pytorch_model.bin")
    model.load_state_dict(weights, strict=True)
    model.eval()

    transform = create_transform(**resolve_data_config(cfg["pretrained_cfg"], model=model))
    ```

3. Running Inference

    Use the pretrained ANCHOR encoder to extract features from histopathology ROIs as follows:
    ```python
    image = Image.open(image_path).convert("RGB")
    batch = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(batch)
        class_token = output[:, 0]
        patch_tokens = output[:, 5:]
        patch_embeddings = torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
    ```

## Acknowledgements
The project was built on many amazing repositories: [Virchow2](https://huggingface.co/paige-ai/Virchow2), [PEFT](https://github.com/huggingface/peft), [DeepSpeed](https://github.com/deepspeedai/DeepSpeed), [Accelerate](https://github.com/huggingface/accelerate). We thank the authors and developers for their contributions.


## License
This repository is released under the CC BY-NC-SA 4.0 license. It may be used only for non-commercial academic research purposes with proper attribution. Commercial use, redistribution for profit, sale, or any other form of monetization is prohibited without prior approval.
