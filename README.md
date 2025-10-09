**Official Implementation for CIKM 2025 Paper**

This repository provides the official implementation of our CIKM 2025 paper:

**ParaStyleTTS: Toward Efficient and Robust Paralinguistic Style Control for Expressive Text-to-Speech Generation**

## 🧠 Overview

<p align="center">
  <img src="media/image.png" alt="ParaStyleTTS Overview" width="720">
</p>


## Abstract
Controlling speaking style in text-to-speech (TTS) systems has become a growing focus in both academia and industry. While many existing approaches rely on reference audio to guide style generation, such methods are often impractical due to privacy concerns and limited accessibility. More recently, large language models (LLMs) have been used to control speaking style through natural language prompts; however, their high computational cost, lack of interpretability, and sensitivity to prompt phrasing limit their applicability in real-time and resource-constrained environments. In this work, we propose ParaStyleTTS, a lightweight and interpretable TTS framework that enables expressive style control from text prompts alone. ParaStyleTTS features a novel two-level style adaptation architecture that separates prosodic and paralinguistic speech style modeling. It allows fine-grained and robust control over factors such as emotion, gender, and age. Unlike LLM-based methods, ParaStyleTTS maintains consistent style realization across varied prompt formulations and is well-suited for real-world applications, including on-device and low-resource deployment. Experimental results show that ParaStyleTTS generates high-quality speech with performance comparable to state-of-the-art LLM-based systems while being 30x faster, using 8x fewer parameters, and requiring 2.5x less CUDA memory. Moreover, ParaStyleTTS exhibits superior robustness and controllability over paralinguistic speaking styles, providing a practical and efficient solution for style-controllable text-to-speech generation. Demo can be found at https://parastyletts.github.io/ParaStyleTTS_Demo/.

## 🚀 How to Run ParaStyleTTS

1. **Download the checkpoints**

   Download the pretrained checkpoints from the following link:  
   [Download Checkpoints (OneDrive)](https://unsw-my.sharepoint.com/:u:/g/personal/z5258575_ad_unsw_edu_au/EVu9cwOmIfJNmeMdI5R3ZtcBV0slBICNHUZBW7bYRy-ZzA?e=itFBau)

   Place the downloaded files inside the `ckp/` folder.

2. **Generate speech samples**

   Run the following command:

   ```bash
   python generate.py -c config/config.json
   ```
