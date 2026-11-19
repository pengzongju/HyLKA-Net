# HyLKA-Net

**HyLKA-Net: Efficient Hybrid CNN-Transformer Network with Multi-Scale Large-Kernel Attention for Light Field Super-Resolution**

## 📰 News

✅ **2026**: **"HyLKA-Net: Efficient Hybrid CNN-Transformer Network with Multi-Scale Large-Kernel Attention for Light Field Super-Resolution"** has been **accepted by IEEE Transactions on Circuits and Systems for Video Technology**.

## **Requirements**

The paper reports the following implementation environment:

- **Framework**: PyTorch.
- **Training hardware**: A single NVIDIA RTX 5080 GPU.
- **Python, PyTorch, Torchvision, and CUDA versions**: 3.10.18, 2.7.1, 0.22.1, 12.8.

### **Installation**

Clone the project repository:

```bash
git clone https://github.com/pengzongju/HyLKA-Net.git
cd HyLKA-Net
```

Dependency installation commands will need to follow the environment specification supplied with the source code. The repository does not currently provide a verified installation procedure.

---

## **Training**

### **1. Prepare Training Data**

The paper uses the following five light field datasets for training and standard benchmark testing:

| Dataset | Training scenes | Testing scenes | Scene type |
| --- | ---: | ---: | --- |
| **EPFL** | 70 | 10 | Real world |
| **HCInew** | 20 | 4 | Synthetic |
| **HCIold** | 10 | 2 | Synthetic |
| **INRIA** | 35 | 5 | Real world |
| **STFgantry** | 9 | 2 | Real world |

Prepare the data according to the paper:

1. Extract the central **5 × 5 sub-aperture images (SAIs)** from each **9 × 9** light field sample.
2. Generate LR samples using **bicubic downsampling** by a factor of **0.5** for 2× SR or **0.25** for 4× SR.
3. Use LR training patches with a spatial size of **32 × 32 per view**.
4. Apply horizontal flipping, vertical flipping, and 90-degree rotation for data augmentation.
5. Train the network on the **Y channel**. Reconstruct the **Cb and Cr channels** using bicubic interpolation.


### **2. Start Training**

Use the following settings to reproduce the configuration reported in the paper:

| Setting | Value |
| --- | --- |
| SR scales | 2× and 4× |
| Angular resolution | 5 × 5 |
| Feature channels | 64 |
| Number of cascaded LKFEMs | 5 |
| LKAMs per dense large-kernel attention module (DLKAM) | 4 |
| Optimizer | Adam |
| Adam β₁ / β₂ | 0.99 / 0.999 |
| Loss function | L1 loss |
| Batch size | 1 |
| Initial learning rate | 0.0002 |
| Learning-rate schedule | Multiply by 0.5 every 10 epochs |
| Training epochs | 40 |


---

## **Testing**

### **1. Prepare Test Data**

Evaluate on **EPFL**, **HCInew**, **HCIold**, **INRIA**, and **STFgantry**, using their testing splits listed above and the central **5 × 5** views.

### **2. Start Testing**

Evaluate **2× and 4× SR** using **PSNR**, **SSIM**, **LPIPS**, and **LF-IQA**. Higher PSNR, SSIM, and LF-IQA values are better; lower LPIPS values are better.

The following PSNR/SSIM results are reported for HyLKA-Net in **Table II**:

| Scale | EPFL | HCInew | HCIold | INRIA | STFgantry | Average |
| --- | --- | --- | --- | --- | --- | --- |
| **2×** | 35.97 / 0.9830 | 38.41 / 0.9811 | 45.14 / 0.9951 | 37.45 / 0.9880 | 41.35 / 0.9952 | **39.66 / 0.9885** |
| **4×** | 29.83 / 0.9264 | 31.66 / 0.9256 | 38.07 / 0.9752 | 31.84 / 0.9562 | 32.47 / 0.9592 | **32.77 / 0.9486** |

PSNR is measured in dB. The Average column is the mean of the five dataset-level results. It is distinct from the scene-level averaging reported separately in Table V.

**Testing command, result directories, and RGB image export command:** Awaiting the source code release. Output formats and paths have not been confirmed.

*Source: paper, Section IV-A and Tables II and V (PDF pp. 7, 8, and 12). These are paper-reported results; this README does not report an independent reproduction.*

---


## 📖 Citation

If you find this work helpful, please consider citing the following paper:

```bibtex
@article{Wang_2026_TCSVT_HyLKANet,
  title     = {{HyLKA-Net}: Efficient Hybrid {CNN-Transformer} Network with Multi-Scale Large-Kernel Attention for Light Field Super-Resolution},
  author    = {Wang, Hongyang and Chen, Fen and Peng, Zongju and Jiang, Gangyi and Zhang, Lianjun},
  journal   = {IEEE Transactions on Circuits and Systems for Video Technology},
  year      = {2026},
  doi       = {10.1109/TCSVT.2026.3723121},
  publisher = {IEEE}
}
```

For questions about the paper, please open an [Issue](https://github.com/pengzongju/HyLKA-Net/issues) or contact:

- 📧 **Hongyang Wang**: 3180495018@qq.com
- 📧 **Fen Chen (corresponding author)**: chenfen1@cqut.edu.cn
