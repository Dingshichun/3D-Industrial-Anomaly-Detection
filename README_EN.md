# 3D Industrial & Agricultural Defect Detection

A complex industrial and agricultural anomaly and defect detection framework based on multi-modal feature fusion (2D RGB + 3D point cloud spatial information).
This project is designed for targets with complex surface topologies and irregular textures, accurately detecting scratches on rigid parts as well as bruises and degradation on agricultural products.
[中文文档 (Chinese Verification Document)](README.md)

## 🌟 Core Features
This project integrates two powerful anomaly detection algorithms customized with a **Strategy Router** to handle various unaligned and inherently different topological industrial objects within a single codebase:

1. **AST (Asymmetric Student-Teacher) Reverse Distillation Baseline**
   * **Mechanism**: Creates a Teacher manifold space, while forcing a Student Decoder to attempt reconstruction of non-anomalous normal features. Defects are scored functionally by measuring teacher and student representational distance at test time.
   * **Best Scope**: Rigid objects and items with flat, low-frequency geometric shifts (e.g., `dowel`).
2. **Spatial PatchCore (3D Enhanced PatchCore)**
   * **Mechanism**: Leverages an image backbone for extracting local patch features, dynamically reinforced with XYZ pseudo-coordinates and **3D Surface Normals derived by discrete pseudo-Sobel computations**. Fast GPU batch mapping via native PyTorch `cdist` ensures rapid evaluation against dense K-Nearest feature memory banks.
   * **Best Scope**: Highly textured objects or items with non-ordered topology or porous volumes (e.g., `foam`, `tire`, `rope`, `cable_gland`).
3. **Smart Category Router System**
   * Fully automated hyperparameter selection (`blur_radius, top_k, subsample, xyz_weight`) explicitly optimized for specific materials.

## 📊 Performance Metrics
Under typical evaluation metrics, the system resolves across multiple objects:

| Category | Routing Strategy | Sample AUROC | Point AUROC |
| :--- | :--- | :--- | :--- |
| `cable_gland` | Spatial PatchCore (`xyz_weight=15.0`) | **0.8331** | **0.9444** |
| `dowel` | Baseline ASTEvaluator | **0.9704** | **0.9967** |
| `foam` | PatchCore + 3D Feature (`blur_radius=2, top_k=5`) | **0.7575** | **0.9332** |
| `tire` | PatchCore + 3D Feature (`subsample=0.2, top_k=50`)| **0.7237** | **0.9878** |
| `rope` | PatchCore + 3D Feature (`blur_radius=8, top_k=400`) | **0.7867** | **0.9901** |
| `carrot` (Agriculture) | Spatial PatchCore (`xyz_weight=0.0, blur_radius=4, subsample=0.1`) | **0.9122** | **0.9938** |
| `potato` (Agriculture) | Spatial PatchCore (`xyz_weight=0.0, blur_radius=4, subsample=0.1`) | **0.8557** | **0.9961** |
| `peach` (Agriculture) | Spatial PatchCore (`xyz_weight=0.0, blur_radius=4, subsample=0.1`) | **0.8414** | **0.9936** |

## 📸 Visualization Example
The system enables automatic diagnostic visualizations by masking irrelevant background point clouds (`depth=0`) and rendering continuous defect heatmaps layered smoothly onto RGB input limits.

**Below are detection examples of surface contamination for `dowel`, `rope`, `tire`, `carrot`, `peach`, and `potato`:**

Industrial Products
![Dowel Contamination](visualizations/dowel/dowel_contamination_4.png)
![Rope Contamination](visualizations/rope/rope_contamination_3.png)
![Tire Contamination](visualizations/tire/tire_contamination_2.png)  
Agricultural Products
![Carrot Contamination](visualizations/carrot/carrot_contamination_4.png)
![Peach Contamination](visualizations/peach/peach_contamination_3.png)
![Potato Contamination](visualizations/potato/potato_contamination_2.png)
*(Inputs mapping -> left to right: RGB | Z-Depth | Predicted Distances Heatmap | Ground Truth Mask)*

## 🛠️ Installation
A capable CUDA environment and compatible PyTorch layout are needed.

```bash
conda create --name pytorch_gpu python==3.10
conda activate pytorch_gpu

pip install -r requirements.txt
```

## 🚀 Quick Usage
Start evaluating and tuning right away via `main.py`.

### Basic Execution
```bash
# Evaluate single class (e.g., foam)
python main.py --categories foam

# Evaluate multiple classes and trigger visually plotted output
python main.py --categories cable_gland dowel foam tire rope --visualize
```

### Argument Flags
* `--categories`: Specific MVTec 3D-AD subjects. Can supply multiple choices space delimited.
* `--visualize`: Outputs robust anomaly heatmaps against RGB backgrounds internally saved to `./visualizations/`.
* `--raw_data_root`: Main dataset location (default: `./data/MVTec3D-AD`).
* `--batch_size`: Batch size used for dataloaders during training and testing (default: `4`).
* `--save_model`: Saves feature banks and model checkpoints inside `checkpoints/`.
* `--load_model`: Skips memory bank building/training and evaluates existing weights straight from `checkpoints/`.

## 📄 Dataset & Citation
Real industrial settings rely heavily on uncontrolled unlabelled data, normally only featuring `good` quality pieces mapping.
You can acquire the dataset manually here: [Dataset Download Portal](https://www.mvtec.com/company/research/datasets/mvtec-3d-ad).
If the approach or the MVTec 3D-AD format serves your research intent, kindly quote the original authors as below:
```bibtex
@inproceedings{mvtec3dad,
  title={The MVTec 3D-AD Dataset for Unsupervised 3D Anomaly Detection and Localization},
  author={Bergmann, Paul and Jin, Xin and Abati, Davide and Grigg, Andreas and Leonhardt, Jan-Hendrik and Schmidt, Maximilian and Zeller, Michael and Hashash, Fadi and Steger, Carsten},
  booktitle={Proceedings of the 17th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications - Volume 4: VISAPP},
  pages={202--213},
  year={2022}
}
```

## ⚖️ License
This project operates under the standard [MIT License](LICENSE). Open for research analysis and independent forks!