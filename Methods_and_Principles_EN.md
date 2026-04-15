# Core Methods and Technical Principles: 3D Defect Detection

This project aims to solve the fully unsupervised defect detection problem in industrial production, where "only good items are available for training". Targeting the multi-modal data (2D RGB images + 3D XYZ depth coordinates) provided by the MVTec 3D-AD dataset, we employed the following core methods and deeply analyzed the scientific principles and rationale behind their selection.

---

## 1. Overall Challenges and Solutions

### Challenges
Industrial defect detection usually faces two extremes:
- **Unsupervised Learning Constraint**: On a real production line, it is extremely costly or impossible to collect all kinds of rare defects (scratches, holes, deformations). Our model must learn to identify "abnormal" purely by seeing "normal" samples.
- **Extreme Divergence in Material Morphology**: The test items include rigid bodies with fixed shapes and relatively smooth surfaces (e.g., `dowel`), materials whose surfaces consist of unordered porous structures (e.g., `foam`), and objects with high-frequency periodic textures and easily deformable shapes (e.g., `tire`, `rope`).

### The Solution: Strategy Router
There is no single set of hyperparameters or even a single model that can perfectly detect all materials. Therefore, we built a **Strategy Router intelligent routing system**. Based on the physical feature distribution of different objects, it dynamically switches between two highly differentiated and complementary algorithm architectures: the **AST Reverse Distillation Baseline** and the **Spatial PatchCore**, assisted by customized 3D feature processing (e.g., 3D Surface Normals).

---

## 2. Core Algorithm 1: AST (Reverse Distillation Baseline)

### Principle Concept
The AST (Asymmetric Student-Teacher) network is an image anomaly detection algorithm based on a "teacher-student" framework.
1. **Teacher Network**: A network pre-trained and **frozen** on natural images (like ImageNet). We feed it normal RGB and 3D features, and it outputs a specific, stable high-dimensional feature manifold.
2. **Student Network**: A network with a "reverse distillation (decoder with inverted bottlenecks)" architecture. Its task is to learn how to **reconstruct and mimic** the Teacher's outputs exclusively from these high-dimensional features.
3. **Detection Logic**: Since the student network **only sees** features of normal samples during training, when it encounters defective samples (with breakage or cuts) during testing, it becomes "confused" and cannot accurately reconstruct the features of the defective area. By calculating the cosine or mean squared distance difference between the Teacher and Student outputs, the "anomaly score" of that pixel can be directly obtained.

### Why AST?
This method works perfectly for rigid structures with **fixed structures and low self-variance**. Take the wooden `dowel` for example, because all its normal samples are basically identical, the Student network can learn a very narrow and precise manifold space limit. Testing shows that this model easily reaches **0.970+ Sample-level AUROC / 0.996+ Point-level AP** on the `dowel`. It is the undisputed baseline rule.

---

## 3. Core Algorithm 2: Spatial PatchCore (3D Enhanced Memory Bank)

### Principle Concept
PatchCore abandons the neural-network reconstruction concept which is prone to "generalization blur", and instead uses "pure brute-force yet elegant" KNN feature memory retrieval:
1. **Building the Memory Bank**: A pre-trained ResNet is used to extract local feature vectors (Patch features) of all training images (good products) under multiple receptive fields. Through greedy subsampling (Coreset Subsampling), tens of thousands of the most representative feature points are selected and flattened into a massive memory bank.
2. **Spatial Enhancements**: We do not just use RGB features. We inject the corresponding 3D absolute spatial coordinates (XYZ) and 3D surface normals (mapped via the requested `xyz_weight` weighting) entirely into this feature vector.
3. **Distance Inference**: During testing, all feature points of the test image are extracted. We use native PyTorch Matrix cross-multiplication (`cdist`) to rapidly find the "most similar feature point" in the memory bank limit. If the retrieved closest Euclidean distance exceeds a threshold, it proves that this local feature (flaw) has never appeared before.

### Why Spatial PatchCore?
For `foam` (sponge holes) and `tire` (high-frequency tread patterns), the AST student network easily and forcibly restores those "anomalous high-frequency flaws" as various reasonable tiny holes, leading to missed detections!
However, PatchCore directly compares local feature point libraries: as long as there is an unusual cut mark, no matter where it is on the image, as long as it cannot find a twin sibling in all the features of the spatial database, it will be caught. It has an immense advantage over objects with extremely irregular edges, periodic textures, and completely unfixed topology locations.

---

## 4. Key Technological Breakthrough: 3D Surface Normals

### Principle Concept
Using the 2D depth map layout, we extract the XYZ tensor space coordinates for the image. Utilizing two sets of 3D convolution kernels (based on the Sobel filter concept), we calculate the horizontal gradient $U$ and vertical gradient $V$ in the 2D plane based on the $Z$ depth value matrix. Applying a continuous **Cross Product** on these two gradients, followed by $L_2$ normalization calculation, allows us to obtain the 3D Surface Normals of the entire item mapped natively at high speeds efficiently within the GPU.

### Why use Surface Normals?
Usually, 3D point cloud matching relies heavily on absolute topology coordinates (e.g., Z=50 mm). But this has a fatal flaw: if a tire on the conveyor belt tilts, or a rope is bent into a different arc due to its flexibility, the absolute depth coordinates will dramatically change, and the model will mistakenly think this represents a massive discrepancy defect!
However, surface normals measure **relative curvature limit and slope**. Even if a rope is placed in a different location, the normal curvature of the local cylindrical surface topology remains the same. The normals will only experience a high-frequency cliff-like anomaly drop when the surface encounters **cuts or protrusions/bumps**.
It is precisely by introducing this feature that is totally unaffected by positional translation bounds, while actively **turning off absolute coordinate weights (`xyz_weight = 0.0`)**, that we elevated the capacity of extremely hard-to-train high-frequency flexible items like `tire` and `rope` effectively to an industrial production tier capability index grade.

---

## 5. Project Engineering Consideration Notes
* **Refusing Faiss, Adopting Native PyTorch CDist**: Faiss is an extremely powerful vector retrieval library. However, under adequate VRAM environments (through batched block comparisons), operations natively written with `torch.cdist` eliminate cumbersome dependency errors caused between diverse `faiss-gpu` and `cudatoolkit` setups, while rendering memory usage completely transparent and controllable to developers.
* **Gaussian Heatmap Aggregation (`blur_radius` & `top_k`)**: A single point's anomaly score is not necessarily reliable naturally (sensor speckle noise, unaligned shadows). We smoothed the entire prediction response matrix through an image-processing algorithmic Gaussian Blur kernel filter, whilst ensuring that when judging the final Sample-level score pooling rule computation, only the hottest specific pixels limit array values are pooled to determine severity limits accuracy correctly mapping metrics bounds seamlessly.
