// Document format
#set text(size: 10pt)
#set page(
  paper: "us-letter",
  header: align(right)[
    A fluid dynamic model for
    glacier flow
  ],
  numbering: "1",
)
#set par(justify: true)
#set heading(numbering: "1.1")

// Title
#let title = [
  Semantic Segmentation of Heritage Buildings with Deep Learning
]
#let subtitle = [
  A Low-Rank Adaptation of the Pointcept PTv3 Point Transformer with HBIM Data
]
#let date=datetime.today()
#set page(
  header: align(
    right + horizon,
    title
  ),
)
#align(center, text(15pt)[
  *#title*,
])
#align(center, text(12pt)[
  *#subtitle*
  *#date.display()*
])
#grid(
  columns: (1fr, 1fr),
  row-gutter: 12pt,
  align(center)[
    Dr. Liam Moore \
    #link("mailto:liam@chiark.io")
  ],
  align(center)[
    Dr. Stephen Ogilvy \
    #link("mailto:steve@chiark.io")
  ]
)
#align(center)[
  #set par(justify: false)
  *Abstract* \
  #lorem(80)
]
#v(15pt) 

// Document body
#show: rest => columns(2, rest)
#outline(
  title: none,
  depth:2,
  indent:auto
  )

= Introduction
3D scanning technologies, such as LiDAR and photogrammetry, are transforming heritage preservation by enabling the capture of highly detailed digital models of historical sites and artifacts. These precise scans allow for accurate documentation, analysis, and virtual restoration, ensuring that cultural heritage is preserved and accessible for future study even if the physical structures degrade or are damaged over time.

AI technologies are becoming essential tools in heritage preservation by automating the processing and analysis of vast amounts of 3D data, such as point clouds generated from LiDAR or photogrammetry. Through techniques like semantic segmentation and pattern recognition, AI can help identify and classify architectural elements, detect structural damage, and even predict future degradation. This reduces manual effort, speeds up analysis, and enables large-scale digital documentation of heritage sites, ultimately enhancing the ability to preserve and restore cultural assets efficiently and accurately.

== Previous Work
Drawing on previous classification work performed on Milan Cathedral @teruggi2020, where a hierarchical machine learning approach was used with Random Forest algorithms, the authors conducted a previous experiment using a 1.2B point dataset captured by LiDAR scan of a heritage building (the Queens House villa in Greenwich).

Hierarchical classification was employed to manage the computational challenges posed by the dataset's scale. The approach divided the dataset into smaller, tractable subproblems by combining spatial subsampling and hierarchical classification. Multiple Random Forest models were trained in tandem, with each model responsible for classifying at different levels of the label hierarchy. While this method showed promise, several key limitations emerged, primarily due to the fixed coupling of semantic classes to rigid spatial resolutions. The optimal scales for classification varied significantly across classes, with multi-scale features likely being more effective for distinguishing elements with class-dependent spatial characteristics.

The experiment's findings revealed three major issues: geometric feature neighborhood values were not well-matched to label hierarchy levels, fixed-scale features imposed restrictive assumptions, and the presence of too-similar classes within levels hindered classifier performance.

== Deep Learning Approach
The Point Transformer models @zhao2021pointtransformer (can offer a more robust solution by inherently encoding multi-scale information through its serialized attention mechanism. Unlike Random Forests, which rely on pre-defined feature sets and spatial resolutions, the Point Transformer dynamically learns to attend to relevant spatial relationships across a wider perceptive field, with up to 1024 points. This enables the model to capture both fine-grained local features and broader structural context simultaneously, effectively addressing the need for multi-scale feature representation. By leveraging self-attention across large point neighborhoods, the Point Transformer allows for more nuanced and flexible classification without the rigid constraints of hierarchical systems, making it better suited to the complexity and scale of point cloud data in heritage preservation tasks.

The current state-of-the-art in this field is the Point Transformer v3 @wu2024ptv3, or PTv3.
This replaces precise neighbor search (as used in previous iterations like PTv2) with a more efficient KNN-based approach, allowing for substantial improvements in processing speed — up to 3x faster — and memory efficiency, with a 10x reduction in memory usage. This allows the model to handle much larger point clouds while maintaining state-of-the-art accuracy. Enhanced further with multi-dataset joint training, PTv3 achieves leading results across over 20 downstream tasks in both indoor and outdoor environments, demonstrating its robustness and versatility in large-scale 3D representation learning.

PTv3 is designed to scale efficiently while maintaining strong performance, making it ideal for large datasets such as those in heritage preservation, where both fine detail and broad context are crucial for accurate semantic segmentation.

This document details an adaptation of an existing Point Transformer v3 (PTv3) model, originally trained on the ScanNet @dai2017scannet, Structured3D @zheng2019structured3d, and S3DIS @armeni2016s3dis datasets, to incorporate a low-rank adaptation using synthetic Heritage Building Information Modeling (HBIM) data.

= Low-Rank Adaptation (LoRA)
Low-rank Adaptation (LoRA) is a modern method for fine-tuning neural networks on new data. By introducing so-called adapter weights throughout a pretrained network and training these while leaving the original network intact, it occupies a middle ground between the traditional approach of training a new classification head on top of the original network and full fine-tuning of all network parameters @hu2021lora. LoRA and its variants have seen near-universal adoption as the go-to fine-tuning method across the fields of image generation and language modelling in recent times. This is largely owing to its strong performance on downstream tasks: for a given task and training dataset, LoRA models often lag only a few percent behind full fine-tuning while training only a small fraction of the parameters @hu2021lora. The ability to control the trainable parameter count through both the rank hyperparameter $r$ and the layers to which LoRA adapters are applied make it intrinsically flexible, usable in both data and compute constrained environments.

== Core Mechanism
LoRA achieves its efficiency by decomposing the weight updates into low-rank matrices. Specifically, for a given layer with weight matrix $W$, LoRA introduces two matrices $A$ and $B$, such that the effective weight becomes $W + A B^T$. The dimensions of $A$ and $B$ are chosen to ensure that their product has the same shape as $W$, while their inner dimension $r$ (the rank) is typically much smaller than the original dimensions @hu2021lora. This low-rank structure significantly reduces the number of trainable parameters while still allowing for meaningful updates to the network's behavior.

The mathematical formulation of LoRA can be expressed as:

$ h = W x + (alpha / r) A B^T x $

Where $h$ is the layer output, $x$ is the input, $W$ is the original weight matrix, $A B^T$ represents the LoRA update, $alpha$ is a scaling factor, and $r$ is the rank @hu2021lora.

Where traditional fine-tuning allocates new parameters to learning a new linear classifier on top of fixed latent representations, the generic formulation of LoRA weight update matrices permits that they be inserted anywhere in a network, including within specialised layers. This ability to adapt the internal representations themselves, as in full fine-tuning, lends it its expressive power as a domain adaptation technique.

== Key Parameters and Considerations
=== Alpha Parameter
The alpha ($alpha$) parameter in LoRA is a scaling factor that controls the magnitude of the LoRA update. It allows for finer control over the contribution of the LoRA update relative to the original weights. A larger alpha increases the impact of the adaptation, while a smaller alpha reduces it @hu2021lora.

=== Rank Selection
The choice of rank $r$ is a key hyperparameter in LoRA:

1. Low rank (e.g., $r = 1, 2, 4$):
   - Suitable for minor adaptations or when computational resources are severely constrained.
   - Ideal when the target task is closely related to the pre-training domain.

2. High rank (e.g., $r = 16, 32, 64$):
   - Appropriate for significant domain shifts or complex adaptation tasks.
   - Provides more expressiveness, potentially approaching full fine-tuning performance.

The optimal rank often depends on the specific task, dataset size, and base model architecture. Empirical studies have shown that performance often saturates at relatively low ranks (e.g., $r = 16$ or $32$) for many tasks @hu2021lora.

== Application in Complex Network Architectures
LoRA can be applied to various types of layers in complex neural network architectures. Within the framework of the PTv3 + PPT architecture, we specifically target:

1. Transformer Blocks:
   - Query (Q), key (K), and value (V) projection matrices in self-attention layers.
   - The output projection matrix of the self-attention layer.
   - Up-projection and down-projection matrices in feed-forward networks (FFN) @hu2021lora.

2. Sparse 3D convolutional layers:
   - For 3D convolutions, the 5D weight tensor is reshaped into a 2D matrix before applying LoRA.

3. Embedding Layers:
   - Applied to token embeddings and CLIP embedding adapters in the PPT point-text cross-encoder.

== Advantages and Recent Developments
One of the key advantages of LoRA is its modularity. Multiple LoRA adapters can be trained independently on different tasks or datasets and then combined or switched dynamically at inference time. This property enables efficient multi-task learning and domain adaptation without the need for storing multiple copies of the full model @hu2021lora.

Recent research has explored variations of LoRA, such as QLoRA (Quantized LoRA), which further reduces memory requirements by using quantization techniques @dettmers2023qlora, and AdaLoRA, which adaptively adjusts the rank during training @zhang2023adalora. These developments continue to push the boundaries of efficient fine-tuning, making it possible to adapt large language models and diffusion models on consumer-grade hardware.

== Conclusion
LoRA represents a significant advancement in the field of transfer learning, offering a compelling balance between performance, efficiency, and flexibility. Its widespread adoption in both academic and industrial settings underscores its importance in the current landscape of deep learning research and applications @hu2021lora.

= Site Data
Heritage Building Information Modeling (HBIM) data consists of detailed, structured 3D models of historical buildings that capture both their geometric features and semantic information. It includes precise representations of architectural elements like walls, columns, windows, and other building components, often derived from laser scans, photogrammetry, and archival records. HBIM data not only provides an accurate digital replica of the building's structure but also embeds relevant historical and construction details, making it a valuable resource for conservation, restoration, and analysis of heritage sites.

Using HBIM mesh data to generate point clouds for semantic segmentation training offers a promising approach for heritage preservation tasks.
This method allows for flexible control over point cloud density and resolution, ensuring consistent labeling across architectural elements.

A pipeline to process the mesh data to produce point clouds was constructed using The Visualisation Toolkit (VTK) @vtk, an open-source and state-of-the-art software system for 3D computer graphics, image processing, and scientific visualization.
VTK is widely used in research and industry for creating high-quality visualizations and for developing applications across fields like medical imaging, computational fluid dynamics, and geospatial analysis.
This allows for fold construction (the process of creating distinct training and testing datasets) to be simplified, and for the point cloud generation to be fully automated compared.
This represents a significant improvement in flexibility and automation compared to the previous CloudCompare @cloudcompare approach, which involved significant manual data manipulation.

== Sites
HBIM site data was used for a number of heritage buildings in the adjoining UNESCO World Heritage Site of Greenwich Maritime in Greenwich, London:
- the National Maritime Museum
- Park Row (an adjunct building of the above housing parking facilities)
- the Royal Observatory
The Royal Observatory was split into two sites, the northern site containing Flamsteed House, and the Southern site containing the Peter Harrison Planetarium and the Altazimuth Pavilion.

Additionally, site data from the Brass Foundry, a National Heritage List for England (NHLE) listed building at the Royal Arsenal site also in Greenwich, was used.

For each site, surrounding landscaping was included in the samples to capture key environmental and spatial contexts around the structures, including courtyards, pathways, and open spaces.

= Fold Allocation
Training, evaluation, and testing datasets serve distinct purposes in the model development pipeline. The training fold is used to train the model, the evaluation fold is periodically run during training to monitor performance and prevent over-fitting on the training sample, and the testing fold is reserved for assessing the final model's performance on unseen data, providing a true measure of its generalization ability.

Constructing training, testing, and evaluation folds for semantic segmentation tasks requires careful consideration to ensure that each fold contains a representative distribution of the different categories present in the dataset. In the case of 3D point clouds, this is particularly challenging, as it is essential that the folds not only balance the number of points from each category but also capture spatially meaningful scenes. Manually constructing these folds is extremely difficult and time-consuming due to the complexity and scale of the data, as well as the need to preserve both category balance and spatial contiguity.

== Automated site partitioning
To address this, an automated fold allocation algorithm was developed to create well-balanced regions with the required overall share of each site, creating spatially contiguous regions for which each category in the taxonomy is well-represented.

The algorithm beings by generating coarse point clouds of the sample, and then simplifying the data by collapsing it into the x-y plane. This ensures that each fold spans the full vertical height of the scene, preserving critical contextual information related to verticality.

Next, the data is voxelized into 6m x 6m bins, and the process of creating folds begins with an iterative algorithm:
- First, a set of seed cells is selected based on the number of required folds and subregions within each fold.
- A priority system is then employed, taking into account the overall population target for each fold, category representation, and geometric metrics such as compactness and aspect ratio. This system ensures that the regions grown from the seed cells remain contiguous and satisfy constraints related to category balance and region shape.
- The algorithm iteratively grows these seed cells by selecting neighboring cells according to the priority system, dynamically adjusting each region as new cells are added.
- Once all cells have been assigned to a fold, the algorithm evaluates an equality score for the configuration. The score penalizes deviations from the desired population distribution and under-represented categories, ensuring that the allocation closely matches predefined weights.

The configuration with the best equality score is selected, and this schematic is used to crop the input meshes into distinct training, evaluation, and testing folds, maintaining both spatial contiguity and meaningful category representation throughout the dataset.

== Allocations


= Experiments and Results

= Summary and Future Work
TODO

#bibliography("bibliography.bib")
