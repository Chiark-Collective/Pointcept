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

#set table(
  stroke: none,
  gutter: 0.1em,
  fill: (x, y) =>
    if x == 0 or y == 0 { black } else { none },
  inset: (left: 0.5em, right: 0.5em),
)
#show table.cell: it => {
  if it.x == 0 or it.y == 0 {
    set text(white)
    strong(it)
  } else {
    it
  }
}

// Document body
// #show: rest => columns(2, rest)
// #set page(columns: 2) //, height: 150pt)

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

== Real Sites
HBIM site data was used for a number of heritage buildings in the adjoining UNESCO World Heritage Site of Greenwich Maritime in Greenwich, London:
- the National Maritime Museum
- Park Row (an adjunct building of the above housing parking facilities)
- the Royal Observatory
The Royal Observatory was split into two sites, the northern site containing Flamsteed House, and the Southern site containing the Peter Harrison Planetarium and the Altazimuth Pavilion.

Additionally, site data from the Brass Foundry, a National Heritage List for England (NHLE) listed building at the Royal Arsenal site also in Greenwich, was used.

For each site, surrounding landscaping was included in the samples to capture key environmental and spatial contexts around the structures, including courtyards, pathways, and open spaces.
// #set page(columns: 1)
#figure(
  image("figs/meshes_brass_foundry.png", width: 100%),
  caption: [Mesh data for the Brass Foundry site.],
  outlined: true,
)
#figure(
  image("figs/meshes_park_row.png", width: 100%),
  caption: [Mesh data for the Park Row site.],
  outlined: true,
)
#figure(
  image("figs/meshes_maritime_museum.png", width: 100%),
  caption: [Mesh data for the Maritime Museum site.],
  outlined: true,
)
#figure(
  image("figs/meshes_rog_north.png", width: 100%),
  caption: [Mesh data for the Royal Observatory Greenwich North site.],
  outlined: true,
)
#figure(
  image("figs/meshes_rog_south.png", width: 100%),
  caption: [Mesh data for the Royal Observatory Greenwich South site.],
  outlined: true,
)

#pagebreak()
== HBIM Library Data
A library of different HBIM components for our taxonomy, featuring various architectural elements in multiple forms, sizes, and colors, was utilized to augment the training dataset. This highly synthetic data allowed for an investigation into whether the inclusion of diverse building elements would enhance the model's ability to generalize to real-world heritage sites for semantic segmentation. By incorporating these well-structured components, the model's performance on actual site data could be evaluated more systematically.

A concern raised in this approach is that stripping the building elements of their surrounding spatial context - such as neighboring structures or environmental details - could negatively impact segmentation performance. Without this contextual information, the model may find it challenging to accurately segment scenes where spatial relationships between elements play a crucial role. This investigation examined whether the synthetic data, while beneficial for certain architectural details, could introduce limitations due to the absence of full scene context.

#figure(
  image("figs/library_raw.png", width: 95%),
  caption: [A subsection of the Library mesh data, showing a selection of stairs and railings.],
  outlined: true,
  placement: auto,
  gap:1em,
) <library_raw>

= Fold Allocation
Training, evaluation, and testing datasets serve distinct purposes in the model development pipeline. The training fold is used to train the model, the evaluation fold is periodically run during training to monitor performance and prevent over-fitting on the training sample, and the testing fold is reserved for assessing the final model's performance on unseen data, providing a true measure of its generalization ability.

Constructing training, testing, and evaluation folds for semantic segmentation tasks requires careful consideration to ensure that each fold contains a representative distribution of the different categories present in the dataset. In the case of 3D point clouds, this is particularly challenging, as it is essential that the folds not only balance the number of points from each category but also capture spatially meaningful scenes. Manually constructing these folds is extremely difficult and time-consuming due to the complexity and scale of the data, as well as the need to preserve both category balance and spatial contiguity.
This also ensures that regardless of the sampling resolution of the resulting point clouds, the samples remain distinct and mutually exclusive.

== Automated site partitioning
To address this, an automated fold allocation algorithm was developed to create well-balanced regions with the required overall share of each site, creating spatially contiguous regions for which each category in the taxonomy is well-represented.

The algorithm beings by generating coarse point clouds of the sample, and then simplifying the data by collapsing it into the x-y plane. This ensures that each fold spans the full vertical height of the scene, preserving critical contextual information related to verticality.

Next, the data is voxelized into 6m x 6m bins, and the process of creating folds begins with an iterative algorithm:
- First, a set of seed cells is selected based on the number of required folds and subregions within each fold.
- A priority system is then employed, taking into account the overall population target for each fold, category representation, and geometric metrics such as compactness and aspect ratio. This system ensures that the regions grown from the seed cells remain contiguous and satisfy constraints related to category balance and region shape.
- The algorithm iteratively grows these seed cells by selecting neighboring cells according to the priority system, dynamically adjusting each region as new cells are added.
- Once all cells have been assigned to a fold, the algorithm evaluates an equality score for the configuration. The score penalizes deviations from the desired population distribution and under-represented categories, ensuring that the allocation closely matches predefined weights.

The configuration with the best equality score is selected, and this schematic is used to crop the input meshes into distinct training, evaluation, and testing folds, maintaining both spatial contiguity and meaningful category representation throughout the dataset.

We include some example fold allocation schematics for Maritime Museum and ROG South in @maritime_fold_schematic and @rog_south_fold_schematic.
The plots are coloured by fold region (1=train, 2=test, 3=evalution) with multiple subregions supported for the testing and evaluation folds.
Each region denotes its seed cell with a red star.
Each cell within each region is colour-graded to show the order it was added to the fold, from darkest to lightest.
Each cell also contains a box denoting the global order in which the cell was allocated. 

Renders for these fold allocations are shown in @maritime_folds_3d and @rog_south_folds_3d.
The corresponding per-category population breakdown across fold and subregion for these two examples are also summarised in @maritime_category_table and @rog_south_category_table.


#figure(
  image("figs/fold_allocation_schematic_maritime_museum.png", width: 100%),
  caption: [Mesh partitioning schematic for the Maritime Museum site.],
  outlined: false,
  placement: auto,
  gap: 0em,
) <maritime_fold_schematic>

#figure(
  image("figs/maritime_folds_3d.png", width: 100%),
  caption: [3D render of the Maritime Museum fold allocation.],
  outlined: false,
  placement: auto,
  gap: 0em,
) <maritime_folds_3d>


#figure(
table(
  columns: 8,
  // Header row
  [], [Fold 1], [Fold 2], [Region 2], [Region 3], [Fold 3], [Region 4], [Region 5],
  // Data rows
  [1_WALL], [58.6%], [20.9%], [8.6%], [12.3%], [20.6%], [12.6%], [7.9%],
  [2_FLOOR], [65.6%], [20.2%], [8.3%], [12.0%], [14.2%], [9.8%], [4.4%],
  [3_ROOF], [55.7%], [25.5%], [15.2%], [10.3%], [18.9%], [9.5%], [9.4%],
  [4_CEILING], [62.3%], [21.4%], [9.1%], [12.4%], [16.3%], [9.6%], [6.7%],
  [5_FOOTPATH], [59.3%], [17.5%], [5.7%], [11.8%], [23.2%], [12.1%], [11.1%],
  [6_GRASS], [45.8%], [20.5%], [11.1%], [9.4%], [33.7%], [14.9%], [18.8%],
  [7_COLUMN], [44.9%], [18.1%], [11.7%], [6.5%], [37.0%], [19.3%], [17.7%],
  [8_DOOR], [52.8%], [23.4%], [7.5%], [15.9%], [23.8%], [15.5%], [8.4%],
  [9_WINDOW], [47.7%], [29.6%], [10.7%], [18.9%], [22.7%], [13.4%], [9.3%],
  [10_STAIR], [62.2%], [22.5%], [10.2%], [12.3%], [15.3%], [9.8%], [5.5%],
  [11_RAILING], [66.2%], [16.8%], [4.4%], [12.4%], [17.0%], [13.2%], [3.8%],
  [13_OTHER], [58.2%], [23.9%], [13.1%], [10.8%], [17.9%], [9.8%], [8.2%],
),
caption: [Per-category population allocation per-fold and per-subregion for Maritime Museum.],
placement: auto,
) <maritime_category_table>


#set table(
  stroke: none,
  gutter: 0.1em,
  fill: (x, y) =>
    if x == 0 or y == 0 { black } else { none },
  inset: (left: 0.5em, right: 0.5em),
)
#show table.cell: it => {
  if it.x == 0 or it.y == 0 {
    set text(white)
    strong(it)
  } else {
    it
  }
}

#figure(
  image("figs/fold_allocation_schematic_rog_south.png", width: 100%),
  caption: [Mesh partitioning schematic for the Royal Observatory South site.],
  outlined: true,
  placement: auto,
  gap: 0em,
) <rog_south_fold_schematic>

#figure(
  image("figs/rog_south_folds_3d.png", width: 100%),
  caption: [3D render of the Royal Observatory South fold allocation.],
  outlined: false,
  placement: auto,
  gap: 0em,
) <rog_south_folds_3d>


#figure(
  table(
    columns: 8,
    // Header row
    [], [Fold 1], [Fold 2], [Region 2], [Region 3], [Fold 3], [Region 4], [Region 5],
    // Data rows
    [1_WALL], [46.6%], [25.7%], [14.9%], [10.8%], [27.7%], [11.0%], [16.7%],
    [2_FLOOR], [49.5%], [20.5%], [13.7%], [6.9%], [30.0%], [9.0%], [21.0%],
    [3_ROOF], [35.3%], [22.2%], [19.0%], [3.2%], [42.5%], [19.4%], [23.1%],
    [4_CEILING], [62.2%], [17.9%], [12.1%], [5.8%], [20.0%], [6.9%], [13.1%],
    [5_FOOTPATH], [45.6%], [17.5%], [10.3%], [7.3%], [36.9%], [20.8%], [16.1%],
    [7_COLUMN], [26.0%], [27.2%], [18.0%], [9.3%], [46.8%], [27.4%], [19.4%],
    [8_DOOR], [48.2%], [21.7%], [14.4%], [7.3%], [30.1%], [3.9%], [26.2%],
    [9_WINDOW], [35.9%], [29.5%], [20.3%], [9.2%], [34.6%], [7.5%], [27.1%],
    [10_STAIR], [46.4%], [44.0%], [23.0%], [20.9%], [9.7%], [6.6%], [3.1%],
    [11_RAILING], [39.0%], [37.0%], [14.9%], [22.1%], [24.0%], [15.1%], [8.9%],
    [12_RWP], [39.9%], [30.2%], [18.8%], [11.4%], [29.9%], [6.8%], [23.1%],
    [13_OTHER], [50.0%], [23.2%], [17.3%], [5.9%], [26.8%], [7.6%], [19.3%],
  ),
  caption: [Per-category population allocation per-fold and per-subregion for ROG South.],
  placement: auto,
) <rog_south_category_table>

#pagebreak()
== Library Data Scene Construction
An issue with the raw library data is that each category of HBIM components is physically separated, often by considerable distances.
This results in an over-clustering of objects within the same category, causing the network to potentially overfit by learning to group proximate objects too strongly in the classification.
Additionally, the isolation of each sample means that the network's receptive field would predominantly encounter only one category at a time, which can lead to significant issues with stability and convergence during training, as the model lacks exposure to diverse category interactions within the same scene.

To mitigate this, the library component meshes were divided into small 2.5m² cells, which were then randomly sorted per category. From this set, 15% of the cells were allocated to the evaluation sample, 20% to the testing sample, and 65% to the training sample. To further ensure variability, each sample was randomly shuffled, and the cells were recombined in a spiral pattern to construct more compact and diverse scenes.
The resulting training set is shown in @library_scene.

#figure(
  image("figs/library_scene.jpg", width: 110%),
  caption: [The recombined library scene for the training fold.],
  outlined: false,
  placement: none,
  gap: 1em,
) <library_scene>

== Taxonomy

The previous project highlighted several challenges that arise when using a taxonomy that is too finely segmented. Overly detailed class distinctions led to difficulties in classification, as certain categories became too similar to differentiate effectively. This fine segmentation not only increased the complexity of the model but also introduced issues of class imbalance, where some highly specific categories had insufficient representation. The segmentation of similar elements at high levels of granularity resulted in confusion and poor performance in those classes. These challenges motivated the decision to adopt a simpler, more generalized taxonomy, reducing ambiguity between categories and improving overall model stability and performance. 

The new taxonomy is as follows:
1. *Wall*
2. *Floor*
3. *Roof*
4. *Ceiling*
5. *Footpath*
6. *Grass*
7. *Column*
8. *Door*
9. *Window*
10. *Stair*
11. *Railing*
12. *Rainwater Pipe*
13. *Other* - This category includes miscellaneous elements that do not fit into the primary architectural classes. It handles objects or features that are not consistently represented or easily categorized, ensuring all data is included.

While the taxonomy has been simplified, Rainwater Pipe remains a particularly sparse category in the input data, which could potentially lead to poor performance due to its limited representation.
Similarly, the Other category is still quite broad, covering a diverse array of features found in heritage sites.
This could introduce variability and complexity, with a great many different architectural and structural elements grouped together under a single label. 

// #pagebreak()
= Experimental Method with Pointcept
TODO: this section will explain the experimental setup that was used across the different site configurations.

== Training and Evaluation Phase
TODO: include train-time transforms, grid sampling, sphere cropping transforms etc

== Testing Phase
TODO

= Results
Three primary experiments were carried out with different sites used in the training and testing. These combinations were:
- Library scene alone
- Park Row and Maritime Museum
- Brass Foundry and both Royal Observatory sites
The more performant of the real-site experiments then had its training re-run, but with the library scene included in the training phase.

== Library Scene

#figure(
  table(
    columns: 4,
    // Header row
    [Class Index], [Category Name], [IoU (%)], [Accuracy (%)],
    // Data rows
    [-], [All], [82.0 (mean)], [88.9 (mean), 92.4 (overall)],
    [1], [Wall], [98.6], [98.8],
    [2], [Floor], [29.3], [39.9],
    [3], [Roof], [99.9], [100.0],
    [4], [Ceiling], [92.8], [98.7],
    [5], [Footpath], [33.1], [56.6],
    [6], [Grass], [98.1], [98.8],
    [7], [Column], [98.1], [98.9],
    [8], [Door], [81.1], [85.4],
    [9], [Window], [72.4], [94.6],
    [10], [Stair], [93.9], [97.0],
    [11], [Railing], [94.4], [97.0],
    [12], [Rainwater Pipe], [81.1], [93.3],
    [13], [Other], [93.4], [96.5],
  ),
  caption: [Overall and per-category IoU and accuracy results for the Park Row, Maritime Museum, and Library test fold.],
  placement: none,
)

== Park Row and Maritime Museum

#figure(
  table(
    columns: 4,
    // Header row
    [Class Index], [Category Name], [IoU (%)], [Accuracy (%)],
    // Data rows
    [-], [All], [62.9 (mean)], [78.9 (mean), 86.2 (overall)],
    [1], [Wall], [80.1], [86.0],
    [2], [Floor], [76.8], [86.6],
    [3], [Roof], [80.2], [82.6],
    [4], [Ceiling], [70.7], [85.7],
    [5], [Footpath], [81.5], [99.1],
    [6], [Grass], [97.5], [99.0],
    [7], [Column], [54.6], [86.5],
    [8], [Door], [43.2], [90.0],
    [9], [Window], [65.3], [82.7],
    [10], [Stair], [70.7], [79.5],
    [11], [Railing], [38.2], [80.4],
    [12], [Rainwater Pipe], [5.4], [9.5],
    [13], [Other], [53.9], [57.8],
  ),
  caption: [Overall and per-category IoU and accuracy results for the Park Row and Maritime Museum test fold.],
  placement: none,
)

== Brass Foundry and Royal Observatory

#figure(
  table(
    columns: 4,
    // Header row
    [Class Index], [Category Name], [IoU (%)], [Accuracy (%)],
    // Data rows
    [-], [All], [61.7 (mean)], [70.1 (mean), 83.5 (overall)],
    [1], [Wall], [81.3], [95.0],
    [2], [Floor], [59.3], [68.3],
    [3], [Roof], [63.4], [77.9],
    [4], [Ceiling], [72.1], [82.7],
    [5], [Footpath], [58.0], [77.2],
    [6], [Grass], [95.9], [96.5],
    [7], [Column], [29.1], [29.5],
    [8], [Door], [68.6], [72.4],
    [9], [Window], [51.5], [57.4],
    [10], [Stair], [46.5], [50.6],
    [11], [Railing], [78.0], [89.5],
    [12], [Rainwater Pipe], [24.7], [38.1],
    [13], [Other], [73.3], [75.5],
  ),
  caption: [Overall and per-category IoU and accuracy results for the Brass Foundry and Royal Observatory test fold.],
  placement: none,
)

== Augmenting Park Row/Maritime with Library Scene

#figure(
  table(
    columns: 4,
    // Header row
    [Class Index], [Category Name], [IoU (%)], [Accuracy (%)],
    // Data rows
    [-], [All], [60.8 (mean)], [74.9 (mean), 86.4 (overall)],
    [1], [Wall], [83.1], [90.9],
    [2], [Floor], [76.9], [84.2],
    [3], [Roof], [77.1], [85.1],
    [4], [Ceiling], [68.8], [82.3],
    [5], [Footpath], [75.8], [99.4],
    [6], [Grass], [96.4], [97.3],
    [7], [Column], [50.6], [59.3],
    [8], [Door], [52.2], [84.4],
    [9], [Window], [64.1], [79.4],
    [10], [Stair], [53.5], [67.8],
    [11], [Railing], [38.5], [75.4],
    [12], [Rainwater Pipe], [8.4], [18.8],
    [13], [Other], [45.3], [48.8],
  ),
  caption: [Overall and per-category IoU and accuracy results for the Park Row, Maritime Museum test fold with training augmentation from the Library scene.],
  placement: none,
)

= Summary and Future Work
TODO

#pagebreak()
#bibliography("bibliography.bib")
