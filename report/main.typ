#import "@preview/fletcher:0.5.2" as fletcher: diagram, node, edge

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
  #set par(justify: true)
  *Abstract* \
  In this study, a low-rank adaptation (LoRA) was applied to a Point Transformer v3 model with Point Prompt Tuning (PPT), pre-trained on ScanNet, S3DIS, and Structured3D datasets, to explore the feasibility of using Heritage Building Information Modeling (HBIM) site data for semantic segmentation on point clouds acquired through real LiDAR.
  Models were trained with an intra-site fold allocation strategy, achieving 83-86% overall accuracy and 62-63% mIoU within the same site.
  Limitations in generalization were observed in both application to HBIM sites the models had not seen, and application to a point cloud of a heritage site acquired through real LiDAR. We identifiy the likely causes as poor colour information in the supplied site models, insufficient diversity and volume of input data to the models, and semantic label noise in the "other" category. We suggest strategies for addressing these issues in future work.
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

AI technologies are becoming essential tools in heritage preservation by automating the processing and analysis of vast amounts of 3D data, such as point clouds generated from LiDAR or photogrammetry. Through techniques like semantic segmentation and object detection, AI can help identify and classify architectural elements, detect structural damage, and even predict future degradation. This reduces manual effort, speeds up analysis, and enables large-scale digital documentation of heritage sites, ultimately enhancing the ability to preserve and restore cultural assets efficiently and accurately.

== Previous Work
Drawing on previous classification work performed on Milan Cathedral @teruggi2020, where a hierarchical machine learning approach was used with Random Forest algorithms, the authors conducted a previous experiment using a 1.2B point dataset captured by LiDAR scan of a heritage building (the Queens House villa in Greenwich).

Hierarchical classification was employed to manage the computational challenges posed by the dataset's scale. The approach divided the dataset into smaller, tractable subproblems by combining spatial subsampling and hierarchical classification. Multiple Random Forest models were trained in tandem, with each model responsible for classifying at different levels of the label hierarchy. While this method showed promise, several key limitations emerged, primarily due to the fixed coupling of semantic classes to rigid spatial resolutions. The optimal scales for classification varied significantly across classes, with multi-scale features likely being more effective for distinguishing elements with class-dependent spatial characteristics.

The experiment's findings revealed three major issues: geometric feature neighborhood values were not well-matched to label hierarchy levels, fixed-scale features imposed restrictive assumptions, and the presence of too-similar classes within levels hindered classifier performance.

== Deep Learning Approach
The Point Transformer models @zhao2021pointtransformer can offer a more robust solution by inherently encoding multi-scale information while learning appropriate internal geometric features. Unlike Random Forests, which rely on pre-defined feature sets and spatial resolutions, the Point Transformer dynamically learns to attend to relevant spatial relationships across a wider perceptive field, with up to 1024 points. This enables the model to capture both fine-grained local features and broader structural context simultaneously, effectively addressing the need for multi-scale feature representation. By leveraging self-attention across large point neighborhoods, the Point Transformer allows for more nuanced and flexible classification without the rigid constraints of hierarchical systems, making it better suited to the complexity and scale of point cloud data in heritage preservation tasks.

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

In summary, LoRA represents a significant advancement in the field of transfer learning, offering a compelling balance between performance, efficiency, and flexibility. Its widespread adoption in both academic and industrial settings underscores its importance in the current landscape of deep learning research and applications @hu2021lora.

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
The corresponding per-category population breakdown across fold and subregions for these two examples are also summarised in accompanying tables.


#figure(
  image("figs/fold_allocation_schematic_maritime_museum.png", width: 100%),
  caption: [Mesh partitioning schematic for the Maritime Museum site.],
  outlined: false,
  placement: none,
  gap: 0em,
) <maritime_fold_schematic>

#figure(
  image("figs/maritime_folds_3d.png", width: 100%),
  caption: [3D render of the Maritime Museum fold allocation.],
  outlined: false,
  placement: none,
  gap: 0em,
) <maritime_folds_3d>


#figure(
  table(
    columns: 4,
    align: left,
    // Header row
    [Category], [Fold 1], [Fold 2], [Fold 3],
    // Data rows
    [1_WALL], [58.6%], [20.9%], [20.6%],
    [2_FLOOR], [65.6%], [20.2%], [14.2%],
    [3_ROOF], [55.7%], [25.5%], [18.9%],
    [4_CEILING], [62.3%], [21.4%], [16.3%],
    [5_FOOTPATH], [59.3%], [17.5%], [23.2%],
    [6_GRASS], [45.8%], [20.5%], [33.7%],
    [7_COLUMN], [44.9%], [18.1%], [37.0%],
    [8_DOOR], [52.8%], [23.4%], [23.8%],
    [9_WINDOW], [47.7%], [29.6%], [22.7%],
    [10_STAIR], [62.2%], [22.5%], [15.3%],
    [11_RAILING], [66.2%], [16.8%], [17.0%],
    [13_OTHER], [58.2%], [23.9%], [17.9%],
  ),
  caption: [Per-category population allocation per fold for Maritime Museum.],
  placement: none,
) <maritime_fold_allocation_table>


#figure(
  table(
    columns: 5,
    align: left,
    // Header row
    [Category], [Fold 2 - Region 2], [Fold 2 - Region 3], [Fold 3 - Region 4], [Fold 3 - Region 5],
    // Data rows
    [1_WALL], [8.6%], [12.3%], [12.6%], [7.9%],
    [2_FLOOR], [8.3%], [12.0%], [9.8%], [4.4%],
    [3_ROOF], [15.2%], [10.3%], [9.5%], [9.4%],
    [4_CEILING], [9.1%], [12.4%], [9.6%], [6.7%],
    [5_FOOTPATH], [5.7%], [11.8%], [12.1%], [11.1%],
    [6_GRASS], [11.1%], [9.4%], [14.9%], [18.8%],
    [7_COLUMN], [11.7%], [6.5%], [19.3%], [17.7%],
    [8_DOOR], [7.5%], [15.9%], [15.5%], [8.4%],
    [9_WINDOW], [10.7%], [18.9%], [13.4%], [9.3%],
    [10_STAIR], [10.2%], [12.3%], [9.8%], [5.5%],
    [11_RAILING], [4.4%], [12.4%], [13.2%], [3.8%],
    [13_OTHER], [13.1%], [10.8%], [9.8%], [8.2%],
  ),
  caption: [Per-category population allocation per subregion for Maritime Museum.],
  placement: none,
) <maritime_subregion_allocation_table>



// #set table(
//   stroke: none,
//   gutter: 0.1em,
//   fill: (x, y) =>
//     if x == 0 or y == 0 { black } else { none },
//   inset: (left: 0.5em, right: 0.5em),
// )
// #show table.cell: it => {
//   if it.x == 0 or it.y == 0 {
//     set text(white)
//     strong(it)
//   } else {
//     it
//   }
// }

#figure(
  image("figs/fold_allocation_schematic_rog_south.png", width: 100%),
  caption: [Mesh partitioning schematic for the Royal Observatory South site.],
  outlined: true,
  placement: none,
  gap: 0em,
) <rog_south_fold_schematic>

#figure(
  image("figs/rog_south_folds_3d.png", width: 100%),
  caption: [3D render of the Royal Observatory South fold allocation.],
  outlined: false,
  placement: none,
  gap: 0em,
) <rog_south_folds_3d>


#figure(
  table(
    columns: 4,
    align: left,
    // Header row
    [Category], [Fold 1], [Fold 2], [Fold 3],
    // Data rows
    [1_WALL], [46.6%], [25.7%], [27.7%],
    [2_FLOOR], [49.5%], [20.5%], [30.0%],
    [3_ROOF], [35.3%], [22.2%], [42.5%],
    [4_CEILING], [62.2%], [17.9%], [20.0%],
    [5_FOOTPATH], [45.6%], [17.5%], [36.9%],
    [7_COLUMN], [26.0%], [27.2%], [46.8%],
    [8_DOOR], [48.2%], [21.7%], [30.1%],
    [9_WINDOW], [35.9%], [29.5%], [34.6%],
    [10_STAIR], [46.4%], [44.0%], [9.7%],
    [11_RAILING], [39.0%], [37.0%], [24.0%],
    [12_RWP], [39.9%], [30.2%], [29.9%],
    [13_OTHER], [50.0%], [23.2%], [26.8%],
  ),
  caption: [Per-category population allocation per fold for ROG South.],
  placement: none,
) <rog_south_category_table>


#figure(
  table(
    columns: 5,
    align: left,
    // Header rows
    [Category], [Fold 2 - Region 2], [Fold 2 - Region 3], [Fold 3 - Region 4], [Fold 3 - Region 5],
    // Data rows
    [1_WALL], [14.9%], [10.8%], [11.0%], [16.7%],
    [2_FLOOR], [13.7%], [6.9%], [9.0%], [21.0%],
    [3_ROOF], [19.0%], [3.2%], [19.4%], [23.1%],
    [4_CEILING], [12.1%], [5.8%], [6.9%], [13.1%],
    [5_FOOTPATH], [10.3%], [7.3%], [20.8%], [16.1%],
    [7_COLUMN], [18.0%], [9.3%], [27.4%], [19.4%],
    [8_DOOR], [14.4%], [7.3%], [3.9%], [26.2%],
    [9_WINDOW], [20.3%], [9.2%], [7.5%], [27.1%],
    [10_STAIR], [23.0%], [20.9%], [6.6%], [3.1%],
    [11_RAILING], [14.9%], [22.1%], [15.1%], [8.9%],
    [12_RWP], [18.8%], [11.4%], [6.8%], [23.1%],
    [13_OTHER], [17.3%], [5.9%], [7.6%], [19.3%],
  ),
  caption: [Per-category population allocation per subregion for ROG South.],
  placement: none,
) <rog_south_subregion_allocation_table>


#pagebreak()
=== Pseudo-code description of algorithm.
```
For iteration = 1 to iterations:
    Initialize an empty grid of size grid_size_y x grid_size_x
    Reset counts for all folds and regions

    // Seed Initialization:
    Select random populated cells as seed cells for each region
    Assign seed cells to their respective regions in the grid
    Update counts and bounding boxes for regions and folds

    // Priority Queue Initialization:
    Initialize a priority queue (min-heap) for cell expansion
    For each seed cell:
        Get its unassigned neighbors
        Compute priority for each neighbor
        Add neighbors to the priority queue

    // Region Growing:
    While the priority queue is not empty:
        Pop the cell with the highest priority
        If the cell is unassigned:
            Assign the cell to the corresponding region
            Update counts and bounding boxes
            Get unassigned neighbors of the cell
            Compute priority for each neighbor
            Add neighbors to the priority queue

    // Assign Unallocated Cells:
    For each unassigned cell in the grid:
        Find the nearest assigned neighbor or closest region
        Assign the cell to that region
        Update counts and bounding boxes

    // Compute Equality Score:
    Compute total penalty based on:
        - Deviation from intended total counts
        - Deviation from intended region sizes
        - Underrepresented categories
        - Region aspect ratios

    // Update Best Configuration:
    If current equality score < best equality score:
        Update best equality score
        Save current grid configuration as best

  Store seed configurations for uniqueness analysis
```

#pagebreak()
== Library Data Scene Construction
An issue with the raw library data is that each category of HBIM components is physically separated, often by considerable distances.
This results in an over-clustering of objects within the same category, causing the network to potentially overfit by learning to group proximate objects too strongly in the classification.
Additionally, the isolation of each sample means that the network's receptive field would predominantly encounter only one category at a time, which can lead to significant issues with stability and convergence during training, as the model lacks exposure to diverse category interactions within the same scene.

To mitigate this, a bespoke algorithm employing the VTK library functionality was deployed to randomly splice and rejoin the library scene to attain a more locally diverse scene.
The library component meshes were divided into small 2.5m² cells, which were then randomly sorted per category.
From this set, 15% of the cells were allocated to the evaluation sample, 20% to the testing sample, and 65% to the training sample. To further ensure variability, each sample was randomly shuffled, and the cells were recombined in a spiral pattern to construct more compact and diverse scenes.
The resulting training set is shown in @library_scene.

#figure(
  image("figs/library_scene.jpg", width: 100%),
  caption: [The recombined library scene for the training fold.],
  outlined: false,
  placement: none,
  gap: 1em,
) <library_scene>

== RGB information
The exported HBIM meshes as supplied were largely lacking in texture information, with only sparse external textures supplied.
This may imply that some textures that were externally linked by the meshes were not included when exported.

The model contains simple RBG colour information, with mesh faces being monochromatic with a simplified colour scheme.
As such, the very limited number of elements with textural information had that information removed so that the model did not use the presence or absence of fine
colour detail to distinguish categories.

== Point Cloud Generation
The input meshes are used to sample point clouds for use as input into the network's training.
In the previous project, manual generation of pointclouds using CloudCompare's user interface was required, necessitating considerable manual work.
This required an extremely fine sampling of points such that voxelisation could be used with PDAL to the required resolution.

This process has been automated with a new pipeline built in VTK, allowing the mesh files exported from the HBIM data to be directly fed into the framework.
The meshes are exported to Stanford File Format, and then fed into the fold allocation algorithm.
The algorithm splits the meshes themselves directly, allowing for quick sampling of training, testing, and evaluation scenes at any required resolution.

The sampling algorithm uses the `vtkPolyDataSampler` to perform an initial sampling of the mesh, creating a relatively uniform output according to the required
sampling density.
The algorithm interpolates the color and normal information from the relevant mesh surfaces and vertices to the samples points.
Some surfaces can be sampled more densely than others, so the clean up the output one of two algorithms can be run to ensure a uniform point distribution:
- `vtkPoissonDiskSampler` - uses a "dart throwing" algorithm that iteratively places points on a surface, ensuring each new point maintains a minimum distance from others, creating a uniform, evenly spaced distribution.
- `vtkVoxelGrid` - subdivides 3D space into a grid of equally sized voxels, then replaces all points within each voxel with a single representative point, creating a uniformly downsampled point cloud.

Given the points are fed into a grid voxelisation filter as part of the network, we chose the simpler `vtkPoissonDiskSampler` to ensure an even point distribution
between mesh faces.

The point clouds generated in this way are ready to be used in the network.
Our pipeline is capable of producing PyTorch state dictionaries containing the point cloud information for direct use as network input, or as `.las` files that
leverage the las format's compression and ease of visualisation in external programs like CloudCompare.
#figure(
diagram(
  node-stroke: 1pt,
  edge-stroke: 1pt,

  // Start Node
  node((0, 0), [CloudCompare .bin files], corner-radius: 2pt),
  edge("-|>"),

  // Extract to .ply format
  node((0, -1), [Extract to .ply format], corner-radius: 2pt),
  edge("-|>"),

  // Fold Allocation
  node((0, -2), [Fold Allocation], corner-radius: 2pt),

  // Edges to Folds
  edge((0, -2), (-1, -3), "-|>", bend: 20deg),
  edge((0, -2), (0, -3), "-|>"),
  edge((0, -2), (1, -3), "-|>", bend: -20deg),

  // Fold Nodes
  node((-1, -3), [Test Fold], corner-radius: 2pt),
  node((0, -3), [Train Fold], corner-radius: 2pt),
  node((1, -3), [Evaluation Fold], corner-radius: 2pt),

  // Edges from Folds to vtkPolyDataSampler
  edge((-1, -3), (0, -4), "-|>", bend: 20deg),
  edge((0, -3), (0, -4), `pointcloud sampling`, "-|>"),
  edge((1, -3), (0, -4), "-|>", bend: -20deg),

  // vtkPolyDataSampler Node
  node((0, -4), [vtkPolyDataSampler], corner-radius: 2pt),
  edge("-|>"),

  // vtkPoissonDiskSampler Node
  node((0, -5), [vtkPoissonDiskSampler], corner-radius: 2pt),

  // Edges to Outputs
  edge((0, -5), (-0.8, -6), "-|>", bend: 20deg),
  edge((0, -5), (0.8, -6), "-|>", bend: -20deg),

  // Output Nodes
  node((-0.8, -6), [.pth format for network ingestion], corner-radius: 2pt),
  node((0.8, -6), [.las format for visualization], corner-radius: 2pt),
),
  placement: none,
  caption: [An illustration of the data ingestion pipeline used for the site data.],
)

// #pagebreak()
= PTv3 with PPT

== Input Variables
The input variables for our point cloud segmentation model include the point coordinates (x, y, z), point normals in each direction (nx, ny, nz), and the RGB color channels.
This streamlined input structure leverages the core geometric and color-based information necessary for segmentation, focusing on features that are universally interpretable across various scenes and capture the essential spatial and color data of each point.

In contrast, the previous Random Forest approach relied on a far broader range of input variables, incorporating numerous hand-crafted features
designed to aid discrimination between classes.
While effective, this approach required extensive feature engineering, and its performance was inherently limited by the quality and relevance of these manually defined variables.

The deep learning approach, however, is not restricted by predefined input variables;
instead, the network learns to create its own pseudo-variables or internal representations that are optimized for segmentation.
Through layers of abstraction, the model identifies patterns and constructs complex features that improve its ability to differentiate between classes, particularly those with subtle or overlapping characteristics.
This flexibility in representation allows deep learning models to achieve higher segmentation accuracy and generalizability, particularly when dealing with complex or large datasets, as it enables the model to learn the most discriminative aspects of the input data autonomously.

== PTv3 backbone

In the interests of completeness we give an overview of the PTv3 architecture described in @wu2024ptv3. 
For a complete picture, we recommend the source publication.

Processing point clouds presents unique challenges compared to structured data like images. 
While images come with an inherent grid structure, point clouds are unordered sets of points in 3D space. 
This unstructured nature has traditionally forced architectures to use computationally expensive 
operations to understand spatial relationships between points, with operations like K-nearest neighbors 
(KNN) consuming up to 28% of forward processing time in previous architectures.

PTv3's key insight is that model performance is more influenced by scale than intricate design details. 
Rather than maintaining strict permutation invariance through complex operations, PTv3 imposes structure 
on point clouds through serialization using space-filling curves. Specifically, it employs four 
patterns: Z-order, Trans Z-order, Hilbert, and Trans Hilbert curves, where the "Trans" variants alter 
the axis traversal order.

The serialization process works by transforming 3D point coordinates into 1D sequence indices:

First, point positions are quantized by dividing by a grid size g and rounding down: ⌊p/g⌋, effectively 
snapping points to a discrete grid. These discrete coordinates are then mapped to a single integer index 
using the space-filling curve mapping function.

Points are then sorted according to these indices, forming an ordered sequence in which they retain 
their original features (RGB values and surface normals).

#figure(
  image("figs/sf_curve_serialisation.png", width: 100%),
  caption: [Illustration of serialisation along the four distinct space-filling curves from Ref. @wu2024ptv3],
  outlined: false,
  placement: none,
  gap: 1em,
) <sf_curve_serialisation>

This process inevitably loses some precise spatial information. For example, two points that are close in 
3D space might end up with quite different sequence indices if they fall on opposite sides of a 
space-filling curve boundary. However, PTv3's use of multiple serialization patterns helps mitigate this: 
points that end up far apart in one pattern's sequence might be closer in another's.

The serialization approach offers a crucial efficiency advantage: it eliminates the need for expensive KNN 
operations by imposing a structured ordering on the points. However, this comes with a potential trade-off 
in spatial relationship accuracy compared to exact neighbour search methods. PTv3's insight is that this 
trade-off becomes negligible when combined with multiple serialization patterns and sufficient scaling of 
the model's receptive field. 

After serialization, points are grouped into non-overlapping patches of 1024 points along the serialized 
order. This large patch size is a key advancement - previous architectures like PTv2 were limited to just 
16 points in their local attention windows due to computational constraints. The efficient serialization 
approach enables this dramatic expansion of the receptive field.

=== Receptive Field and Multi-Scale Processing

The network processes point clouds at multiple scales through its U-Net structure. At each encoder stage, 
grid pooling downsamples the points by a factor of 2, effectively doubling the receptive field. Combined 
with the large 1024-point patches, this means that deeper layers in the network 
can view increasingly large spatial contexts.

This multi-scale approach works in concert with the serialization strategy. While serialization might 
lose some precise local spatial relationships, the hierarchical processing through the U-Net structure 
helps recover and understand spatial patterns at different scales. The four different serialization 
patterns provide different perspectives on these spatial relationships, with the shuffle mechanism 
ensuring the model doesn't become overly reliant on any single pattern.

=== Network Architecture and Data Flow

PTv3 processes point clouds through initialization followed by encoder stages:

1. Initialization

  The input point cloud is first serialized using one of the four patterns (Z-order, Trans Z-order, Hilbert, or Trans Hilbert).
  These space-filling curves map 3D coordinates to 1D sequences while preserving some degree of spatial locality.
  An embedding layer then maps the input features to the initial channel dimension.

2. Encoder Processing

  Grid pooling downsamples points while increasing feature dimensionality. The "Shuffle Orders" operation then randomly 
  varies which serialization pattern will be used for the next block. This variation means points that are separated 
  in one pattern might be grouped together in another, enabling information flow across the point cloud without 
  expensive shift or dilation operations. Points are then processed sequentially by multiple blocks (depths [2,2,6,2] across 
  each of the four encoder stages).

3. Block Structure

  Each encoder block is comprised of the following operations:

  - xCPE (enhanced Conditional Positional Encoding)

    Implemented as a sparse convolution layer with a skip connection. The sparse convolution operates on local neighborhoods 
    defined by the voxel grid. This provides each point with information about its position relative to nearby points.
    Unlike traditional relative positional encoding that requires computing pairwise distances (26% of forward time in PTv2), 
    xCPE achieves similar goals through efficient sparse operations.

  - LayerNorm

    Normalizes features independently for each point, maintaining consistent scales throughout the network 
    to stabilise training. The mean and standard deviation are computed across feature dimensions, then 
    normalised by applying learned scaling and offset parameters. LayerNorm works well with variable batch 
    sizes and sequence lengths, making it ideal for point cloud processing where input sizes can vary.
    Applied both before and after self-attention.

  - Self-attention

    Points are grouped into non-overlapping patches of 1024 points along the serialized order
    Each patch processes independently through standard query-key-value attention
    The large patch size of 1024 (vs PTv2's 16 points) is made possible by the efficiency gains from serialization

  - MLP layer for pointwise feature transformation

The feature dimensions follow a [64→128→256→512] pattern through the encoder stages, with corresponding decoder stages following [256→128→64→64].

#figure(
  image("figs/ptv3_1.png", width: 100%),
  caption: [PTv3 encoder structure described in Ref. @wu2024ptv3],
  outlined: false,
  placement: none,
  gap: 1em,
) <ptv3_encoder>

This architectural design represents a careful balance between efficiency and effectiveness. By replacing expensive operations like KNN search and relative positional encoding with structured serialization and sparse convolutions, PTv3 achieves both faster processing and larger receptive fields. The combination of multiple serialization patterns and multi-scale processing helps overcome the potential limitations of any single spatial organization scheme.

=== Architecture Performance

The architecture achieves significant efficiency improvements over its predecessor PTv2:
- 3.3× faster inference speed.
- 10.2× lower memory consumption.
- expansion of receptive field from 16 to 1024 points while maintaining efficiency.

State-of-the-art performance across key benchmarks:
- Indoor semantic segmentation: 79.4% mIoU on ScanNet test set
- Outdoor semantic segmentation: 83.0% mIoU on nuScenes test set, 75.5% mIoU on SemanticKITTI test set
- Waymo object detection (2-frame): 72.5%/72.1% mAP/APH for vehicles, 77.6%/74.5% mAP/APH for pedestrians

With multi-dataset joint training, these results improve further, demonstrating the architecture's ability to leverage larger-scale training effectively.

PTv3 shows that simplifying architecture design while focusing on scalability can lead to superior performance without sacrificing accuracy. Its reduced computational requirements make high-performance point cloud processing more practical for real-world applications, while its ability to leverage larger-scale training through multi-dataset approaches points to promising future developments in the field.
The success of PTv3 challenges the notion that increasing architectural complexity is necessary for improved performance, suggesting instead that thoughtful simplification enabling better scaling might be a more productive direction for future research.

== Point Prompt Training (PPT) module

Here we provide an overview of Ref. @wu2024ppt for completeness. Point Prompt Training represents a novel approach to enabling multi-dataset learning for 3D point cloud models. 
Where previous approaches struggled with negative transfer when training on multiple datasets simultaneously, PPT introduces mechanisms to handle domain differences while maintaining model performance. The key insight of PPT is that dataset-specific prompts can help the model adapt to different data distributions while sharing a common feature extraction backbone.

#figure(
  image("figs/ppt_1.png", width: 100%),
  caption: [Schematic from Ref. @wu2024ppt of the high-level design of the PPT module. Domain prompt adapters are learnable layers injected into a frozen segmentation backone network and allow it to adapt to the distributional qualities of distinct datasets, while categorical alignment replaces a fixed classification head with a label embedding-based alternative logit computation. This has the nice property of permitting zero-shot inference by providing a metric between correlated label categories.],
  outlined: false,
  placement: none,
  gap: 1em,
) <ppt_1>

The PPT framework consists of two primary components:

1. Domain Prompt Adapter with Prompt-driven Normalization
2. Language-guided Categorical Alignment

=== Domain Prompt Adapter

The domain prompt adapter allows the model to handle different dataset contexts through learnable domain-specific prompts. For each dataset i, PPT generates a learnable d-dimensional vector that serves as the domain-specific prompt. These prompts are then integrated into the network through Prompt-driven Normalization (PDNorm).

PDNorm replaces standard normalization layers throughout the network with a prompt-aware variant:

PDNorm(x, c) = ((x - E[x̄])/√(Var[x̄] + ϵ)) · γ(c) + β(c)

where:
- x is the input feature map
- c is the domain-specific prompt
- γ(c) and β(c) are learned scale and shift parameters generated from the prompt
- E[x̄] and Var[x̄] are computed independently for each dataset

This approach allows the network to adapt its feature normalization behavior based on the source dataset while maintaining a shared backbone. The domain prompts are trained jointly with the backbone network, allowing the model to learn optimal dataset-specific adaptations.

To ensure stable training, PPT employs:
- Zero-initialization of γ(c) and β(c) parameters
- Learning rate scaling for prompt-related parameters (typically 0.1x the backbone learning rate)
- Shared prompts across network layers rather than layer-specific prompts

=== Language-guided Categorical Alignment

A major challenge in multi-dataset training is handling different label spaces across datasets. PPT addresses this through Language-guided Categorical Alignment (LCA), which projects point features into a shared semantic space aligned with language embeddings of category labels.

The process works as follows:

1. Category names from all datasets are embedded using a pre-trained text encoder (e.g., CLIP)
2. Point features are projected into the same embedding space
3. Classification is performed by computing similarities between point embeddings and category embeddings
4. The InfoNCE loss is used to align point representations with their corresponding category embeddings

This approach has several advantages:
- Creates a unified semantic space across datasets
- Leverages semantic relationships between categories
- Enables zero-shot transfer to new categories
- Allows the model to benefit from semantic similarity between categories across datasets

The loss function for a point feature p and its corresponding category text embedding t is:

L = -log(exp(p·t/τ) / Σ(exp(p·t_i/τ)))

where:
- τ is a temperature parameter (typically set to 0.07)
- The sum in the denominator is over negative samples from the same dataset

=== Training Process

#figure(
  image("figs/ppt_2.png", width: 100%),
  caption: [Schematic from Ref. @wu2024ppt of the two main components of PPT in the network. Prompt-driven normalisation modules injected into the PTv3 backbone layers permits learning a lightweight dataset-specific feature rescaling. The language-guided categorical alignment module projects the point representations from the backbone into the space of the class-label CLIP embedding vectors and computes logits from their inner product.],
  outlined: false,
  placement: none,
  gap: 1em,
) <ppt_2>

PPT was demonstrated in two training scenarios:

1. Supervised joint training:
   - Train on multiple datasets simultaneously
   - Use domain prompts and LCA to handle dataset differences
   - Evaluate directly on target datasets

2. Supervised pre-training:
   - Pre-train on multiple datasets using PPT
   - Subsequently fine-tune on specific target dataset(s)
   - Benefits from better initialization due to multi-dataset exposure 

The sampling ratio between datasets during training was determined based on the optimal number 
of iterations needed for each dataset. This ensures each dataset contributes proportionally to 
the learned representations.

==== Performance Impact

PPT demonstrates significant improvements over baseline approaches:
- Eliminates negative transfer between datasets
- Improves performance on individual datasets compared to single-dataset training
- Enables effective knowledge transfer between synthetic and real datasets
- Achieves state-of-the-art results on multiple benchmarks with a single shared-weight model

For example, on ScanNet validation:
- Baseline SparseUNet: 72.2 mIoU
- With PPT joint training: 75.7 mIoU (+3.5)
- With PPT fine-tuning: 76.4 mIoU (+4.2)

Similar improvements are seen across other datasets and architectures, demonstrating PPT's effectiveness as a general approach to multi-dataset 3D representation learning.




= Experimental Setup

== Hardware/Software setup
The experiments were carried out on a machine with a 6-core processor, 32GB of RAM, and a 24GB VRAM RTX4090.
The sole exception to this is the experiment using the smaller library scene, which was carried out on a similar machine with a 12GB VRAM RTX4070.
The software platform used was Arch Linux, running the recent CUDA 12.6.
A docker container can be used to replicate the training environment precisely.

== Training and Evaluation Phase
A series of train-time transforms are utilised to help prevent overfitting and ensure VRAM constraints are respected, described below:

- *CenterShift 1*: shifts the center of the point cloud, including the z-axis.
- *RandomRotate Z*: rotates the point cloud around the z-axis by a random angle.
  - Angle: `[-1, 1]` radians
  - Probability: `25%`
- *RandomRotate X*: applies a small rotation around the x-axis (tilt)
  - Angle: `[-0.015625, 0.015625]` radians
  - Probability: `25%`
- *RandomRotate Y*: applies a small rotation around the y-axis (tilt)
  - Angle: `[-0.015625, 0.015625]` radians
  - Probability: `25%`
- *RandomScale*: uniformly scales the point cloud within a specified range.
  - Scale Range: `[0.9, 1.1]`
- *RandomFlip*: randomly flips the point cloud along one or more axes.
  - Probability: `50%`
- *RandomJitter*: adds Gaussian noise to simulate measurement errors.
  - Sigma: `0.005`
  - Clip: `0.02`
- *ChromaticJitter*: randomly adjusts color values to simulate lighting variations.
  - Probability: `95%`
  - Standard Deviation: `0.05`
- *GridSample*: samples points based on a grid to standardize spatial distribution.
  - Grid Size: 10cm
- *SphereCrop*: crops the point cloud within a random sphere to ensure VRAM limits are respected
- *CenterShift 2*: recenters the point cloud horizontally, excluding the z-axis.
- *NormalizeColor*: normalizes color channels to a standard range.
- *ShufflePoint*: randomizes the order of points to prevent any ordering bias.

Additionally, the Mix3D algorithm @nekrasov2021mix3doutofcontextdataaugmentation is used with an 85% probability per-batch.
From the abstract of that paper:
- "Since scene context helps reasoning about object semantics, current works focus on models with large capacity and receptive fields that can fully capture the global context of an input 3D scene. However, strong contextual priors can have detrimental implications like mistaking a pedestrian crossing the street for a car. In this work, we focus on the importance of balancing global scene context and local geometry, with the goal of generalizing beyond the contextual priors in the training set. In particular, we propose a "mixing" technique which creates new training samples by combining two augmented scenes. By doing so, object instances are implicitly placed into novel out-of-context environments and therefore making it harder for models to rely on scene context alone, and instead infer semantics from local structure as well."

In the training process, each mini-batch comprises a single scene from the training dataset.
Approximately 20 to 30 mini-batches are processed before conducting an evaluation epoch.
No gradient accumulation occurs between batches; instead, gradients are reset after each mini-batch to ensure updates are based solely on the current data.

The optimizer used is AdamW @loshchilov2019decoupledweightdecayregularization, which combines adaptive learning rates with weight decay regularization.
This optimizer accelerates convergence while preventing overfitting.
The use of the Lovász loss function directly optimizes the IoU metric, ensuring scarce categories receive adequate attention during training.

A OneCycleLR learning rate scheduler is employed to dynamically adjust the learning rate throughout the training process, thereby enhancing convergence and overall model performance.
The scheduler is configured with a maximum learning rate of 0.003, allowing significant parameter updates during the initial stages of training.
The learning rate will increase to its peak value within the first 5% of the training iterations.
Following this initial increase, the learning rate decreases according to a cosine annealing strategy, ensuring a smooth and gradual reduction as training progresses.

During evaluation epochs, all input scenes are processed multiple times to ensure comprehensive coverage of all categories within the sphere crops.
This repetition leads to a more reliable assessment of the model's performance.
The evaluation employs only deterministic transformations - CenterShift, GridSample, SphereCrop, and NormalizeColor - to maintain consistency.
Random data augmentations used during training are omitted to prevent stochastic variations from influencing the evaluation results.

== Testing Phase
The best model as determined during training via the evaluation epochs is then used to acquire predictions for the test scene, which the model has
not been exposed to at all during training.
Minimal transforms are applied here to ensure consistency with the network.
No test-time augmentations are performed at present due to issues with RAM consumption in the current implementation of the inference software, to ensure that
the test scenes are both as large as possible and able to be run through the network all at once.
Making the inference code more RAM efficient to enable test-time augmentations is a possible avenue for future work.

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
= Results
Three primary experiments were carried out with different sites used in the training and testing. These combinations were:
- Library scene alone
- Park Row and Maritime Museum
- Brass Foundry and both Royal Observatory sites
The more performant of the real-site experiments then had its training re-run, but with the library scene included in the training phase.

A model was also tested on a site it had not see at all, to test the ability to generalise when using such small training data sets.
Finally, inference on a real LiDAR cloud of the Queens House site was run using the most performant model.

== Intra-site experiments
Each of these experiments refer to a model that was trained and tested on the same site/sites.

=== Library Scene
The experiment training on just the HBIM library scenes functions as a metric for how distinguishable the different category geometries are in isolation.
The extremely synthetic nature of this scene means the model is unable to learn from typical geometric context between elements, limiting its ability to distinguish geometrically similar categories that would normally be distinguishable by that context (e.g. walkways and floors are almost identical given the lack of surface texturing and color texturing).

This reflects in the inference results for the test scene.
As can be seen in the summarised metrics in @lib_metrics and the confusion matrix in @lib_confmatrix, features that are geometrically similar are frequent points of confusion.
We note that classification for grass, ceilings and roofs perform well while floor and footpath are frequently mis-identified for one another:
- grass has enough RGB data (being the only green surface) to be identified by the network, showing the network is capable of using RGB information when the information makes for a powerful discriminator.
- ceiling and roof elements in the scene are elevated with a random offset, it's likely the network is learning to distinguish these two from other categories because of their relative height in the scene, showing the network is properly using z-axis contextual information.
- ceiling and roof are well-distinguished between each other because their shapes in the HBIM library are quite different. Ceilings tend to be sloped, roofs tend to be flat. As such we should not a priori expect strong separation in real scenes based on geometry alone: contextual information will be very important.

The results for categories which are geometrically distinct like railings, columns, RWP etc, the low-rank adaptation has sufficient felxibility to capture those differences.

#figure(
  image("figs/lib_inference1.png", width: 110%),
  caption: [Library test scene inference (panoramic).],
  outlined: false,
  placement: none,
  gap: 1em,
) <lib_inf1>

#figure(
  image("figs/lib_inference2.png", width: 110%),
  caption: [Library test scene inference (top-down).],
  outlined: false,
  placement: none,
  gap: 1em,
) <lib_inf2>

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
  caption: [Overall and per-category IoU and accuracy results for the Library test scene.],
  placement: none,
) <lib_metrics>

#figure(
  image("figs/lib_confmatrix.png", width: 120%),
  caption: [Library test scene confusion matrix.],
  outlined: false,
  placement: none,
  gap: 1em,
) <lib_confmatrix>

#pagebreak()
=== Park Row and Maritime Museum
Compared to the library experiment, these experiments begin to test the ability of the architecture to adapt to parts of buildings it has not yet seen, and to utilise contextual information found in real buildings.

As can be seen in the below results, the model performs overall very well across the board with the exception of Rainwater pipe, which remains a difficult feature to distinguish.
For these sites, this is likely because the number of instances of Rainwater Pipe in the input data is very limited, with only Park Row featuring any such features and Maritime Museum having none.


#figure(
  image("figs/mm_inference1a.png", width: 110%),
  caption: [Maritime Museum test scene 1. This shows some limited instances of the input data using the wrong classification (some of the ceiling here should clearly be wall). Other than this, the scene demonstrates a powerful ability to resolve both macroscopic and local structures.],
  outlined: false,
  placement: none,
  gap: 1em,
) <mm_inf1a>

#figure(
  image("figs/mm_inference1b.png", width: 110%),
  caption: [Maritime Museum test scene 1. Some limited misidentication of ceiling as footpath can be shown, indicating that perhaps contextual information on the edge of the scene is missing, or that these categories are still occasionally ambiguous to the network.],
  outlined: false,
  placement: none,
  gap: 1em,
) <mm_inf1b>

#figure(
  image("figs/pr_inference1.png", width: 110%),
  caption: [Park Row test scene, demonstrating excellent agreement between ground truth and prediction. The ability of the model to resolve fine geometric features on the periphery of the scene despite the scene edges lacking context is encouraging. The resolution of ceiling/roof/floor is also encouraging, indicating that perhaps the model is able to acquire enough local context to resolve these otherwise similar surface categories.],
  outlined: false,
  placement: none,
  gap: 1em,
) <mm_inf1a>

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

#figure(
  image("figs/prmm_confmatrix.png", width: 120%),
  caption: [Park Row and Maritime museum confusion matrix.],
  outlined: false,
  placement: none,
  gap: 1em,
) <prmm_confmatrix>

#pagebreak()
=== Brass Foundry and Royal Observatory
The results for this experiment largely echo those of the Park Row/Maritime Museum experiment, with a few exceptions:
- the geometry for buildings in this experiment, particularly for the ROG sites, is much more complex. This leads to a slightly poorer ability of the model to distinguish walls from categories typically proximate to them, like doors and windows.
- the performance for RWP is much stronger here, although still one of the least accurate classifications. This is likely because these sites are more rich in RWP structures than the Park Row/Maritime Museum setup.
- the performance for columns suffers here due to the columns in these sites being not only more scarce, but much smaller than in Park Row/Maritime Museum.

The above differences between the two experiments demonstrates quite strongly the limitations of using such low numbers of sites in the training; more sites with better and more diverse coverage of elements across the taxonomy is key for the model's ability to generalise.

#figure(
  image("figs/rogsouth_inference1.png", width: 110%),
  caption: [ROG South test scene 1. Compared to the previous experiment, the more complex building geometry makes the resolution of doors and windows somewhat weaker. Nevertheless, the model still is able to resolve nearly all the macroscopic structures in the scene.],
  outlined: false,
  placement: none,
  gap: 1em,
) <rogsouth_inf1a>

#figure(
  image("figs/rognorth_inference1.png", width: 110%),
  caption: [ROG North test scene 1. Despite the intricate structure in the complex of buildings, the model is largely able to well-resolve ceilings and walls. The outdoor elements are more weakly resolved.],
  outlined: false,
  placement: none,
  gap: 1em,
) <rogsouth_inf1a>

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

#figure(
  image("figs/rogbr_confmatrix.png", width: 120%),
  caption: [Brass Foundry and ROG confusion matrix.],
  outlined: false,
  placement: none,
  gap: 1em,
) <bfrog_confmatrix>

#pagebreak()
=== Augmenting Park Row/Maritime with Library Scene
This experiment investigated the impact of including the HBIM library scene in the model training upon the inference for the Park Row and Maritime Museum test scenes.

Generally, we observe that despite some small features being slightly more well resolved, an overall degradation is observed in the resolution of macroscopic features.
In particular, this version of the model is slightly more prone to the misidentication of other categories as footpath.

As such, it can be concluded that the loss of scene context in the HBIM library data results in a detrimental effect when the model is applied to real building configurations.

#figure(
  image("figs/libaugmented_inference1.png", width: 110%),
  caption: [Maritime Museum test scene 1, inferred using a model augmented with the library training scene. Compared to the model without training augmentation, the resolution of the central chimney feature is improved, but with the stairs outside almost entirely being mis-identified as footpath. ],
  outlined: false,
  placement: none,
  gap: 1em,
) <libaugmented_1>

#figure(
  image("figs/libaugmented_inference2.png", width: 110%),
  caption: [Park Row test scene, inferred using a model augmented with the library training scene. Compared to the model without training augmentation, we can observe a higher incidence of floors being misidentified as footpath.],
  outlined: false,
  placement: none,
  gap: 1em,
) <libaugmented_2>

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

#figure(
  image("figs/libaugmented_confmatrix.png", width: 120%),
  caption: [Park Row and Maritime Museum with library training augmentation confusion matrix.],
  outlined: false,
  placement: none,
  gap: 1em,
) <libaugmented_confmatrix>

#pagebreak()
=== Intra-site summary
For each experiment, the following metrics were constructed:
#figure(
  table(
    columns: 3,
    // Header row
    [Metric], [Formula], [Use Case],
    // Data rows
    [Accuracy], [$ (T P + T N) / "total points" $], [Measures overall performance but can be skewed by class imbalance.],
    [Balanced Accuracy], [$ (1 / N) sum_(i=1)^N T P_i / "total points in class"_i $], [Useful for imbalanced datasets, as it averages accuracy across classes.],
    [Recall], [$ T P / (T P + F N) $], [Measures completeness for each class; high recall means fewer false negatives.],
    [Precision], [$ T P / (T P + F P) $], [Measures correctness when the model predicts a class; high precision means fewer false positives.],
    [F1-Score], [$ 2 * ("Precision" * "Recall") / ("Precision" + "Recall") $], [Balances precision and recall, useful in class-imbalanced cases.],
  ),
  caption: [Metrics used in model evalutaions],
  placement: none,
) <table>

- $T P$: True Positives — Points correctly predicted as belonging to a class.
- $T N$: True Negatives — Points correctly predicted as not belonging to a class.
- $F P$: False Positives — Points incorrectly predicted as belonging to a class.
- $F N$: False Negatives — Points that belong to a class but were not predicted as such.

The results are shown in @allmetrics.

#figure(
  image("figs/allmetrics.png", width: 89%),
  caption: [Park Row and Maritime Museum with library training augmentation confusion matrix.],
  outlined: false,
  placement: none,
  gap: 1em,
) <allmetrics>

As expected, the very simple library experiment demonstrates the strongest metrics across the board.
For the experiments with actual sites, the model with Park Row and Maritime Museum performs the strongest.
This does not necessarily entail that the model is better for general use than the one trained on Brass Foundry and the ROG sites;
the increased complexity and variety within the ROG sites compared to Park Row/Maritime Museum means that the testing scenes
in those sites are less similar to the training and evaluation scenes.

We note that the metrics for the library augmented scene are broadly similar to the Park Row/Maritime Museum control sample,
but with a significantly lower balanced accuracy, suggesting that the model trained with library augmentation has slightly more bias 
towards the dominant classes in the data.

#pagebreak()
== Inter-site experiments: Park Row/MM model with Brass Foundry test scene

As a check on how well the models are able to generalise to buildings they have not seen at all, predictions for the brass foundry test
scene were acquired using the model trained on Park Row/Maritime Museum.

@bfgood1 shows the inference on the Brass Foundry test site using the model trained on the Brass Foundry and ROG sites, while @crosscheck1
shows the same scene run through the model trained on Park Row and Maritime Museum.
A significant degradation in performance can be observed, with the model commonly confusing footpath, wall, and other in particular.
The ceiling is generally identified well, but some sections are still misidentified as footpath or railing.

#figure(
  image("figs/bfgood1.png", width: 89%),
  caption: [Brass Foundry test scene when the model has been trained on part of the Brass Foundry Site.],
  outlined: false,
  placement: none,
  gap: 1em,
) <bfgood1>

#figure(
  image("figs/crossmodel1.png", width: 89%),
  caption: [Brass Foundry test scene using the model trained on Park Row/Maritime Museum.
  The model displays significant degradation in performance compared to the intra-site experiments.],
  outlined: false,
  placement: none,
  gap: 1em,
) <crosscheck1>

In @bfgood2 and @crosscheck2 we can see the same scene from the other side, exposing the building interior.
This reveals a strong misidentication of the shelving units inside the building (in the Other category) as a variety of 
other categories, including

#figure(
  image("figs/bfgood2.png", width: 89%),
  caption: [Brass Foundry test scene when the model has been trained on part of the Brass Foundry Site.],
  outlined: false,
  placement: none,
  gap: 1em,
) <bfgood2>

#figure(
  image("figs/crossmodel2.png", width: 89%),
  caption: [Brass Foundry test scene using the model trained on Park Row/Maritime Museum.
  The interior angle reveals that the shelving units in the Other class are being misidentified as a variety of other categories.
  We can also see the floors and ceilings being confused for footpaths.],
  outlined: false,
  placement: none,
  gap: 1em,
) <crosscheck2>

While the performance degradation is quite pronounced, it is not entirely unexpected.
The model used has only been trained on two buildings, Park Row and Maritime Museum.
These buildings are part of the same complex, and are quite similar, so it's reasonable to expect that the model
might have a harder time generalising to buildings it has not seen.

This is especially true for the Other category, where the types of objects that can fall under this category is of course exceptionally broad.
There are no close analogues to the shelving units in the Park Row/Maritime Museum sites;
nevertheless the model does assign a significant number of the points corresponding to those shelves as Other.

The more significant issue is the model's frequent confusion of flat surface categories for footpath.
The meshes for these surfaces are geometrically similar, so the model has to rely on local context to differentiate between floor, ceiling, footpath etc.
It should be noted that the use of properly textured meshes could provide the necessary colour information for the model to distinguish between these categories.
Texture mapping would be an obvious choice here for immediate future improvements to the approach.

#pagebreak()
== Inference on Queens House LiDAR data
As a final experiment, the most performant model (the one trained on Park Row and Maritime Museum) was used to run inference on the Queens House LiDAR-acquired pointcloud.
The finely-sampled LiDAR pointcloud was downsampled to a 5cm resolution, and predictions were acquired using the model.
Ground Truths for the updated taxonomy were not yet available, and so the only checks currently possible are visual inspections of the inference results.

It should be noted that due to memory limitations arising in the current implementation of the inference software, the Queens House data had to be 
passed through the network in fairly aggressively chunked subsamples.
This could be resulting in the model losing some context around the edges of each subsample, deteriorating performance.
Adjustments to make the algorithm more memory-efficient, or even simply running the inference on a machine with a larger RAM capacity, would be a high priority in any future work.

In @qh1 we show an exterior corner of the QH building.
The major 2 issues visible, that occur across the QH subsamples, are the following:
- walls are frequently misidentified as Other features.
- Flat surfaces like roofs, floors, and ceiling, are frequently misidentified as footpaths.

#figure(
  image("figs/qh1.png", width: 89%),
  caption: [Model predictions for an exterior section of the Queens House. The most obvious classification errors present are walls being misidentified as Other features.
  We can also see the ceiling is mostly misidentified as footpath.],
  outlined: false,
  placement: none,
  gap: 1em,
) <qh1>

For the former issue, there are several possible explanations for this, and ways to improve upon the approach.
The most obvious issue is that for this model, a great many features classes as Other resemble walls (or could indeed be perceived by 
a person as a wall of some kind).
A render of the Other mesh category for Maritime Museum is shown in @mm_other.

#figure(
  image("figs/mm_other_meshes.png", width: 89%),
  caption: [Render of the meshes designated "Other" in the Maritime Museum HBIM data. A great many of these structures are geometrically very similar to walls.],
  outlined: false,
  placement: none,
  gap: 1em,
) <mm_other>

Given how many of these museum plinths, boundaries, and infrastructural elements are very distinctly similar to walls, the model's poor resolution between Wall and Other
is not too surprising.
An obvious way to address this would be to incorporate more sites in the training such that the network can be exposed to more miscellaneous structures.

Other avenues of possible improvement include refinements of the loss function used in the network's training (see @customloss).
Another is to incorporate the library scene to more strictly reinforce the expected geometrical differences within Other.

The second dominant error that can be seen in the visualisations, the miscategorisation as footpath of other flat surface categories, can be likely improved
dramatically by the inclusion of more complex colour information and texturing.
This was verified by converting the RGB information on the QH data into a grayscale colourspace by using the luminence formula.
It could be observed that the network's output barely changed at all without colour information present in the input data, meaning that
the network is very likely not deriving much useful information from colour when distinguishing most cateogories.

It should be noted that the RGB data present in the HBIM meshes is sometimes a very useful discriminator, as is the case for grass
which is always uniquely coloured green in the training data, and is generally well identified relative to the other flat surface 
categories.
For all of these reasons, capturing the genuine colour and texture information of the real sites in our HBIM data is a very compelling next step
for improving the network's performance on flat surfaces more generally.

Despite these two very significant types of error observed in the model predictions, there are some positives to take away from this limited test.
As demonstrated in @qh2 and @qh3, the model is already relatively adept at classifying structures that are geometrically distinct, like columns and stairs.
For a model trained on such a finite amount of data, this is a very encouraging result.

#figure(
  image("figs/qh2.png", width: 89%),
  caption: [Model predictions for an exterior section of the Queens House. The columns supporting the covered walkway are fairly well identified by the model.],
  outlined: false,
  placement: none,
  gap: 1em,
) <qh2>

#figure(
  image("figs/qh3.png", width: 89%),
  caption: [Model predictions for an interior section of the Queens House. The stairway in the center of the image is well resolved from the surrounding elements.],
  outlined: false,
  placement: none,
  gap: 1em,
) <qh3>

== Results Summary

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
= Future Work
In this section, we will describe possible efforts that could improve the performance of the network, with particular focus on improvements that could benefit
applications to point clouds gathered with LiDAR instrumentation.

== Training with more HBIM site data
Model performance could likely be improved substantially by expanding the HBIM dataset to include data from additional sites.
Currently, only two out of four available sites are used in each model's training, which limits the model's exposure to a diverse range of architectural styles and environmental variations present across heritage buildings.
Training a model with data from all four sites would provide a more comprehensive understanding of these variations, enhancing the model's ability to generalize and accurately segment across different structures.
Further augmentation with additional HBIM data from new sites added to the project in the future would likely yield even greater gains in performance.

== Enhancing colour information
Performance of the network could be greatly improved by the inclusion of proper colour texture information in the meshes, allowing geometrically similar features such as floors and footpaths to be more easily distinguished.
These features, often difficult to differentiate based solely on geometry, can present similar shapes and structural characteristics.
However, by introducing accurate color information, additional distinctions could be made based on the natural variations in texture and color patterns found in real-world scenes.

The presence of some limited mesh color and texture information suggests that complete texture information is indeed available in the HBIM data.
Therefore, a relatively minor adjustment to the data ingestion pipeline could ensure that these textures are fully preserved and utilized.
By enabling color to complement geometry, the network's segmentation accuracy across challenging classes could be significantly enhanced.

== Custom Loss Functions <customloss>
the implementation of a loss function incorporating explicit class weights could be explored to differentiate the severity of classification errors based on both class and error type.
This approach would allow for the punishment of misclassifications to vary depending on the specific classes involved; for example, erroneously identifying a "roof" as a "wall" would be penalized more heavily than misclassifying "grass" as a "footpath." Additionally, imposing stringent penalties on the misattribution of cardinal classes to the "other" category would encourage the network to adopt a more conservative approach when predicting "other."
This would ensure that such predictions are made only when the network achieves a high level of certainty, thereby enhancing the precision of class-specific predictions and reducing the likelihood of ambiguous classifications.
By tailoring the loss function in this manner, the overall robustness and reliability of the model could be significantly improved, leading to more accurate and trustworthy outcomes in diverse classification scenarios.

== Threshold-Based Prediction Strategies
As an alternative to a conventional softmax approach, alternative prediction strategies beyond the conventional argmax approach could be explored to enhance classification accuracy.
Specifically, the implementation of post-hoc thresholds for the "other" class might be considered.
This method would involve assigning a prediction of "other" only when the probabilities of all other classes fall below a predefined threshold, and conversely, ensuring that the "other" class probability is sufficiently high.
In instances where these conditions are not met, the model could default to selecting the class with the second-highest probability.

While this thresholding technique offers a potential avenue for refining predictions, it may introduce a degree of brittleness and rely on additional statistical heuristics.
Therefore, it is acknowledged that more robust and sophisticated methods could be developed to address classification challenges at their core.
Nonetheless, experimenting with threshold-based approaches could provide valuable insights and serve as a supplementary mechanism to improve the model's decision-making process in specific scenarios.

== Refactoring the "Other" Category

=== N-1 Classification Approach

An alternative approach to handling the "other" category involves transforming the classification problem from an N-class to an N-1 class scenario by removing the explicit "other" category.
This modification necessitates altering the network's output layers to abandon the prediction of explicit class probabilities that sum to one. Instead, an element-wise sigmoid output layer could be employed, allowing the model to predict independent scores between zero and one for each cardinal class.
In this paradigm, predictions of "other" would be defined as instances where the scores of all other classes fall below a predetermined threshold.
Conversely, if the "other" class score is sufficiently high, it would be selected as the prediction. In cases where these threshold conditions are not met, the model would default to selecting the class with the second-highest probability.
Although this method offers a means to circumvent the ambiguous semantics of the "other" label within the PPT module, it may introduce complexity and rely on additional heuristics.
Consequently, more robust and sophisticated techniques are recommended to address the underlying classification challenges directly rather than implementing workaround solutions.

=== Language Semantic Augmentation

The inherent semantic ambiguity of the "other" category, coupled with the interplay between label CLIP embeddings and PTv3 latent representations, presents significant challenges.
The language embeddings associated with "other" are likely to be non-contributory or potentially detrimental to accurate classification.
To mitigate this issue, the introduction of more granular information regarding the "other" category at a coarse textual level is proposed.
For instance, providing summaries of the contents encompassed by "other" within a specific dataset—such as "shelves" or "machinery" - could help resolve label degeneracy by enhancing the PPT categorical alignment module with embeddings of these finer-grained labels.

Additionally, the network's loss function would require careful modification to accommodate this refined labeling, as point-level labels would remain under the broader "other" category without distinguishing between the distinct subclasses.
This enhancement could be achieved manually or delegated to a Vision-Language Model (VLM), which might be trained to analyze static images of "other" regions within the input cloud and generate descriptive labels for the objects constituting the "other" class.
By explicitly incorporating more detailed semantic information, the model's ability to accurately classify and differentiate between various objects within the "other" category could be substantially improved, thereby enhancing overall classification performance and reliability.


#pagebreak()
#bibliography("bibliography.bib")
