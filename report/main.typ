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
  *#title*
])

#align(center, text(12pt)[
  *#subtitle*
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
#show: rest => columns(2, rest)


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


= Conclusion

Conclude your findings here.

#bibliography("bibliography.bib")