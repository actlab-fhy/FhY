=========
Overview
=========

.. figure:: /_static/img/fhy_high_level_overview.png
   :alt: High-level overview of the FhY compiler.
   :align: center

   High-level overview of the compilation flow.

*FhY* is a next-generation cross-domain programming language pioneered by the Alternative Computing Technologies (ACT) Lab to break the rigid boundaries of traditional domain-specific approaches.
In an era where performance demands have led to an explosion of highly specialized accelerators -- each tailored for a specific computational domain -- *FhY* emerges as a transformative solution that reimagines acceleration across multiple domains without sacrificing expressiveness or efficiency.

Traditional domain-specific languages (DSLs) are tightly coupled to individual algorithmic domains, and often, hardware architectures, offering exceptional performance at the cost of flexibility.
This approach has let the pendulum swing from general-purpose processors to bespoke hardware for each domain.
For instance, deep learning workloads continue to demand more compute, resulting in the continued development of specialized tensor units like Google's TPU and NVIDIA's Tensor Core to perform high-throughput matrix multiplications with minimal energy consumption.
Similarly, digital signal processing (DSP) for real-time applications, such as audio and video processing, necessitates highly efficient domain-specific accelerators to meet stringent latency and power constraints.
However, the growing need for end-to-end applications spanning multiple domains in areas such as robotics and autonomous driving demands a new paradigm.

*FhY* provides a unified computational stack that bridges diverse domains through a high-level cross-domain language (CDL) designed to represent operations using mathematics in a modular and reusable manner.
This unique approach allows *FhY* to express computation across heterogeneous workloads, empowering developers to harness the best of various domain-specific accelerators while maintaining a consistent, expressive programming model.

At the heart of *FhY* lies a fractalized intermediate representation (IR), f-DFG, that unlocks recursive access to all levels of operation granularity.
This multi-granular approach is crucial because accelerators operate at varying granularities of computation, unlike traditional CPUs, which execute instructions at a fixed level of granularity.
For instance, while a CPU processes operations at a fine-grained level -- executing basic arithmetic instructions one at a time -- specialized accelerators can encapsulate complex operations, such as matrix multiplications, within a single instruction to maximize efficiency.

To fully harness the potential of diverse hardware, an effective IR must provide flexibility across these different granularities, allowing developers to target the coarsest operations to accelerators while delegating finer operations to CPUs.
The f-DFG IR achieves this by representing computations in a hierarchical manner, offering recursive access to all levels of abstraction -- whether it be individual operations or high-level algorithmic constructs.
This adaptability enables efficient mapping of computations across a heterogeneous landscape, ensuring optimal performance without sacrificing expressiveness.

*FhY* is more than just a language -- it's a gateway to multi-acceleration in the modern computational landscape, enabling a future where software and hardware converge across domains effortlessly.

Join us as we redefine the boundaries of acceleration with *FhY*.
