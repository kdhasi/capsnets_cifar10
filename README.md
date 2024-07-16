# Keras Implementation and t-SNE Visualization of a Dynamic Routing CapsNet

This code provides a Keras implmentation of the CapsNet using the CIFAR-10 dataset and visualizes the activation capsules for different classes using the T-distributed stochastic neighbor embedding (t-SNE) technique.

The plots provide insight onto the clustering of the capsules as we see more discrete clusters formed with high-resolution images compared to the low-resolution counterparts.

## Credit
The base capsule network model follows the implementation here: <a href="https://github.com/XifengGuo/CapsNet-Keras" target="_blank">link</a>.

## t-SNE Diagrams for High-Resolution (32x32) vs Low-Resolution (8x8) CIFAR-10 Images

![dynamicrouting_tsne_dcaps_HR](https://github.com/kdhasi/capsnets_cifar10/assets/146899852/7f431428-14a2-4be5-a7ee-241f8a51b2de)
![dynamicrouting_tsne_dcaps_VLR](https://github.com/kdhasi/capsnets_cifar10/assets/146899852/ba47ddd5-79cc-488e-8b42-65ab7c74be78)
