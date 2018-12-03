# MM18-Deep-Cognitive-Subspace-Clustering
This is a TensorFlow implementation of the Deep Cognitive Subspace Clustering model as described in our paper:

>Y. Jiang, Z. Yang, Q. Xu, X. Cao and Q. Huang. When to Learn What: Deep Cognitive Subspace Clustering. ACM MM 2018.

## Acknowledge
The implementation is based on [Deep Subspace Clustering Network](https://github.com/panji1990/Deep-subspace-clustering-networks). We sincerely thank the authors for their work.

## Dependencies
- Tensorflow
- numpy
- sklearn
- munkres
- scipy

## Data
The data and pre-trained auto-encoder model can be found [here](https://github.com/panji1990/Deep-subspace-clustering-networks/tree/master/Data). Please put the data in `data/` and the pre-trained model in `models/`.

## Citation
Please cite our paper if you use this code in your own work:

```
@inproceedings{jiang2018learn,
  title={When to Learn What: Deep Cognitive Subspace Clustering},
  author={Jiang, Yangbangyan and Yang, Zhiyong and Xu, Qianqian and Cao, Xiaochun and Huang, Qingming},
  booktitle={2018 ACM Conference on Multimedia},
  pages={718--726},
  year={2018},
  organization={ACM}
}
```
