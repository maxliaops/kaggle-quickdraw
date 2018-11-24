### Countdown

* calculate confusion on val set, then use category confidence to "correct" category of noisy samples
* k-fold
* cnn tree
* pseudo labeling
* analyze test predictions which are likely to be wrong
* fine tuning
* consider [entropy](https://www.kaggle.com/sorokin/sketch-entropy) to remove noisy samples for training or find unlikely predictions
* use more/all data for training
* use soft/hard bootstraping losses


### Ideas

* train separate models for small number of categories
  * use to decide upon "ambiguous" results
* break down precision per
  * category
  * recognized vs. non-recognized
* visualize confusion matrix
  * which categories are usually confused?
  * train separate models for categories which are often confused?
* ignore samples with recognized=false
* check how close the top-3 predictions softmax is for wrongly classified samples
* normalize batch
* mmap numpy images
* https://www.kaggle.com/c/quickdraw-doodle-recognition/discussion/68006
* add batch normalization as first layer in resnet/drn models
* train x classifiers on disjoint category groups, then classify with every classifier and pick the most discriminative one
* train first with cce loss, then with smooth_topk loss (tune smooth_topk loss parameters)
* remove the first maxpool layer from resnet/senet models
* use attention
* use hypercolumns
* add swa
* use small epsilon threshold to consider a model improvement a relevant improvement (for early abort etc.)
* add residual connections
* add simple augmentation (fliplr + scaling) which can be implemented while drawing strokes
* use cv2.polylines?
* stratify mini-batches (https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911)
* combine lr-plateau scheduling with sgdr
* augment the number of feature maps in the cnn model and add se/scse block
* use drop connect
* use fully convolutional model with progressive image resizing
* train on category subset and add "other" category for samples on in complementary category set
* http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
* include statistics in one of the channels used for training: http://rstudio-pubs-static.s3.amazonaws.com/292508_8ef4c9ec5f76421e92803fccba9765df.html
* https://github.com/KaiyangZhou/pytorch-center-loss
* ensembling
  * https://mlwave.com/kaggle-ensembling-guide/
  * https://www.analyticsvidhya.com/blog/2015/09/questions-ensemble-modeling/
* https://github.com/batzner/indrnn
* noise
  * https://arxiv.org/pdf/1705.03419.pdf
  * https://hackernoon.com/cnns-with-noisy-labels-using-deepcognition-research-paper-1bf4094e3bd9
  * https://arxiv.org/pdf/1703.08774.pdf
  * https://arxiv.org/pdf/1412.6596.pdf
  * https://arxiv.org/pdf/1710.09412.pdf


### Papers

* https://arxiv.org/pdf/1704.03477.pdf
* https://arxiv.org/pdf/1706.03762.pdf
* https://arxiv.org/pdf/1802.07595.pdf
* https://arxiv.org/pdf/1806.05594.pdf
* http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/2763.pdf
* https://arxiv.org/pdf/1501.07873.pdf
* https://arxiv.org/pdf/1803.00942.pdf
* https://arxiv.org/pdf/1804.00021.pdf
* http://cybertron.cg.tu-berlin.de/eitz/pdf/2012_siggraph_classifysketch.pdf
* https://arxiv.org/pdf/1511.04534.pdf
* https://arxiv.org/pdf/1405.3080.pdf
* https://ydwen.github.io/papers/WenECCV16.pdf
* https://arxiv.org/pdf/1706.02677.pdf
