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
* train first with cce loss, then with smooth_topk loss
* remove the first maxpool layer from resnet/senet models
* use attention
* use hypercolumns
* use cv2.polylines?
* stratify mini-batches
* combine lr-plateau scheduling with sgdr
* augment the number of feature maps in the cnn model and add se/scse block
* use drop connect
* use fully convolutional model with progressive image resizing
* train on category subset and add "other" category for samples on in complementary category set
* http://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/
* papers
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
