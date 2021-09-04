# Copy-and-Paste
Copy and Paste Implementation for Semantic Segmentation. Link to the [paper](https://arxiv.org/abs/2012.07177). </br>
</br>
</br>

## Notice
- Annotation should be in object index. For instance: </br>
Class #0: 0, 0, 0 </br>
Class #1: 1, 1, 1 </br>
Class #2: 2, 2, 2 </br>
etc. </br>
An example can be found [here](https://github.com/WeiChihChern/Copy-Paste-Semantic-Segmentation/tree/master/Example/data/train_mask "here"). Although you might not see the mask since each class has low intensity values.
</br>



### Current Status
- Tested. Works with albumentations. (See demo [here](https://github.com/WeiChihChern/copy-and-paste/blob/main/Example/Demo.ipynb "here"))
- Current implementation contains copy then paste only. Since semantic segmentation annotation may not be labeled as instance segmentation (instance wise annotated).
- Paste with transaltion (x,y shift) supported.
- Rotation and Scaling supported.
- Augmentation probability control added, so you can ignore/increase augmentation to certain class(es) 

### Augmentation Flowchart:
1.  Put `SemanticCopyandPaste()` before other albumentations augmentation (See demo [here](https://github.com/WeiChihChern/copy-and-paste/blob/main/Example/Demo.ipynb "here"))
2. Then follow other augmentation such as flip, transpose, random crop, etc.


### Before and After
Before Augmentation: </br>
<img src="https://user-images.githubusercontent.com/40074617/113963987-9a385a00-97f8-11eb-8ee3-6c3f0bbdb426.png" width="800"> </br>
After Augmentation: </br>
<img src="https://user-images.githubusercontent.com/40074617/114114686-581e1f80-98af-11eb-8e34-45dfea8344cc.png" width="800"> </br>

Another Example using another dataset: </br>
<img width="800" alt="Screen Shot 2021-08-17 at 11 47 23 PM" src="https://user-images.githubusercontent.com/40074617/129833869-e9e1b184-1de2-43c8-a9ff-79e1d1570c72.png"> </br>


# Citing
    @misc{Chern:2021,
      Author = {Wei Chih Chern},
      Title = {Copy-Paste for Semantic Segmenetation},
      Year = {2021},
      Publisher = {GitHub},
      Journal = {GitHub repository},
      Howpublished = {\url{https://github.com/WeiChihChern/Copy-Paste-Semantic-Segmentation}}
    }
