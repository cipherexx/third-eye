# Third-Eye
## Implementation of F3-Net: Frequency in Face Forgery Network
This model is made to implement F3-Net and is not the official implementation. To know about F3-Net, go through the [paper](https://arxiv.org/abs/2007.09355) here.

## Dependencies
Requires PyTorch, Torchvision, Numpy, SKLearn and Pillow. 
Simply run
`pip install requirements.txt`

## Usage

#### Hyperparameters

Hyperparameters are in `train.py`.

| Variable name   | Description                             |
| --------------- | --------------------------------------- |
| dataset_path    | The path of the dataset                 |
| pretrained_path | The path of pretrained Xception model.  |
| batch_size      | 128 in paper.                           |
| max_epoch       | How many epochs to train the model.     |
| loss_freq       | Print loss after how many iterations    |
| mode            | Mode of the network                     |

#### Load pretrained Xception model
Download *Xception* model trained on ImageNet (through this [link](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth)) or use your own pretrained *Xception*.

Then modify the `pretrained_path`  variable.

#### Use Face Forensics++ Dataset
This model has been built to work with the Face Forensics++ Dataset. It is currently coded to only take the Deepfakes and Face2Face folders in account, but this can be changed by adding other methods like FaceSwap and NeuralTextures from FF++ to `fake_list` in `utils.py`

After preprocessing the data should be organised as below whenever applicable:

```
|-- dataset
|   |-- train
|   |   |-- real
|   |   |	|-- 000_frames
|   |   |	|	|-- frame0.jpg
|   |   |	|	|-- frame1.jpg
|   |   |	|	|-- ...
|   |   |	|-- 001_frames
|   |   |	|-- ...
|   |   |-- fake
|   |   	|-- Deepfakes
|   |   	|	|-- 000_167_frames
|   |		|	|	|-- frame0.jpg
|   |		|	|	|-- frame1.jpg
|   |		|	|	|-- ...
|   |		|	|-- 001_892_frames
|   |		|	|-- ...
|   |   	|-- Face2Face
|   |		|	|-- ...
|   |   	|-- FaceSwap
|   |   	|-- NeuralTextures
|   |-- valid
|   |	|-- real
|   |	|	|-- ...
|   |	|-- fake
|   |		|-- ...
|   |-- test
|   |	|-- ...
```

#### Model modes

There are four modes supported in F3-Net​.

| Mode(string)       |                                                              |
| ------------------ | -------------------------------------------------------      |
| 'FAD'              | Only Frequency Aware Image Detection                         |
| 'LFS'              | Only uses Local Frequency Statistics                         |
| 'Both'             | Use both of branches and concatenates before classification. |
| 'Mix' | Uses a cross attention model to combine the results of FAD and LFS        |


## Running
To train the model, run
`python train.py`

## Reference

Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining frequency-aware clues. arXiv preprint arXiv:2007.09355, 2020
[Paper Link](https://arxiv.org/abs/2007.09355)


