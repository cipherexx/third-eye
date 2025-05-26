# Third-Eye
## Implementation of F3-Net: Frequency in Face Forgery Network
This model is made to implement F3-Net and is not the official implementation. To know about F3-Net, go through the [paper](https://arxiv.org/abs/2007.09355) here.

## Deployment Note:
We have a deployed version of the model up at https://third-eye-by-mihir.streamlit.app/
This is currently a much lighter version and transforms every image to 32x32 before checking and is trained on facial as well as non facial data. We shall soon switch to the original model that this code trains, on the demo as well!

## Dependencies
Requires PyTorch, Torchvision, Numpy, SKLearn Pillow, Gradio and email-validation(this is needed purely for gradio to function properly). 
Simply run
`pip install requirements.txt`

## Usage

#### Hyperparameters

Hyperparameters are in `train.py`.

| Variable name   | Description                             |
| --------------- | --------------------------------------- |
| dataset_path    | The path of the dataset                 |
| pretrained_path | The path of pretrained Xception model.  |
| batch_size      | Batch size for training                 |
| max_epoch       | How many epochs to train the model.     |
| loss_freq       | Print loss after how many iterations    |
| mode            | Mode of the network                     |

#### Load pretrained Xception model
*Xception* model trained on ImageNet ([link](http://data.lip6.fr/cadene/pretrainedmodels/xception-b5690688.pth)) has been used, and is present in the project files as xception.pth
You can choose to use another pretrained Xception model and modify the `pretrained_path`  variable.

#### Use Face Forensics++ Dataset
This model has been built to work with the Face Forensics++ Dataset. It is currently coded to only take the Deepfakes and Face2Face folders in account, but this can be changed by adding other methods like FaceSwap and NeuralTextures from FF++ to `fake_list` in `utils.py`
The version used during the project is present in the ff_raw folder.

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
|   |-- val
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
| 'FAD'              | Only Frequency Aware Image Decomposition                        |
| 'LFS'              | Only uses Local Frequency Statistics                         |
| 'Both'             | Use both of branches and concatenates before classification. |
| 'Mix' | Uses a cross attention model to combine the results of FAD and LFS        |


## Running
To train the model, run
`python train.py`

Once the model is trained and saved as end.pkl in ckpts, run
`python gradio.py`
to run a gradio powered inference where you can upload images and determine whether or not they are deepfakes. Ensure the images used have a full face at their centre.

We also have a live demo up at https://third-eye-by-mihir.streamlit.app/
Note that this is running a much lighter model, which resizes every image to 32x32 before checking it. We shall soon switch to the full-sized model on the demo as well.

## Reference

Yuyang Qian, Guojun Yin, Lu Sheng, Zixuan Chen, and Jing Shao. Thinking in frequency: Face forgery detection by mining frequency-aware clues. arXiv preprint arXiv:2007.09355, 2020
[Paper Link](https://arxiv.org/abs/2007.09355)


