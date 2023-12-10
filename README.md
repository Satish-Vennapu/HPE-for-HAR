# HPE-for-HAR
Human Pose Estimation for multi-view Human Action Recognition 

# Dependencies
The dependencies are listed in requirements.txt. You can install them with the following command:
```bash
pip install -r requirements.txt
```

# Dataset Preparation
The dataset used in this project is the [NTU RGB+D](
http://rose1.ntu.edu.sg/datasets/actionrecognition.asp) dataset. The dataset is divided into 3 parts:
* RGB videos
* Depth videos
* Skeleton data

## Data Preprocessing
- We use the skeleton data for this project. The skeleton data is in the form of .skeleton files. Each .skeleton file contains the 3D coordinates of 25 joints of a person in a frame. The skeleton data is extracted from the .skeleton files and stored as .npy files. The code for this can be found here: [Skeleton Data Extraction](https://github.com/shahroudy/NTURGB-D). 
- Create a folder called dataset in the root folder of the project. The folder structure should be as follows:
    ```bash
    HPE-for-HAR
    ├── dataset
    │   ├── S001C001P001R001A001.skeleton.npy
    │   ├── ...
    ├── remaining files
    ```
- Use the code from the link above to extract the skeleton data from the .skeleton files into the dataset folder.

- The skeleton data is stored in the form of numpy arrays. Each numpy array contains the 3D coordinates of 25 joints of a person in a frame. The shape of the numpy array is (T, 25, 3), where T is the number of frames in the video. The skeleton data is stored in the form of numpy arrays to reduce the time taken to load the data. 

## Data Augmentation
- The skeleton data can be augmented by occluding the joints of the skeleton. The code for this can be found here: [Skeleton Data Augmentation](./data_mgmt/datasets/ntu_dataset.py). 

# Training
- The training code uses the [PyTorch](https://pytorch.org/) framework.
- To start training, run the following command:
    ```bash
    python main.py --dataset ./dataset
    ```
- Other arguments can be found in the [main.py](./main.py) file.
- The trained models are stored in the `./output` folder.
- The hyperparameters of the model can be changed in the [./config/model.json](./config/model.json) file.
- To augment the data, pass the `--occlude` argument to the training script.