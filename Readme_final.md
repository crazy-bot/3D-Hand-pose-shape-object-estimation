# 3D Hand and Manipulated Object Pose and Shape Reconstruction using 3D Convolutional Networks

## Basic Setup
1. Install basic requirements:
    ```
    conda create -n python3.6 python=3.6
    source activate python3.6
    pip install requirements.txt
    ```
2. Download Models&code from the MANO website
    ```
    http://mano.is.tue.mpg.de
    ```
3. Assuming ${MANO_PATH} contains the path to where you unpacked the downloaded archive, use the provided script to setup the MANO folder as required.
    ```
    python setup_mano.py ${MANO_PATH}

Note: In the server required packages are installed and MANO folder is also included


**All the below scripts are to be run from "src" directory.**

### Training Procedure 
There are total 4 training scripts in the "src/final" directorcy which are named as "train_HO_PNet", "train_HO_VNet", "train_HO_SNet", "train_HO_PVSNet"
Corresponding to each training file there are .sh files.

For e.g train_HO_PNet.sh file contains below command:

CUDA_VISIBLE_DEVICES=0 python3 final/train_HO_PNet.py --save_ckpt <checkpoint folder> --multiGPU True

1. To resume training add --load_ckpt <checkpoint folder/checkpoint file> to the command inside .sh file.

2. As per the availability of GPUs kindly change, or add more GPUs like CUDA_VISIBLE_DEVICES=0,1,2,3

3. Refer the final/config.py to change any training parameters like batch size, no of epochs etc.


### validation of models
There are 4 validation scripts which are named as "valid_HO_PNet", "valid_HO_VNet", "valid_HO_SNet", "valid_HO_PVSNet".

* For pose and mesh results run "valid_HO_PVSNet" on "valid" and "evaluation" set.

* For only pose results run "valid_HO_PNet" on "valid" and "evaluation" set.

* For intermediate results "valid_HO_VNet", "valid_HO_SNet" only on "valid" set.

Please change "validset = valid/evaluation" accordingly in the src/final/config.py.

#### To run above validation scripts:

CUDA_VISIBLE_DEVICES=0 python3 final/valid_HO_PNet.py --saveJSON False --figMode "2D"

Kindly refer the scripts for all possible command line parser arguments:

--saveJSON : whether to save all the results as json file. It will be saved as "hand_<network name>_<dataset name>.json" and "obj_<network name>_<dataset name>.json"

--figMode :value can be ['2D', '3DPC','3DMesh','']. It will save the result as per the mentioned format in the output/<network name> folder

** For VNet the --saveJSON argument is not valid. Hence remove it during running those scripts.

### validation of models in pipeline (output from previous network is fed to next)
To run inference on pipeline run "inference_pipeline.py" on either "valid" and "evaluation" set. please change "validset" accordingly in the final/config.py

To run above validation scripts run for e.g:
CUDA_VISIBLE_DEVICES=0 python3 final/inference_pipeline.py --saveJSON False --figMode "2D"

***When validset is "valid" then in the console you can see per joint loss for each sample as below:
SB12/0974

hand points loss: 0.004708770243116137
hand mesh loss: 0.0062248850978511615
obj points loss: 0.005834763477793131
obj mesh loss: 0.007060478370763217

######################### evaluation of results as per codelab challenge ################
To evaluate object complying with the codelab challenge format run "eval_codelab_object.py". For evaluation we need '.json' file containing all the results created follwing one of the validation scripts. 

Assuming we have "pip_hand_snetv6_valid.json" and "pip_obj_snetv6_valid.json" please set in the src/final/config.py as follows:

evalset = 'valid'

objJson_file = 'pip_obj_snetv6_valid.json'

####################### to visualize data ########################
1. visualize as per official Honnotate git hub:
please refer the official github link:https://github.com/shreyashampali/ho3d


###################################### Thanks. Please contact for any doubts ##########################



