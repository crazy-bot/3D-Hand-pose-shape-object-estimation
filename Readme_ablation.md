# Ablation Study

### Basic Setup
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

**Note: In the server required packages are installed and MANO folder is also included**

**All the below scripts are to be run from "src" directory.**

### Training Procedure
The "src/ablation" directory includes total 9 training scripts which are named as "train_<Network>". The networks are described as below:

| Network | Description |
| --- | --- |
| HO_PNet | Individual Pose network: Depth voxel –> Pose HM |
| HO_VNet | Individual Voxel network: Depth voxel + Pose HM → shape voxel |
| HO_SNet\_v1 | Individual Shape network: Depth voxel –> shape |
| HO_SNet\_v2 | Individual Shape network: Depth voxel + Pose HM –> shape |
| HO_SNet\_v3 | Individual Shape network: Depth voxel + Pose HM + shape voxel –> shape |
| HO_SNet\_v4 | Individual Shape network: Enriched Depth voxel + Pose HM  –> shape |
| study1 | combined network: HO_PNet + HO\_SNet\_v2  –> Pose HM + shape |
| study2 | combined network: HO_PNet + HO\_SNet\_v3  –> Pose HM + shape |
| study3 | combined network: HO_PNet + HO\_SNet\_v4  –> Pose HM + shape |

To train any network modify the train.sh file as for e.g:

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 src/train_HO_PSNet.py --save_ckpt <checkpoint folder> --multiGPU True

1. To resume training add --load_ckpt <checkpoint folder/checkpoint file> to the command inside .sh file.

2. As per the availability of GPUs kindly change, or add more GPUs like CUDA_VISIBLE_DEVICES=0,1,2,3

3. Refer the final/config.py to change any training parameters like batch size, no of epochs etc.


### validation of models

Corresponding to each individual network 9 validation scripts are named as "valid\_<Network>" . For validation of baseline run "valid_Baseline".

* Baseline, Study1, study2, study3 can be run on "valid" and "evaluation" set. 

* For intermediate results run other networks only on "valid" set.

please change "validset" accordingly in the src/ablation/config.py

#### To run above validation scripts:

CUDA_VISIBLE_DEVICES=0 python3 ablation/valid_HO_PNet.py --saveJSON False --figMode "2D"

Kindly refer the scripts for all possible command line parser arguments:

--saveJSON : whether to save all the results as json file. It will be saved as "hand_<network name>_<dataset name>.json" and "obj_<network name>_<dataset name>.json"

--figMode :value can be ['2D', '3DPC','3DMesh','']. It will save the result as per the mentioned format in the output/<network name> folder

** For VNet the --saveJSON argument is not valid. Hence remove it during running those scripts.

### validation of models in pipeline (output from previous network is fed to next)

There are 3 pipeline validation scripts which are named as "pipeline_<Network>". 

To run above validation scripts run for e.g:

CUDA_VISIBLE_DEVICES=0 python3 ablation/pipeline_study1.py --saveJSON False --figMode "2D"

***When validset is "valid" then in the console you can see per joint loss for each sample as below:
SB12/0974

hand points loss: 0.004708770243116137
hand mesh loss: 0.0062248850978511615
obj points loss: 0.005834763477793131
obj mesh loss: 0.007060478370763217
######################### evaluation of results as per codelab challenge ################
For evaluation we need .json file containing all the results created follwing one of the above scripts.

Assuming we have "pip_hand_snetv6_valid.json" and "pip_obj_snetv6_valid.json" please set in the src/ablation/config.py as follows:

evalset = 'valid'
objJson_file = 'pip_obj_snetv6_valid.json'
handJson_file = 'pip_hand_snetv6_valid.json'

Then run ablation/eval_hand.py and ablation/eval_object.py respectively for hand and object evaluation

####################### to visualize data ########################
1. visualize as per official Honnotate git hub:
please refer the official github link:https://github.com/shreyashampali/ho3d


###################################### Thanks. Please contact for any doubts ##########################



