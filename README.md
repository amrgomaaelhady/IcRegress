# IcRegress #

IcRegress is an incremental learning approach specifically designed for regression problems.

The repository has the code for the implementation of this incremental learning method, named IcRegress, as introduced in the paper, "Looking for a better fit? An Incremental Learning Multimodal Object Referencing Framework adapting to Individual Drivers" at IUI 2024.

![alt text](https://github.com/amrgomaaelhady/IcRegress/blob/main/Fig.png)

- - -

## Conda Enviroment ##
- To set it up in conda, please use the simplified requirements.txt as follows:
```
conda create --name icregress --file requirements.txt
```
- - -

## How to use ##

The entire code is in one jupyter notebook named "Incremental_learning.ipynb" (there is another jupyter notebook called "Incremental_learning_per_participant.ipynb" to run per participant for convenience with the same structure). The code can be adjusted to different datasets and personalized groups.

The following parameters control the framework's initial training, incremental learning approach, and inference-only (i.e., prediction) functions. The datasets are kept in the "Data" Folder. 

You can choose whether to use CNN, LSTM or a Transformer network. There hyperparameters can be adjusted in their corresponding folders.

The training_type 'regress' and the regress_type 'regress_center' are the ones used in the paper, however, you can use to do direct classification or use a different type of regression.

The other hyperparameters are self explained. The default is train from scratch. To perform incremental learning or naive finetuning, you have to enable the flag "fine_tune_or_not" and give the appropriate "exemplar_ratio" (setting it to zero would mean a naive finetuning approach. To do prediction only without further training, the "inference_only" flag should be enabled. Corresponding checkpoints paths should be given after each flag as seen from the below code snippet.

```
folder_path = "../Data/"
mode = 'CNN' #CNN or LSTM or Transformer network
training_type = 'regress' #classify or regress
regress_type = 'regress_center' #regress_facade or regress_center
batch_size = 128
number_epochs = 20000
learning_rate = 0.0001
fine_tune_or_not = False
fine_tune_ckp_path = "checkpoint.pt"
inference_only = True
inference_ckp_path = "checkpoint_earlystop.pt"
exemplar_ratio = 0 # 2 is half, 4 is quarter and so on while 0 is naive finetuning
```

Finally, you run the entire code by calling the "multiple_runs_concatenate" function using the appropriate datasets, the function also contains the participants numbers that should be selected for training and testing so it should be adjusted accordingly. Additionally, the number of runs for each training should be set inside this function, however, only the last run checkpoint is currently being saved.

```
multiple_runs_concatenate('Per_Participant/Data_based_on_Speech_onset_P74.csv','/Data_based_on_Speech_onset.csv','y_train_indeces_priotirized.csv')

```

## Citation ##

- If you find this work helpful, please cite our paper:
```
@inproceedings{2024gomaaincremental,
author = {Gomaa, Amr and Reyes, Guillermo and Feld, Michael and Kr{\"u}ger, Antonio},
title = {Looking for a better fit? {A}n Incremental Learning Multimodal Object Referencing Framework adapting to Individual Drivers},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {},
doi = {},
booktitle = {Proceedings of the 29th International Conference on Intelligent User Interfaces},
location = {},
series = {IUI '24}
}

```
- - -
