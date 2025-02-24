# Evading Anti-Malware Measures in Windows PE with Counterfactual Explanations

The repository contains code refered to the work:

_Evading Anti-Malware Measures in Windows PE with Counterfactual Explanations_
 
Please cite our work if you find it useful for your research and work.

## Data

1) The dataset used for OLIVANDER can be found at the following [LINK](https://practicalsecurityanalytics.com/pe-malware-machine-learning-dataset/). In the config.ini, point at this folder using the parameter "pe_folder". You need to download the complete dataset to proceed in the generation of the Adversarial Samples.

2) Dumps of lief features dataset and counterfactuals generated by the method are also available [HERE](https://drive.google.com/drive/folders/1WJFbRPP9dEFccRM5J7kyraO1Fuy9tB3y?usp=sharing). In the config.ini, point at this file on "counterfactual_path"
3) For each configuration of OLIVANDER, GAMMA and AMG, SHA256 hashes are been made available [HERE](https://drive.google.com/drive/folders/1uihbJi-Wvrkgc_RMo0WfzkkJGerzTVib?usp=sharing). In the repository you can find also the VirusTotal dump used in the study.


## Code requirements
The code relies on the following python3.6+ libs.
Packages needed are:
* liblief==0.9
* numpy==1.19.2
* dice-ml==0.11
* keras==2.6.0
* tensorflow==2.6.2
* scikit-learn==0.24.2
* pandas==1.1.5
* ember==0.1.0
* scipy==1.5.2
* secml-malware==0.2.4.1

## Structure

The repository contains the following scripts and folders:
* main.py:  script to execute OLIVANDER_PADDING or OLIVANDER_INJECTION
* /results: results folder that OLIVANDER uses to store the adversarial PEs 
* /pe_folder: folder containing the PEs downloaded from [LINK](https://practicalsecurityanalytics.com/pe-malware-machine-learning-dataset/)
* /pickle: folder used to save the results of the method. Specifically, a folder with a name containing the specific parameters used will be created. **The actual evasive PE files will be referred to as "final-"**
* /libs: script used by the method. 
* /models: it contain the DNN used as oracle
* GAMMA.py: script to execute GAMMA_PADDING or GAMMA_INJECTION
* adversarial_training.py : adversarial training using different epsilon.  
* config.ini: file containing the paths used by the method
# Parameters
A config.ini template:
```ini

[SETTINGS]
# Counterfactuals Dump
counterfactual_path=pickle/counterfactuals.pickle
# Path to the folder containing the dataset. In particular, the folder must contain the samples.csv and the samples folder with all the executable pe files 
pe_folder=/pe_folder/
# Dump of the extracted lief dataset 
lief_dataset=pickle/dataset_lief.pickle

[GAMMA]
# Path required by GAMMA to take goodware executables
goodware_folder=/goodware/
#gamma_manipulation_type can be "injection" or "padding"
gamma_manipulation_type=padding

[ADVERSARIAL_TRAINING]
EPS_ARRAY=[0.01, 0.001, 0.0001, 0.1]

```





## How to use

The python script supports the following parameters:
* --mode: 
  * generate_dataset: extracts the lief features from the original PE files, generate the counterfactuals and finally, it creates the adversarial samples
  * load_dataset: uses the already extracted lief features from the "lief_dataset" folder and execute the method. **Please take into account that you require also the data in the "pe_folder" to proceed the generation of adversarial samples.**
  * load_counterfactual: loads the counterfactual examples from the "counterfactual_path" folder and generates the adversarial samples. **Please take into account that you require also the data in the "pe_folder" to proceed the generation of adversarial samples.**
* injection_type: 
  * section:  use OLIVANDER injection, default
  * padding: use OLIVANDER padding
* --eta: [10,100,1000]
* --c: [10,100,1000]
* --offsetmin: min offset used to select from the true positive extracted from the testing set, the examples that will be used  for the  counterfactual generation phase and to create OLIVANDER adversarial sample. Default:100
* --offsetmax: max offset used to select the examples that will be used for the counterfactual generation phase and to create OLIVANDER adversarial sample. Default:200

## Workflow
![Workflow Image](Workflow.png)

The OLIVANDER adversarial generation process is composed by the following steps:
* Step A: Computing LIEF feature encoding from the original PE dataset
* Step B: Generating Counterfactuals for each PE to consider
* Step C: Apply the OLIVANDER process using the Counterfactuals computed to generate Adversarial Samples 


## Use Case to recreate the LIEF encoding and the COUNTERFACTUALS
### Step A->B->C

To start from the scratch, you can set the mode parameter to "generate dataset" to compute the lief encoding dataset, the creation of counterfactuals and the adversarial generation using OLIVANDER:
```console
python3 main.py --mode generate_dataset --injection_type section --step 1000 --c 100 # default using OLIVANDER injection
python3 main.py --mode generate_dataset --injection_type padding --step 1000 --c 100  # default using OLIVANDER padding
```

## Use Case for Reproducibility
### Step C

To replicate the selection of the 100 examples used in the study, you can find the dump reported [HERE](https://drive.google.com/drive/folders/1WJFbRPP9dEFccRM5J7kyraO1Fuy9tB3y?usp=sharing) containing also the same training set and testing set in the LIEF format. 
To create the adversarial samples using OLIVANDER INJECTION using the same set used in the study, set the mode parameter as reported below:
```console
python3 main.py --mode load_counterfactual --injection_type section --step 1000 --c 100  #default setup
```
To create the adversarial samples using OLIVANDER PADDING using the same set used in the study, set the mode parameter as reported below::
```console
python3 main.py --mode load_counterfactual --injection_type padding --step 1000 --c 100 
```
### Step B->C
To create counterfactuals using the LIEF  [DUMP](https://drive.google.com/drive/folders/1WJFbRPP9dEFccRM5J7kyraO1Fuy9tB3y?usp=sharing)  for reproducibility, set the mode parameter to "load_dataset":
```console
python3 main.py --mode load_dataset --injection_type padding --step 1000 --c 100 # default using OLIVANDER padding 
python3 main.py --mode load_dataset --injection_type section --step 1000 --c 100 # default using OLIVANDER injectiom 
```
Change the offset to select the examples from the true positive set that will be used to generate counterfactuals and OLIVANDER adversarial samples.
```console
python3 main.py --mode load_dataset --injection_type padding --step 1000 --c 100 --offsetmin 100 --offsetmax 200  # default using OLIVANDER padding  
python3 main.py --mode load_dataset --injection_type section --step 1000 --c 100 --offsetmin 100 --offsetmax 200  # default using OLIVANDER injection 
```

## GAMMA
To use GAMMA, please change the parameter "goodware_folder" in the config.ini in order to link to the folder where the goodware files are located.
Change GAMMA injection type using the "gamma_manipulation_type" parameter in the config.ini using "injection" or "padding"

```console
python3 GAMMA.py   
```
## AMG
To use AMG, please refer to the official [github repository ](https://github.com/matouskozak/AMG). Please install the library LIEF==0.9.0 for consistent results. 

## ADVERSARIAL TRAINING
To use adversarial training, change the list of epsilon in the config.ini, editing the parameter "EPS_ARRAY" like the following example:
```ini
[ADVERSARIAL_TRAINING]
EPS_ARRAY=[0.01, 0.001, 0.0001, 0.1]
```
To run the adversarial training configuration, please use the following command:
```console
python3 adversarial_training.py   
```

