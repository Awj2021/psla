# Info for the Testing of the code

## Environment Setup
- conda create -n psla python=3.8
- conda activate psla
- conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
- pip install -r requirements.txt
# Due to the sox package, it is recommended to install sox using conda.
- conda install conda-forge::sox 

## Dataset Preparation
- Generate the fake-data using the methods: `/vol/research/wenjieProject/projects/audio/Stage1.ipynb`
- The dataset is combined with the original dataset and the fake dataset, double the size of the original dataset.
- This dataset is two class classification dataset, with the original dataset as class 0 and the fake dataset as class 1.
- Split the dataset into training and testing dataset, the ratio is 0.8:0.2. The tool is sklearn.model_selection.train_test_split.
- Without the mixup, noise setting, data augmentation.

## Training
- Traning branch is 'finetune', modification using the dataset 'LFS' is on the branch.
- [] Add the log to the wandb, the tool is wandb.
- [] Move the code to the server, the server is A6.
 
## Testing

