## Boosting Semi-supervised Image Segmentation with Global and Local Mutual Information Regularization

This is the minimal reproduce code for the paper "Boosting Semi-supervised Image Segmentation with Global
and Local Mutual Information Regularization" recently submitted to a journal. 


We release the code, together with the well-preprocessed `ACDC` dataset for reviewers. The dataset should be keep private based on the dataset agreement and I will delete it once the reviewer process finishes.

Our code is based on `deepclustering2` package, which is a personal research framework. It will automatically install all dependency on a conda virtual environment and without resorting to `requirement.txt`.


-----------------
##### Basic script for setting a conda-based virtual environment.
```bash
conda create -p ./venv python=3.7

conda activate ./venv

conda install pytorch torchvision cudatoolkit=10.2 -c pytorch  # install pytorch 1.6.0
pip install deepclustering2-2.0.0-py3-none-any.whl
python setup.py install  
# all packages should be set properly automatically.
```
In case of failure of running the experiments, please refer to `requirement.txt` to see the packages

----------------
##### Basic script to start training 
```bash
cd semi_seg
# our proposed method
python main.py  Data.labeled_data_ratio=0.05  Data.unlabeled_data_ratio=0.95  Trainer.num_batches=300  Trainer.max_epoch=100  Data.name=acdc  Arch.num_classes=4  Optim.lr=0.0000001000 Trainer.name=udaiic Trainer.save_dir=udaiic/10.0_0.1  IICRegParameters.weight=0.1 UDARegCriterion.weight=10.0 
# ps baseline (lower bound)
python main.py  Data.labeled_data_ratio=0.05  Data.unlabeled_data_ratio=0.95  Trainer.num_batches=300  Trainer.max_epoch=100  Data.name=acdc  Arch.num_classes=4  Optim.lr=0.0000001000 Trainer.name=partial Trainer.save_dir=ps  
# fs baseline (upper bound)
python main.py  Data.labeled_data_ratio=1.0  Data.unlabeled_data_ratio=0.0  Trainer.num_batches=300  Trainer.max_epoch=100  Data.name=acdc  Arch.num_classes=4  Optim.lr=0.0000001000 Trainer.name=partial Trainer.save_dir=fs  
```
One can change the parameters on the cmd if needed.
Please refer to the default configuration in `config/semi.yaml` all set of controllable hyperparameters. All of them can be changed using cmd as above.


---------------------
##### Performance
Based on different random seed, the ACDC performance varies within 1% in terms of DSC. Above scripts gives a DSC of ~85.5% for our proposed method vs 62.0% for ps and 89.2% for fs.




