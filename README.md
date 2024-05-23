 # Private Regression via Data-Dependent Sufficient Statistics

## Requirements
python 3.10
```setup
pip install -r requirements.txt
```

## Running linear and logistic regression experiments

To run the experiments presented in the paper, you can use the line commands in "line_commands.txt". For example 
(change "dataset" as needed):

LINEAR REGRESSION:
```
python main_lin.py --dataset adult --method public --num_experiments 5 --n_limit 50000 --one_hot True --scale_y True
python main_lin.py --dataset adult --method adassp --num_experiments 5 --n_limit 50000 --one_hot True --scale_y True
python main_lin.py --dataset adult --method aim --num_experiments 5 --n_limit 50000 --one_hot True --scale_y True
```

LOGISTIC REGRESSION:
```
python main_log.py --dataset adult --method public --num_experiments 5 --n_limit 50000 --one_hot True
python main_log.py --dataset adult --method genobjpert --num_experiments 5 --n_limit 50000 --one_hot True
python main_log.py --dataset adult --method aim --num_experiments 5 --n_limit 50000 --one_hot True
```

Results are progressively saved in .csv format a folder named [dataset]_linear or [dataset]_logistic depending on the model.
.csv file names follow this format:
```
f'{dataset}_{method}_{epsilon}_{trial_num}_{num_experiments}exps_{data_size_limit}limit_{seed}seed.csv'
```

References for external packages and assets used in this repo:
```
mbi and private-pgm: https://github.com/ryan112358/private-pgm
hdmm: https://github.com/ryan112358/hdmm-1
hd-datasets-master: https://github.com/ryan112358/hd-datasets
ACS data: https://github.com/socialfoundations/folktables
```
