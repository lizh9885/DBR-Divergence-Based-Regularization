# DBR-Divergence-Based-Regularization




## requirements
```
pip install -r requirements.txt
```


## Train the identification model

In run.py run "train_and_evaluate_orign_model(args, logger, device)" ,the ckpt will be saved as orign_model_path in the corresponding dir

## top-k index output

run index_output.py, the new training set will be saved as "train_withindex.json"

## Train the bias-only model

In run.py, run"train_and_evaluate_bias_head(args, logger, device, orign_model_path)", the ckpt will be saved as bias_model_path in the corresponding dir


## Trian the debiased model
In run.py run"train_and_evaluate_final_model(args, logger, device, orign_model_path, bias_model_path)"
