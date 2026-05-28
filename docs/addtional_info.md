# Additional Info

## ML vs DL Training Workflow

The high-level process is the same for both: prepare data, feature engineer, split, train, evaluate, retrain loop. The difference is ML requires manual feature engineering while deep learning learns features automatically.

## Epochs

A deep learning concept only. One epoch = one full pass through all training data. Traditional ML (Random Forest, SVM) does not use epochs.

## Batches

Also deep learning specific. Batch size = how many samples processed at once before updating weights. Default is usually 32. With 8,000 images and batch size 32, one epoch = 250 batches.

## Loss Function vs Weights

Loss function measures how wrong the model is. Weights are the numbers inside the model that get adjusted to reduce the loss. They work together every batch.