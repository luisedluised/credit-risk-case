# German Credit Risk - Finding the highest bang for buck

![Alt text][def]


On this project I implemented a classification model using a Bayes Search tuned LGBM on the German Credit Risk dataset. Having selected the model by the roc-auc criterion I took a further look into the model classification threshold business consequences. For that I estimated model expected returns based on the numbers of avoided bad customers and lost good customers, under some assumptions which I outlined on the Jupyter Notebooks.


[def]: figures/returns_versus_sensitivity_and_sensibility.png?raw=true "Classification threshold choice return"
