Simple ANN Classifier in PyTorch with Shapley explanations for Binary Classification - Base problem category as per Ready Tensor specifications.

- ANN
- shapley
- XAI
- HPT
- sklearn
- python
- pandas
- numpy
- scikit-optimize
- flask
- nginx
- uvicorn
- docker
- binary classification
- tensorflow
- keras

This is a Binary Classifier that uses Simple ANN implemented through PyTorch. Feature impacts are provided with Shapley values for model interpretability.

The data preprocessing step includes missing data imputation, standardization, one-hot encoding for categorical variables, datatype casting, etc. The missing categorical values are imputed using the most frequent value if they are rare. Otherwise if the missing value is frequent, they are give a "missing" label instead. Missing numerical values are imputed using the mean and a binary column is added to show a 'missing' indicator for the missing values. Numerical values are also scaled using a Yeo-Johnson transformation in order to get the data close to a Gaussian distribution.

Hyperparameter Tuning (HPT) is conducted by finding the optimal l1 and l2 regularization values as well as the optimal learning rate for SGD.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as email spam detection, customer churn, credit card fraud detection, cancer diagnosis, and titanic passanger survivor prediction.

This Binary Classifier is written using Python as its programming language. PyTorch is used to implement the main algorithm. Scikitlearn is used in the data preprocessing pipeline and model evaluation. Numpy, pandas, and feature_engine are used for the data preprocessing steps. SciKit-Optimize was used to handle the HPT. Flask + Nginx + gunicorn are used to provide web service which includes two endpoints- /ping for health check and /infer for predictions in real time.
