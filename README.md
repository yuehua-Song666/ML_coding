# ML_coding
1. 2022_6_26 Day_1 Done
- class sklearn.impute.SimpleImputer(*, missing_values=nan, strategy='mean', fill_value=None, verbose='deprecated', copy=True, add_indicator=False)
  - Imputation transformer for completing missing values.
- class sklearn.preprocessing.OneHotEncoder(*, categories='auto', drop=None, sparse=True, dtype=<class 'numpy.float64'>, handle_unknown='error', min_frequency=None, max_categories=None)
  - Encode categorical features as a one-hot numeric array.
- class sklearn.preprocessing.LabelEncoder
  - Encode target labels with value between 0 and n_classes-1.
- class sklearn.compose.ColumnTransformer(transformers, *, remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False, verbose_feature_names_out=True)
- sklearn.model_selection.train_test_split(*arrays, test_size=None, train_size=None, random_state=None, shuffle=True, stratify=None)
  - Split arrays or matrices into random train and test subsets.
- class sklearn.preprocessing.StandardScaler(*, copy=True, with_mean=True, with_std=True)
  - Standardize features by removing the mean and scaling to unit variance.

2. 2022_6_27 Day_2 Done

3. 2022_6_28 Day_3 Done

4. 2022_7_1 Day_4, Day_5, Day_6 Done
- Logistic Regression
  - Predict the group to which the current object under observation belongs to.
  - Give a discrete binary outcome between 0 and 1
- Difference between Logistic Regress and Linear Regression
  - Logistic Regress -- discrete outcome
  - Linear Regression -- continuous outcome
- from sklearn.metrics import confusion_matrix
  - confusion_matrix -- The diagonal elements represent the number of points for which the predicted label is equal to the true label, while off-diagonal elements are those that are mislabeled by the classifier. The higher the diagonal values of the confusion matrix the better, indicating many correct predictions.
