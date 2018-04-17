# Machine learning

**Machine learning** is a field of computer science that uses statistical techniques to give computers the ability to "learn" (i.e. progressively improve performance on a specific task) with data.

A machine learning system is trained rather than being explicitly programmed.
It's presented with many examples relevant to a task, and it finds statistical structure in these examples
that eventually allow the system to come up with rules for automating the task.

A machine learning model transforms the input data into meaningful outputs, a process that is "learned" from exposure to known examples of inputs and outputs. Its central problem is therefore to meaningfully transform data, to learn meaningful representations of the data. Learning describes the automatic search process for better representations of the data.

## Two types of machine learning

Machine-learning problems fall into two camps: **supervised** and **unsupervised**.

**Supervised** problems are the ones in which you have access to the target variable.
Here, humans input data as well as the answers expected from the data, and the machine learning algorithm discovers the rules.

**Unsupervised** problems are ones in which there's no identified target variable.
Here, the training process tries to find hidden structure in unlabaled data.

## Use cases for supervised machine learning

* **classification**: determine the discrete class to which each individual belongs; examples: spam filtering, fraud detection, detection of manufacturing defects
* **regression**: predict the real-valued output for each individual; examples: stock-market prediction, demand forecasting, weather forecasting, sports prediction, price estimation, risk management
* **recommendation**: predict which alternatives a user would prefer; examples: product recommendation, job recruiting, online dating, content recommendation
* **imputation**: infer the values of missing input data; examples: incomplete patient medical records, missing customer data

## Use cases for unsupervised learning

* **clustering**: use the input features to discover natural groupings in the data and to divide the data into those groups; methods: k-means, Gaussian mixture models, hierarchical clustering
* **dimensionality reduction**: transform the input features into a small number of coordinates that capture most of the variability of the data; methods: principal component analysis, multidimensional scaling, manifold learning

## Machine learning workflow

* data preparation
* model building
* evaluation
* optimization
* prediction
