# Traditional Machine Learning (ML) vs Deep Learning (DL)
## _Powered by The Deep Sleeping Crew (Group6)_
**`Deep Learning`** is a subset of **`Machine Learning`** which improves so fast to process big and complex data today. The spotlight focus on deep learning. Many people may forget about traditional machine learning. This experiment aims to compare the performance of prediction between **`Traditional Machine Learning (ML)`** and **`Multilayer Perceptron (MLP)`**. Including hyperparameters tuning, modeling. We expect a sensational result and a good experience for our team members. 
## 1. Dataset
Before modeling, the dataset selection is the important portion of the comparison between Multilayer Perceptron (MLP) and traditional machine learning. First, we prepared the **`Wine dataset`** but it’s too small for training and testing with Multilayer Perceptron (MLP) model. Let’s say the size of the dataset is the reason why we need deep learning. 

Then, the **`Adult dataset`** from UCI is so interesting and big enough for our experiment. this dataset predict whether income exceeds $50K/yr based on census data (48,842 rows). Also known as the **`Census Income`** dataset.

Ref: https://archive.ics.uci.edu/ml/datasets/Adult

The first part of this study is Exploratory data analysis for understanding the character and pattern of the dataset such as shape, size, type of data of each attribute, correctness, missing value, etc. by statistics method and data visualization. 

## 2. Data Pre-processing

Data Preparation for training, our team members try sampling several methods of pre-processing and observe the result for testing with the random Multilayer Perceptron (MLP) models. Next, we discuss self-experiment and found two interesting pre-processing methods and possible to use.

Additional from **`correctness`** and **`missing value`**, this dataset needs to be managed the difference of many attributes including numerical and categorical values. Usually, traditional machine learning requires a numerical format for training and prediction. So we have to **`convert the categorical values to numerical values`** such as WorkClass, Education, Marital-Status, Occupation, Relationship. 

  

Finally, we combine the 2 methods and get a better result  
[test with model 97-99% & dropout 0.3]

## 3. Experiment result and discussion
### 3.1 Multilayer Perceptron (MLP)

3.1.1 Several combinations of hyperparameter comparisons 

In this process, we need to design and tune the hyperparameters of the Multilayer Perceptron (MLP) network. Due to deep learning having a black-box characteristic, we need to tune the hyperparameters by trial-and-error method for designing the most suitable model for this dataset. All hyperparameters are prioritized and randomly observed in the following table. 

| Groups of hyperparameter tuning | List of hyperparameters | Hyperparameter ranges |
| ------ | ------ | ------ |
| Major | 1) The number of neurons node in each layer | 1-10,000 |
|| 2) The activation function | Sigmoid, ReLU |
|| 3) Batch norm | Before/After Act.Func |
|| 4) Initial learning rate | 10-10,000 |
|| 5) Initial weight | 0.0001-0.1 |
| Minor | 1) Batch size |
|| 2) Drop out | None, after hidden layers
|| 3) Optimizer | ADAM, SGD, RMSProp

From our experiment, we have got interesting ideas for hyperparameter tuning. 

1) If The number of neurons node in each layer is over 500 nodes, there is no significant improvement. 

2) If Initial Weight = 0 at all hidden layers, the sequence will not converge, fixed weights by defining both seed & initial weights. 

3) If the learning rate is too small, the sequence will converge slowly.

3.1.2 What is the best one?

Accuracy:			99.97% 

Recall:				99.97% 

Precision:			99.97% 

F1 Score:			99.97%

### 3.2 Traditional Machine Learning (ML)

For training this dataset to traditional machine learning, we use Scikit-learn, the popular library in Python. The algorithm we select **`Logistic Regression (LR)`**, **`Support Vector Machine (SVM)`**, **`Random Forest (RF)`**, and **`K-Nearest Neighbor (KNN)`** for training and making predictions.

| Algorithm | Accuracy | Recall | Precision | F1 Score |
| ------ | ------ | ------ | ------ | ------ |
| Logistic Regression (LR) | 0.819 | 0.821 | 0.820 | 0.819 |
| Support Vector Machine (SVM) | 0.834 | 0.837 | 0.834 | 0.834 |
| Random Forest (RF) | 0.842 | 0.847 | 0.842 | 0.842 |
| K-Nearest Neighbor (KNN | 0.842 | 0.847 | 0.842 | 0.842 |

## 4. Conclusion

### 4.1 recommendation (using MLP vs. using traditional ML for structure data) 
