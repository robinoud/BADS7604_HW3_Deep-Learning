# Traditional Machine Learning (ML) vs Deep Learning (DL)
## _Powered by The Deep Sleeping Crew (Group6)_
**`Deep Learning`** is a subset of **`Machine Learning`** which improves so fast to process huge and complex data nowadays. Since the spotlight focuses on deep learning, most users may forget about traditional machine learning. This experiment aims to compare the performance of prediction between **`Traditional Machine Learning (ML)`** and **`Multilayer Perceptron (MLP)`**, including hyperparameters tuning and modeling. We expect a sensational result and a good experience for our team members and for readers. 
## 1. Dataset
Before modeling, the dataset selection is the crucial part of the comparison between Multilayer Perceptron (MLP) and traditional machine learning. First, we prepared the **`Wine dataset`** but it’s too small for training and testing with Multilayer Perceptron (MLP) model. It is obvious that a huge number of input data is one of the key reasons why we need deep learning. 

Then, the **`Adult dataset`** from UCI is so interesting and big enough for our experiment. this dataset predict whether income exceeds $50K/yr based on census data (48,842 rows). Also known as the **`Census Income dataset`**.

Ref: https://archive.ics.uci.edu/ml/datasets/Adult

The first part of this study is Exploratory data analysis for understanding the character and pattern of the dataset such as shape, size, type of data of each attribute, correctness, missing value, etc. by statistics method and data visualization.

<img src="https://github.com/robinoud/BADS7604_Deep-Learning/blob/4ec9a052a7fe7526bcd967b542eebf7e1574f438/Workflow%20of%20the%20experiment.png" style="width:600px;"/>

## 2. Data Pre-processing

Data Preparation for training, our team members try sampling several methods of pre-processing and observe the result for testing with the random Multilayer Perceptron (MLP) models. Next, we discuss self-experiment and found two interesting pre-processing methods and possible to use.

Additional from **`correctness`** and **`missing value`**, this dataset needs to be managed the difference of many attributes including numerical and categorical values. Usually, traditional machine learning requires a numerical format for training and prediction. So we have to **`convert the categorical values to numerical values`** such as WorkClass, Education, Marital-Status, Occupation, Relationship. 

  

Finally, we combine the 2 methods and get a better result  
[test with model 84-84% & dropout 0.3]

<img src="https://github.com/robinoud/BADS7604_Deep-Learning/blob/4ec9a052a7fe7526bcd967b542eebf7e1574f438/Data%20preperation%20steps.png" style="width:600px;"/>

## 3. Experiment result and discussion
### 3.1 Multilayer Perceptron (MLP)

#### 3.1.1 Several combinations of hyperparameter comparisons 

In this process, we need to **`design and tune the hyperparameters`** of the Multilayer Perceptron (MLP) network. Due to deep learning having a black-box characteristic, we need to tune the hyperparameters by **`trial-and-error method`** for designing the most suitable model for this dataset. All hyperparameters are **`prioritized`** and **`randomly observed`** in the following table. 

| Groups of hyperparameter tuning | List of hyperparameters | Hyperparameter ranges |
| :------: | ------ | ------ |
| Major | 1) The number of neurons node in each layer | 1-10,000 |
|| 2) The activation function | Sigmoid, ReLU, Tanh, Softplus |
|| 3) Batch norm | Before/After Act.Func |
|| 4) Initial learning rate | 0.0001-0.1  |
|| 5) Initial weight |
| Minor | 1) Batch size |
|| 2) Drop out | None, after hidden layers
|| 3) Optimizer | ADAM, SGD, RMSProp

From our experiment, we have got interesting ideas for hyperparameter tuning. 

- If **`the number of neurons node`** in each layer is **`over 500 nodes`**, there is **`no significant improvement`**. 

- If **`Initial Weight = 0`** at all hidden layers, the sequence will **`not converge`**, fixed weights by defining both seed & initial weights. 

- If **`the learning rate is too small`**, the sequence will **`converge slowly`**.

#### 3.1.2 What is the best one?

<img src="https://github.com/robinoud/BADS7604_Deep-Learning/blob/804533861f928d325a4bb1752069c7b39894917e/Model%20Summary.jpg" style="width:600px;"/>

|Indicator|Score|
| ------ | :------: |
|Accuracy|84.20%|
|Recall|84.51%|
|Precision|84.24%|
|F1 Score|84.17%|

### 3.2 Traditional Machine Learning (ML)

For training this dataset to traditional machine learning, we use Scikit-learn, the popular library in Python. The algorithm we select **`Logistic Regression (LR)`**, **`Support Vector Machine (SVM)`**, **`Random Forest (RF)`**, and **`K-Nearest Neighbor (KNN)`** for training and making predictions.

| Algorithm | Accuracy | Recall | Precision | F1 Score |
| ------ | :------: | :------: | :------: | :------: |
| Logistic Regression (LR) | 0.822 | 0.823 | 0.822 | 0.821 |
| Support Vector Machine (SVM) | 0.832 | 0.832 | 0.833 | 0.832 |
| Random Forest (RF) | 0.842 | 0.847 | 0.842 | 0.841 |
| K-Nearest Neighbor (KNN | 0.842 | 0.847 | 0.842 | 0.841 |

## 4. Conclusion

| Implementation Details | Multilayer Perceptron (MLP)  | Traditional Machine Learning (ML) |
| ------ | :------: | :------: |
| 1. A huge number of input data required | Yes | No |
| 2. Data pre-processing required | Yes | Yes |
| 3. Suffers from black-box issue | Yes | No, thanks to its model representation  |
| 4. High computing power consuming | Yes | No |
| 5. A lot of parameters and training time consuming | Yes | No |

| Performance Measures | Our self-designed MLP | Logistic Regression (LR) | Support Vector Machine (SVM) | Random Forest (RF) | K-Nearest Neighbor (KNN) |
| ------ | :------: | :------: | :------: | :------: | :------: |
| 1. Accuracy | 0.849 | 0.819 | 0.834 | 0.842 | 0.842 | 
| 2. Recall | 0.852 | 0.821 | 0.837 | 0.847 | 0.847 | 
| 3. Precision | 0.850 | 0.820 | 0.834 | 0.842 | 0.842 |
| 4. F1 Score | 0.849 | 0.819 | 0.834 | 0.842 | 0.842 |

### 4.1 recommendation (using MLP vs. using traditional ML for structure data)

## End Credit
### _The Deep Sleeping Crew (Group6) Contribution - Uniform_
**`16.67%`** 🍕 - **`6310422057`** Natdanai Thedwichienchai

**`16.67%`** 🍕 - **`6310422061`** Wuthipoom Kunaborimas

**`16.67%`** 🍕 - **`6310422063`** Nuj Lael

**`16.67%`** 🍕 - **`6310422064`** Krisna Pintong

**`16.67%`** 🍕 - **`6310422065`** Songpol Bunyang

**`16.67%`** 🍕 - **`6310422069`** Phawit Boonrat

<img src="https://th-test-11.slatic.net/p/49b63f074bd226e6871cc97c5525fc15.jpg" alt="drawing" style="width:200px;"/>
