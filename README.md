# SmartPhoneAnalysis

# What Does Your Smartphone Know About You?

Welcome to the exploration of smartphone data, where we unravel the insights your device holds about your activities. In this GitHub repository, you'll find a Jupyter notebook (`what-does-your-smartphone-know-about-you.ipynb`) that delves into the wealth of information extracted from smartphone sensors.

## Notebook Overview

### 1. Import Libraries
To kick things off, we import essential libraries such as Pandas for data handling, NumPy for linear algebra, and various visualization tools like Matplotlib and Plotly. These tools set the stage for our in-depth analysis.

```python
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import init_notebook_mode, iplot
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from lightgbm import LGBMClassifier
from time import time

init_notebook_mode(connected=True)
```

### 2. Load Data
Next, we load the dataset comprising information gathered from smartphone sensors during daily activities. The dataset is divided into training and test sets, allowing for comprehensive analysis.

```python
# Load Datasets
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# Combine both dataframes
# Additional preprocessing steps are performed to set up the data for analysis
```

### 3. Dataset Exploration
We embark on an exploration journey, answering fundamental questions about the dataset:

- **Which Features Are There?**
  - Grouping and counting the main names of columns to get an overview of features.

```python
# Group and count main names of columns
# Displaying the count of features related to acceleration, gyroscope, and gravity
```

- **What Types Of Data Are There?**
  - Checking for null values and providing information on the data types.

```python
# Get null values and dataframe information
# Ensuring there are no missing values in the dataset
```

- **How Are The Labels Distributed?**
  - Visualizing the distribution of smartphone activity labels.

```python
# Plotting data to show the distribution of activity labels
# Confirming relatively equal distribution of labels
```

### 4. Activity Exploration
Moving deeper, we explore the separability of activities and evaluate a basic model's accuracy in predicting smartphone user activities.

```python
# Creating datasets, reducing dimensions, and visualizing separability
# Using a basic LGBMClassifier to predict activities with high accuracy
```

### 5. Participant Exploration
Now, let's focus on the individuals. We investigate how well participants can be distinguished based on their activities and analyze the duration the smartphone gathers data with impressive accuracy.

```python
# Assessing the separability of participants based on their activity
# Analyzing how long the smartphone takes to accurately predict participants
```

### 6. Exploring Personal Information
Diving into more personal aspects, we explore the walking frequency, frequencies in self-experiments, and conclude our findings.

```python
# Investigating walking frequency, self-experiment frequencies, and concluding insights
```

### 7. Conclusion
In the final section, we summarize our findings and draw conclusions from the extensive exploration of smartphone data.

Feel free to explore the notebook, suggest new ideas, and contribute to uncovering the secrets your smartphone holds about you!
