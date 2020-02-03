from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
import os
from sklearn import datasets
import pandas as pd
import seaborn as sns


# Download a dataset
# First way
!curl http: // archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data - o ./iris.data

# Second way with scikitlearn
iris = datasets.load_iris()

# Read the file and print the 12 first lines
iris2 = pd.read_csv("./iris.data", sep=',')
iris2.columns = ["sepal_l", "sepal_w", "petal_l", "petal_w", "species"]
iris2.head(12)

# Print the 12 last lines
iris2.tail(12)

# Give the number of categories
# The list of species
iris2.species.unique()
# The number of species
iris2.species.nunique()

# Give the number of observations per category
iris2.species.value_counts()

# Get a subset of the data (iris setosa)
iris_setosa = iris2[iris2["species"] == "Iris-setosa"]
iris_setosa.head()
iris_versicolor = iris2[iris2["species"] == "Iris-versicolor"]
iris_versicolor.head()
iris_virginica = iris2[iris2["species"] == "Iris-virginica"]
iris_virginica.head()

# Save the subsets to disk, read from disk and then merge them
iris_setosa.to_csv("./setosa.csv")
iris_versicolor.to_csv("./versicolor.csv")
iris_virginica.to_csv("./virginica.csv")

# OR


def to_csv(arr_df, arr_filenames):
    for idx, dataframe in enumerate(arr_df):
        dataframe.to_csv(arr_filenames[idx])
    print("it works!")


array_dataframes = [iris_setosa, iris_versicolor, iris_virginica]
array_names = ["./setosa_func.csv", "./versicolor_func.csv", "./virginica_func.csv"]
to_csv(array_dataframes, array_names)


def check_my_files(array_of_files):
    for file in array_of_files:
        if os.path.exists(file):  # ==True:
            print("the file exists: {0}".format(file))


check_my_files(array_names)


def read_array_csv_files(arr_files, separator=',', concat_axis=0):
    data_f = []
    for file in arr_files:
        data_f.append(pd.read_csv(file, sep=separator, index_col=0))
    res = pd.concat(data_f, axis=concat_axis).reset_index(drop=True)
    return res


df_iris_from_disk = read_array_csv_files(array_names)

# Zip
ex1 = ["Iris-", "Iris-", "Iris-"]
ex2 = ["setosa", "versicolor", "virginica"]
my_list = list(zip(ex1, ex2))
print(my_list)

for element in my_list:
    print(element)


# Plot data
df_iris_from_disk.head()
%matplotlib inline
sepal_length = df_iris_from_disk["sepal_l"]
sepal_length.hist()

# Correlation matrix
corr_mat = df_iris_from_disk.corr()

# Heatmap
heat_iris = sns.heatmap(corr_map, annot=True)
df_iris_from_disk.describe()

# Average
average_petal_width = df_iris_from_disk["petal_w"].mean()
print(average_petal_width)
data_above_average_petal_width = iris2[iris2["petal_w"] > average_petal_width]
print(data_above_average_petal_width)

average_sepal_length = df_iris_from_disk["sepal_l"].mean()
data_under_sepal_length = iris2[iris2["sepal_l"] < average_sepal_length]
print(data_under_sepal_length)

iris_versicolor_virginica = iris2[~iris2["species"].str.contains("Iris-setosa")]
# Or >> iris2[iris2["species"] != "Iris-setosa"]
print(iris_versicolor_virginica)

# Transform species names into IDs
res = iris2.replace(to_replace=["Iris-setosa", "iris_versicolor",
                                "iris_virginica"], value=[0, 1, 2])
res.head()

# OR
my_dict = {"Iris-setosa": 0, "iris_versicolor": 1, "iris_virginica": 2}
iris2["species"].apply(lambda x: my_dict[x])

iris2["flower_name"] = [x[x.find("-")+1:] for x in iris2["species"]]
print(iris2)

# OR
iris2["last_method"] = iris2["species"].str.split("-", expand=True).iloc[:, 1]
print(iris2)

# Train a classifier
# Splitting into train and test (30%)
train, test = train_test_split(iris2, test_size=0.3, random_state=3)
# Visual checkup
train.shape, test.shape
# Columns names
train.columns
# Creating independent variables and target variables
X_train, y_train, X_test, y_test = train[train.columns[:4]
                                         ], train["species"], test[test.columns[:4]], test["species"]
# Sanity check
assert(len(X_train) == len(y_train))
# Instantiation of the decision tree
clf = tree.DecisionTreeClassifier()
# Training the model
model = clf.fit(X_train, y_train)
# Predictions
predictions = model.predict(X_test)
print(predictions)

y_test.value_counts()
predictions[predictions == 'iris_versicolor'].shape
matrix_conf = confusion_matrix(y_test, predictions)
print(matrix_conf)
confusion = pd.DataFrame(matrix_conf, columns=["setosa", "versicolor", "virginica"], index=[
                         "setosa", "versicolor", "virginica"])
print(confusion)

# Evaluate the score
accuracy_score(predictions, y_test)
# OR
clf.score(X_test, y_test)

# ________________________________________________

# Wine quality dataset
!curl https: // archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv - o ./white.csv
!curl https: // archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv - o ./red.csv

white_wine = pd.read_csv("./white.csv", sep=';')
red_wine = pd.read_csv("./red.csv", sep=';')
white_wine.isna().sum()
red_wine.isna().sum()
white_wine.describe()
white_wine["color"] = 0
red_wine["color"] = 1
white_wine.head()
red_wine.head()

my_list = [white_wine, red_wine]
master_dataframe = pd.concat(my_list, axis=0)
master_dataframe.head()
master_dataframe["color"].value_counts()

# Splitting into train and test (30%)
train, test = train_test_split(master_dataframe, test_size=0.3, random_state=3)


def split_dataset(df_train, df_test, columns_names, target="color"):
    X_train = df_train[list(train.columns[:-1])].values
    y_train = df_train["colors"].values
    X_test = df_test[list(test.columns[:-1])].values
    y_test = df_test["colors"].values
    return(X_train, y_train, X_test, y_test)


list(train.columns[:-1])

X_train, y_train, X_test, y_test = split_dataset(train, test, train.columns[:-1])
clf = svm.SVC()
model = clf.fit(X_train, y_train)
predictions = model.predict(X_test)
model.score(X_test, y_test)
confusion_mat = confusion_matrix(y_test, predictions)
pd.DataFrame(confusion_mat, columns=["white_wine", "red_wine"], index=["white_wine", "red_wine"])

%matplotlib inline
pd.DataFrame(predictions).hist()

# ________________________________________________
# Titanic
