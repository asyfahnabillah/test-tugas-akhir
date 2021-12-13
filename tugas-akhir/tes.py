import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

"""**Load Dataset**"""

uploaded = files.upload()
df = pd.read_csv('df.csv')
df.head()

df

df.info()


df.describe()

**EDA (EXPLORATORY DATA ANALYSIS )**

Proporsi Label Death_Event
#Memvisualisasikan data
#Menampilkan visualisasi bar plot dan pie chart untuk menghitung frekuensi dari kolom Di Diagnosis hipertensi dalam dataset data
f,ax=plt.subplots(1,2,figsize=(18,8))
df['DI DIAGNOSIS HIPERTENSI'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_xlabel('DI DIAGNOSIS HIPERTENSI')
ax[0].set_ylabel('JUMLAH')
sns.countplot('DI DIAGNOSIS HIPERTENSI',data=df,ax=ax[1])
ax[1].set_title('DI DIAGNOSIS HIPERTENSI')



"""DATA CLEANSING

Mengecek Missing Value
"""

df.isnull().sum())


Feature Selection

Pembagian Data Training dan Data Testing dengan Persentase 80% dan 20%
"""

#Pembagian Data Training 80% Data Testing 20%
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

"""Feature Scalling

Proses feature scaling ini bertujuan agar setiap data mempunyai range nilai yang sama terhadap data yang lain. Proses feature scaling yang digunakan adalah StandardScaler. 
"""

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

print('Nilai x_train :')
print(x_train)

print('Nilai x_test :')
print(x_test)

"""**Pelatihan Data**"""

#Training model Random forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 10, criterion='entropy', random_state=0)
forest.fit(x_train, y_train)

"""**Pengujian Model**

Confusion Matrix
"""

print('Training Accuracy :', forest.score(x_train, y_train))
print ('Testing Accuracy:', forest.score(x_test, y_test))
#Confusion Matrix
print('\nConfusion Matrix :')
y_pred = forest.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

"""Pada confusion matrix terdapat 4 nilai yaitu True Positive, False Positive, False Negative dan True Negative. True Positive merupakan data positif yang diprediksi benar. Pada Gambar diatas nilai True Positive sebesar 38. 
