# ==================================
# Image Classification of parking lots - Scikitlearn
# Bu model, dolu/boş park alanlarını sınıflandıran basit bir image classification modeli üretir
# ==================================

import os
import numpy as np
import pickle #modeli kaydetmek için

from skimage.io import imread # görüntü dosyası -> NumPy array
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #en düşük hata oranına sahip hiperparametrelerin seçimi. CV: Cross-Validation
from sklearn.svm import SVC # Support Vector Classification
from sklearn.metrics import accuracy_score

# PREPARE DATA,
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
input_dir = './clf-data'
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)): #resim ismi
        img_path = os.path.join(input_dir, category, file) #her bir resime dokunacak
        img = imread(img_path)
        img = resize(img, (15, 15))

        data.append(img.flatten()) #2D veye 3D görüntüleri tek bir vektör(satır) haline getirir
                                   #img.reshape(-1) aynı sonucu verir
        labels.append(category_idx)

data = np.asarray(data) #train için uygun olan formata getirir
labels = np.asarray(labels)


# TRAIN/TEST/SPLIT
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
#stratify: veri setinde dengesizlik varsa her iki setin de aynı oranda 0 1 içermesini sağlar. etiketlerin orantılı dağıtılmasını sağlar


# TRAIN CLASSIFIER
classifier = SVC()

svm_parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}] # 3 x 4 = 12 classifiers. burada detaya inmedim.

grid_search = GridSearchCV(classifier, svm_parameters)
grid_search.fit(x_train, y_train)

# TEST PERFORMANCE
best_estimator = grid_search.best_estimator_ #12 arasından en iyi classifier seçiyoruz

y_prediction = best_estimator.predict(x_test)

score = accuracy_score(y_prediction, y_test)

print('{}% of samples were correctly classified'.format(str(score * 100)))

pickle.dump(best_estimator, open('./model.p', 'wb')) #modeli kaydet
