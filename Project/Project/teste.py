import numpy as np 
import pandas as pd 
import os
import cv2
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from skimage.feature import hog
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from skimage.color import rgb2gray



#----

#path server
path = "UTKFace/"

##path martim
#path = "../../../../DadosProj/UTKFace/"
##path alex
#path = "../../../data_project/UTKFace/"
##path server
path = "../../UTKFace/"
files = os.listdir(path)
size = len(files)
print("Total samples:",size)
print(files[0])

#----

images = []
ages = []
genders = []
ethnicities = []
counter = 0

for file in files:
    if counter == 50:
        break
    
    image = cv2.imread(path+file)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, dsize=(200, 200))
    image = image / 255.0
    
    images.append(image)
    delimit = file.split('_')
    ages.append(int(delimit[0]))
    # first number is age (0-116 possible values)
    genders.append(int(delimit[1]))
    # second number is gender (0 for male, 1 for female)
    ethnicities.append(int(delimit[2]))
    # third number is ethnicity 
    # (0 for white, 1 for black, 2 for asian, 3 for indian, 4 for any other ethnicity)

    counter += 1

#----

# Extract HOG features from the images
hog_features = []
for img in images:
    #hog_feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
    #                cells_per_block=(2, 2), transform_sqrt=True, feature_vector=True, multichannel=True, channel_axis=2)
    hog_feature = hog(img, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), transform_sqrt=True, feature_vector=True, channel_axis=2)
    
    hog_features.append(hog_feature)
hog_features = np.array(hog_features)

# Normalize the features to have zero mean and unit variance
mean = np.mean(hog_features, axis=0)
std = np.std(hog_features, axis=0)
hog_features_norm = (hog_features - mean) / std

#----

X_train, X_test, y_train, y_test = train_test_split(hog_features_norm, genders, test_size=0.2, random_state=42)

#----

clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate the classifier on the testing data
accuracy = clf.score(X_test, y_test)
print("Accuracy:", accuracy)

#----

#lista topicos para prof:
#onde colocar o nosso dataset para usar no servidor
#como ter um registo acessivel por nos dos outputs
#opiniao do prof sobre nossos metodos de extracao de features
#falar sobre orb e problemas que existem a tentar usar como metodo de extracao de feats

#----
#respostas:
#no servidor
#descer no less teste.py com space e subir para topo com b
#q para sair

#o que temos agora não é transfer learning

#testar autoencoder para feat. extract. (secalhar preferivel a orb)
#----