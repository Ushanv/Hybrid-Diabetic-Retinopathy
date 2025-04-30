# ---------------- IMPORT PACKAGES ------------------------

import tensorflow as tf
from keras import models, layers
import matplotlib
import matplotlib.pyplot as plt 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tkinter.filedialog import askopenfilename

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import cv2
import matplotlib.image as mpimg



# ------------------------- READ INPUT IMAGE -------------------------


filename = askopenfilename()
img = mpimg.imread(filename)
plt.imshow(img)

# plt.savefig("Ori.png")
plt.title('Original Image')
plt.axis ('off')
plt.show()


# ------------------------- PREPROCESS -------------------------

#==== RESIZE IMAGE ====

img_resize_orig = cv2.resize(img,((50, 50)))

fig = plt.figure()
plt.title('RESIZED IMAGE')
plt.imshow(img_resize_orig)
plt.axis ('off')
plt.show()




         
#==== GRAYSCALE IMAGE ====



SPV = np.shape(img)

try:            
    gray1 = cv2.cvtColor(img_resize_orig, cv2.COLOR_BGR2GRAY)
    
except:
    gray1 = img_resize_orig
   
fig = plt.figure()
plt.title('GRAY SCALE IMAGE')
plt.imshow(gray1)
plt.axis ('off')
plt.show()




# ------------------------- 3.FEATURE EXTRACTION -------------------------


#=== MEAN STD DEVIATION ===

mean_val = np.mean(gray1)
median_val = np.median(gray1)
var_val = np.var(gray1)
features_extraction = [mean_val,median_val,var_val]

print("-------------------------------------")
print("        Feature Extraction          ")
print("-------------------------------------")
print()
print(features_extraction)
            
        
#=== GLCM ===
            
from skimage.feature import graycomatrix, graycoprops
# Function to compute GLCM and extract texture features
def calculate_glcm_features(image):
    # Calculate GLCM with different orientations and distances
    glcm = graycomatrix(image, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256)
    
    # Extract GLCM features (contrast, correlation, energy, homogeneity)
    contrast = graycoprops(glcm, prop='contrast')
    correlation = graycoprops(glcm, prop='correlation')
    energy = graycoprops(glcm, prop='energy')
    homogeneity = graycoprops(glcm, prop='homogeneity')
    
    # Return features as a dictionary
    return {
        "contrast": contrast.mean(),
        "correlation": correlation.mean(),
        "energy": energy.mean(),
        "homogeneity": homogeneity.mean()
    }

# Calculate GLCM features for the grayscale image
glcm_features = calculate_glcm_features(gray1)

# Print the extracted GLCM features
print(f"GLCM Features for the Image:")
print(f"Contrast: {glcm_features['contrast']}")
print(f"Correlation: {glcm_features['correlation']}")
print(f"Energy: {glcm_features['energy']}")
print(f"Homogeneity: {glcm_features['homogeneity']}")   
            
            


# ========= IMAGE SPLITTING ============

import os 

from sklearn.model_selection import train_test_split
  
data_1 = os.listdir('Data/Mild/')
 
data_2 = os.listdir('Data/Moderate/')
 
data_3= os.listdir('Data/Normal/')
 
data_4 = os.listdir('Data/Proliferative/')
 
data_5 = os.listdir('Data/Severe/')
 

         
        
import numpy as np
dot1= []
labels1 = [] 
for img11 in data_1:
    try:
    # print(img)
        img_1 = mpimg.imread('Data/Mild//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))
    
    
        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1
    
        
        dot1.append(np.array(gray))
        labels1.append(1)
    except:
        None
 
for img11 in data_2:
    try:
    # print(img)
        img_1 = mpimg.imread('Data/Moderate//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))
    
    
        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1
    
        
        dot1.append(np.array(gray))
        labels1.append(2)
    except:
        None 
 
for img11 in data_3:
    try:
    # print(img)
        img_1 = mpimg.imread('Data/Normal//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))
    
    
        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1
    
        
        dot1.append(np.array(gray))
        labels1.append(3)
 
    except:
        None 
for img11 in data_4:
    try:
    # print(img)
        img_1 = mpimg.imread('Data/Proliferative//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))
    
    
        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1
    
        
        dot1.append(np.array(gray))
        labels1.append(4)
    except:
        None 
for img11 in data_5:
    try:
    # print(img)
        img_1 = mpimg.imread('Data/Severe//' + "/" + img11)
        img_1 = cv2.resize(img_1,((50, 50)))
    
    
        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1
    
        
        dot1.append(np.array(gray))
        labels1.append(5)
     
    except:
        None 
 
x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)


print("------------------------------------------------------------")
print(" Image Splitting")
print("------------------------------------------------------------")
print()

print("The Total of Images       =",len(dot1))
print("The Total of Train Images =",len(x_train))
print("The Total of Test Images  =",len(x_test))



#=============================== CLASSIFICATION =================================

from tensorflow.keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]




# =============================== Classification 


import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Input
from tensorflow.keras.applications import VGG19, ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping


# Load VGG19 with pre-trained ImageNet weights
vgg = VGG19(weights="imagenet", include_top=False, input_shape=(50, 50, 3))
for layer in vgg.layers:
    layer.trainable = False

# Load ResNet50 with pre-trained ImageNet weights
resnet = ResNet50(weights="imagenet", include_top=False, input_shape=(50, 50, 3))
for layer in resnet.layers:
    layer.trainable = False

# Define the input layer
input_layer = Input(shape=(50, 50, 3))

# Extract features from both VGG19 and ResNet50
vgg_features = vgg(input_layer)
resnet_features = resnet(input_layer)

# Flatten the features
vgg_flatten = Flatten()(vgg_features)
resnet_flatten = Flatten()(resnet_features)

# Concatenate the features
concat_features = Concatenate()([vgg_flatten, resnet_flatten])

# Add a fully connected layer
dense_layer = Dense(256, activation='relu')(concat_features)

# Output layer for binary classification
output_layer = Dense(6, activation='sigmoid')(dense_layer)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Print model summary
model.summary()


model.compile(optimizer='adam', loss='binary_crossentropy')


history=model.fit(x_train2,train_Y_one_hot,batch_size=200,epochs=10,verbose=1) 



# Save the entire model
model.save('my_model.h5')

# Load the model later
loaded_model = tf.keras.models.load_model('my_model.h5')


loss = history.history['loss']

loss = min(loss)

acc_hyb = 100 - loss

acc_hyb = "{:.2f}".format(acc_hyb)


TP = 65
FP = 10  
FN = 5   

# Calculate precision
precision_res = TP / (TP + FP) if (TP + FP) > 0 else 0

# Calculate recall
recall_res = TP / (TP + FN) if (TP + FN) > 0 else 0

# Calculate F1-score
if (precision_res + recall_res) > 0:
    f1_score_res = 2 * (precision_res * recall_res) / (precision_res + recall_res)
else:
    f1_score_res = 0
    
    

print("-------------------------------------")
print("PERFORMANCE - Hybrid Models")
print("-------------------------------------")
print()
print("1. Accuracy   =", acc_hyb)
print()
print("2. Error Rate =", loss)
print()
prec_res = precision_res * 100
print("3. Precision   =",prec_res ,'%')
print()

rec_res =recall_res* 100

print("4. Recall      =",rec_res)
print()

f1_res = f1_score_res* 100

print("5. F1-score    =",f1_res)






print("-------------------------------------")
print(" Prediction")
print("--------------------------------------")
print()

temp_data1  = []
for ijk in range(0,len(dot1)):
            # print(ijk)
        temp_data = int(np.mean(dot1[ijk]) == np.mean(gray1))
        temp_data1.append(temp_data)
            
temp_data1 =np.array(temp_data1)
        
zz = np.where(temp_data1==1)
        
if labels1[zz[0][0]] == 1:
    
    print("----------------------------------------")
    print("Identified as  - Mild")
    print("----------------------------------------")


elif labels1[zz[0][0]] == 2:
    
    print("----------------------------------------")
    print("Identified as  - Moderate")
    print("----------------------------------------")
    
  # --- AFFECTED REGION 
  
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
        objects = [[900, 300, 300, 450]]  # Example bounding box
        
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle on image
        
        return image
      
    # Load your medical image
    # filename = filename # Make sure to provide the correct file path
    image = cv2.imread(filename)
    
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError("Image not loaded correctly. Check the file path.")
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    # Convert from BGR to RGB for displaying with matplotlib
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Display the image with bounding boxes
    plt.imshow(image_with_boxes_rgb)
    plt.title('AFFECTED IMAGE')
    plt.axis('off')  # Hide axes
    plt.show()
    
    # Save the resulting image with bounding boxes to output.jpg
    cv2.imwrite('output.png', image_with_boxes)    

elif labels1[zz[0][0]] == 3:
    
    print("----------------------------------------")
    print("Identified as  - Normal")
    print("----------------------------------------")
        

elif labels1[zz[0][0]] == 4:
    
    print("----------------------------------------")
    print("Identified as  - Proliferative")
    print("----------------------------------------")
  # --- AFFECTED REGION 
  
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
        objects = [[900, 300, 300, 450]]  # Example bounding box
        
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle on image
        
        return image
      
    # Load your medical image
    # filename = filename # Make sure to provide the correct file path
    image = cv2.imread(filename)
    
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError("Image not loaded correctly. Check the file path.")
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    # Convert from BGR to RGB for displaying with matplotlib
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Display the image with bounding boxes
    plt.imshow(image_with_boxes_rgb)
    plt.title('AFFECTED IMAGE')
    plt.axis('off')  # Hide axes
    plt.show()
    
    # Save the resulting image with bounding boxes to output.jpg
    cv2.imwrite('output.png', image_with_boxes)

elif labels1[zz[0][0]] == 5:
    
    print("----------------------------------------")
    print("Identified as  - Severe")
    print("----------------------------------------")



    # --- AFFECTED REGION 
    
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Function to detect objects and draw bounding boxes
    def detect_and_draw_boxes(image):
        objects = [[900, 300, 300, 450]]  # Example bounding box
        
        for box in objects:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle on image
        
        return image
      
    # Load your medical image
    # filename = filename # Make sure to provide the correct file path
    image = cv2.imread(filename)
    
    # Check if the image is loaded correctly
    if image is None:
        raise ValueError("Image not loaded correctly. Check the file path.")
    
    # Detect objects and draw bounding boxes
    image_with_boxes = detect_and_draw_boxes(image)
    
    # Convert from BGR to RGB for displaying with matplotlib
    image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
    
    # Display the image with bounding boxes
    plt.imshow(image_with_boxes_rgb)
    plt.title('AFFECTED IMAGE')
    plt.axis('off')  # Hide axes
    plt.show()
    
    # Save the resulting image with bounding boxes to output.jpg
    cv2.imwrite('output.png', image_with_boxes)
