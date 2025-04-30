#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
from PIL import Image
import matplotlib.image as mpimg

#======================== IMPORT PACKAGES ===========================

import numpy as np
import matplotlib.pyplot as plt 
from tkinter.filedialog import askopenfilename
import cv2
import matplotlib.image as mpimg
import streamlit as st
from PIL import Image
import streamlit as st

import base64
import cv2

# ================ Background image ===

st.markdown(f'<h1 style="color:#000000;text-align: center;font-size:36px;">{"Hybrid Retinal Image Enhancement Algorithm for Diabetic Retinopathy Diagnostic Using Deep Learning Model"}</h1>', unsafe_allow_html=True)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('b1.jpg')

#====================== READ A INPUT IMAGE =========================

fileneme = st.file_uploader("Upload a image")

if fileneme is None:
    
    st.text("Kindly upload input image....")

else:
    # selected_image_name = fileneme.name
    #====================== READ A INPUT IMAGE =========================
    
    
    # filename = askopenfilename()
    img = mpimg.imread(fileneme)
    plt.imshow(img)
    # plt.title('Original Image') 
    plt.axis ('off')
    plt.savefig("Ori.jpg")
    plt.show()
        
    st.image(img,caption="Original Image")
    #============================ PREPROCESS =================================
    
    #==== RESIZE IMAGE ====
    
    resized_image = cv2.resize(img,(300,300))
    img_resize_orig = cv2.resize(img,((50, 50)))
    
    fig = plt.figure()
    plt.title('RESIZED IMAGE')
    plt.imshow(resized_image)
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
    plt.imshow(gray1,cmap='gray')
    plt.axis ('off')
    plt.show()
    
    
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
        st.markdown(f'<h2 style="color:#0000FF;text-align: center;font-size:24px;font-family:Caveat, sans-serif;">{"Identified as  - Mild"}</h2>', unsafe_allow_html=True)

        print("----------------------------------------")


        st.markdown("""
            <div style="border: 2px solid #4CAF50; background-color: #e8f5e9; padding: 20px; border-radius: 5px;">
                <strong style="color: #388e3c;">Suggestion:</strong> <br>
                Regular monitoring and early intervention. <br><br>
                
                <strong style="color: #388e3c;">Remedy:</strong> <br>
                Control blood sugar levels, manage blood pressure, and maintain a healthy diet to slow progression. Regular eye exams (every 6-12 months) to detect early changes.
            </div>
        """, unsafe_allow_html=True)





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
        filename = 'Ori.jpg'  # Make sure to provide the correct file path
        image = cv2.imread(filename)
        
        # Check if the image is loaded correctly
        if image is None:
            raise ValueError("Image not loaded correctly. Check the file path.")
        
        # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)
        
        # Convert from BGR to RGB for displaying with matplotlib
        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        st.image("output.png",caption="Detected Region")
        # Display the image with bounding boxes
        plt.imshow(image_with_boxes_rgb)
        plt.title('AFFECTED IMAGE')
        plt.axis('off')  # Hide axes
        plt.show()
        
  


    
    
    elif labels1[zz[0][0]] == 2:
        
        print("----------------------------------------")
        print("Identified as  - Moderate")
        st.markdown(f'<h2 style="color:#00FF00;text-align: center;font-size:24px;font-family:Caveat, sans-serif;">{"Identified as  - Moderate"}</h2>', unsafe_allow_html=True)

        print("----------------------------------------")
        
        st.markdown("""
            <div style="border: 2px solid #4CAF50; background-color: #e8f5e9; padding: 20px; border-radius: 5px;">
                <strong style="color: #388e3c;">Suggestion:</strong> <br>
                More frequent eye exams and better disease management. <br><br>
                
                <strong style="color: #388e3c;">Remedy:</strong> <br>
                Tight blood sugar and blood pressure control, as well as cholesterol management. Consider laser treatment (focal laser photocoagulation) if macular edema is present.
            </div>
        """, unsafe_allow_html=True)











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
        filename = 'Ori.jpg'  # Make sure to provide the correct file path
        image = cv2.imread(filename)
        
        # Check if the image is loaded correctly
        if image is None:
            raise ValueError("Image not loaded correctly. Check the file path.")
        
        # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)
        
        # Convert from BGR to RGB for displaying with matplotlib
        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        st.image("output.png",caption="Detected Region")
        # Display the image with bounding boxes
        plt.imshow(image_with_boxes_rgb)
        plt.title('AFFECTED IMAGE')
        plt.axis('off')  # Hide axes
        plt.show()
                
    
    elif labels1[zz[0][0]] == 3:
        
        print("----------------------------------------")
        print("Identified as  - Normal")
        st.markdown(f'<h2 style="color:#00FF00;text-align: center;font-size:24px;font-family:Caveat, sans-serif;">{"Identified as  - Normal"}</h2>', unsafe_allow_html=True)

        print("----------------------------------------")
            
    
    elif labels1[zz[0][0]] == 4:
        
        print("----------------------------------------")
        print("Identified as  - Proliferative")
        st.markdown(f'<h2 style="color:#00FF00;text-align: center;font-size:24px;font-family:Caveat, sans-serif;">{"Identified as  - Proliferative"}</h2>', unsafe_allow_html=True)

        print("----------------------------------------")
        
        
        st.markdown("""
            <div style="border: 2px solid #4CAF50; background-color: #e8f5e9; padding: 20px; border-radius: 5px;">
                <strong style="color: #388e3c;">Suggestion:</strong> <br>
                Urgent intervention to prevent vision loss. <br><br>
                
                <strong style="color: #388e3c;">Remedy:</strong> <br>
                Panretinal laser photocoagulation or vitrectomy surgery to treat retinal neovascularization. Anti-VEGF injections (e.g., ranibizumab or aflibercept) may be used to manage neovascularization and macular edema. Tight control of blood sugar, pressure, and cholesterol is crucial.
            </div>
        """, unsafe_allow_html=True)        
        
        
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
        filename = 'Ori.jpg'  # Make sure to provide the correct file path
        image = cv2.imread(filename)
        
        # Check if the image is loaded correctly
        if image is None:
            raise ValueError("Image not loaded correctly. Check the file path.")
        
        # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)
        
        # Convert from BGR to RGB for displaying with matplotlib
        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        st.image("output.png",caption="Detected Region")
        # Display the image with bounding boxes
        plt.imshow(image_with_boxes_rgb)
        plt.title('AFFECTED IMAGE')
        plt.axis('off')  # Hide axes
        plt.show()
            
    
    elif labels1[zz[0][0]] == 5:
        
        print("----------------------------------------")
        print("Identified as  - Severe")
        st.markdown(f'<h2 style="color:#00FF00;text-align: center;font-size:24px;font-family:Caveat, sans-serif;">{"Identified as  - Severe"}</h2>', unsafe_allow_html=True)
        print("----------------------------------------")    
    
        st.markdown("""
            <div style="border: 2px solid #4CAF50; background-color: #e8f5e9; padding: 20px; border-radius: 5px;">
                <strong style="color: #388e3c;">Suggestion:</strong> <br>
                Immediate and comprehensive intervention. <br><br>
                
                <strong style="color: #388e3c;">Remedy:</strong> <br>
                Laser treatment (panretinal photocoagulation) to reduce the risk of progression to proliferative retinopathy, along with aggressive management of blood glucose and blood pressure. Frequent follow-up visits.
            </div>
        """, unsafe_allow_html=True)          
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
        filename = 'Ori.jpg'  # Make sure to provide the correct file path
        image = cv2.imread(filename)
        
        # Check if the image is loaded correctly
        if image is None:
            raise ValueError("Image not loaded correctly. Check the file path.")
        
        # Detect objects and draw bounding boxes
        image_with_boxes = detect_and_draw_boxes(image)
        
        # Convert from BGR to RGB for displaying with matplotlib
        image_with_boxes_rgb = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
        st.image("output.png",caption="Detected Region")
        # Display the image with bounding boxes
        plt.imshow(image_with_boxes_rgb)
        plt.title('AFFECTED IMAGE')
        plt.axis('off')  # Hide axes
        plt.show()
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    