import time
from datetime import timedelta
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import time
import numpy as np
from tensorflow.keras.utils import Sequence
import pandas as pd
import numpy as np
import os
import matplotlib.image as mpimg
from PIL import Image











class TripletDataGenerator(Sequence):

    def __init__(self, csv_file, output_size, shuffle=False, batch_size=10):
        #we initialize the class with some attributes
        #we can ommit the base_dir since our triplets have the real path 
        self.df = pd.read_csv(csv_file)
        #self.base_dir = base_dir
        self.output_size = output_size
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            #self.indices = np.arange(len(self.df))
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(len(self.df) / self.batch_size)
    
    #we have our custom load image so, we will avoid the 
    #use of cv2 

    def load_image(self, img_path):
        #path = "/content/drive/MyDrive/C-Minds phase 3/data/"
        #imsize = 224
        image = load_img(img_path)
        image = img_to_array(image)

        return image/ 255.0   # Normalize to [0, 1]
    
    def preprocess_image(self,image_path):
    # Load and resize the image using PIL
        image = Image.open(image_path)
        #mage = image.resize(224,224)

        # Convert the image to a NumPy array
        image_array = np.array(image)

        # Expand dimensions to match the input shape expected by most deep learning models
        image_array = np.expand_dims(image_array, axis=0)

        return image_array



    def __getitem__(self, idx):
        #getting the image, here we have to  be carefull to understand 
        #what is happenning inside the class and the function. 
        
        X_anchor = np.empty((self.batch_size, *self.output_size, 3))
        X_positive = np.empty((self.batch_size, *self.output_size, 3))
        X_negative = np.empty((self.batch_size, *self.output_size, 3))
        
        #X_anchor = []
        #X_positive = []
        #X_negative = []

        indices = self.indices[idx * self.batch_size: (idx + 1) * self.batch_size]
        
        index_drop = []

        for i, data_index in enumerate(indices):
            #print(f"Processing index {i} out of {len(indices)}")
            anchor_path = self.df.iloc[data_index,1]
            positive_path = self.df.iloc[data_index,2]
            negative_path = self.df.iloc[data_index,3]
            
            #to see how the class works lets ouput the path of the image 
            X_anchor[i] = self.preprocess_image(anchor_path)
            X_positive[i] = self.preprocess_image(positive_path)
            X_negative[i] = self.preprocess_image(negative_path)
            #index_drop.append(data_index)
            #print(f"Data index: {data_index}, DataFrame length: {len(self.df)}")
            #self.df = self.df.drop(data_index)
        
        #eliminate from the dataset the triplets that we
        #have seen previously
        #self.df = self.df.reset_index(drop=True)
            

        return [X_anchor, X_positive, X_negative],np.zeros(self.batch_size)  # No labels for triplets
