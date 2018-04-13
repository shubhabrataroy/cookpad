In this method we created a simple CNN based model described as follows:

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 32, 64, 64)    896         convolution2d_input_1[0][0]      
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 32, 64, 64)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 32, 62, 62)    9248        activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 32, 62, 62)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 32, 31, 31)    0           activation_2[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 32, 31, 31)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 64, 31, 31)    18496       dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 64, 31, 31)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 64, 29, 29)    36928       activation_3[0][0]               
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 64, 29, 29)    0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 64, 14, 14)    0           activation_4[0][0]               
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 64, 14, 14)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 128, 14, 14)   73856       dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 128, 14, 14)   0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
convolution2d_6 (Convolution2D)  (None, 128, 12, 12)   147584      activation_5[0][0]               
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 128, 12, 12)   0           convolution2d_6[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 128, 6, 6)     0           activation_6[0][0]               
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 128, 6, 6)     0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 4608)          0           dropout_3[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 1024)          4719616     flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 1024)          0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 1024)          0           activation_7[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 2)             2050        dropout_4[0][0]                  
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 2)             0           dense_2[0][0]                    
____________________________________________________________________________________________________
Total params: 5008674
____________________________________________________________________________________________________

Performance of the model is:

             precision    recall  f1-score   support
____________________________________________________________________________________________________
   Sandwich       0.62      0.89      0.74       132
____________________________________________________________________________________________________
      Sushi       0.82      0.47      0.60       134
____________________________________________________________________________________________________
avg / total       0.72      0.68      0.67       266

'Area under the curve (AUC) is:', 0.68


We see it is not a very high performing model, but it has been kept simple and lightweighted keeping processing power and time in mind. In a Nvidia Quatro 4000 system it takes around 2 minutes to train the data on the current settings.


In order to deploy this model in production I would set up this script into preprocessing, training and testing to easily encapsiulate the required components for deployment. Then I would take my preprocessing, test, as well as weights file (saved weights as json or pickled format during training) and put them in one folder. Next, I will want to host this on a server and wrap an API server around this. We can use a Flask Restful API so that we can use query parameters as my inputs and output our response in standard JSON blobs.

To host it on a server, we can deploy the Flask API on EC2.


