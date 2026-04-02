                                          Comment Toxicity Detection

Overview:

          Domain : Comment Detection Analytics / Deep Learning

          This project is a Deep learning project. In this project we are going to analyze
comments that are toxic, severe toxic, obscene, threat, insult and identity_hate. Develop
deep learning model-based comment toxicity model using Python. This model will
analyze text input from online comments and predict the likelihood of each comment
being toxic.

Project Deliverables:

         EDA
         Data Preprocessing
         Embedding matrix calculation(glove)
         Model Building (LSTM(Bidirectional) and RNNs)
         Model evaluation
         Project Insights

        
EDA(Exploratory data analysis) :

         Loading the data
         Understanding the data
         Dropping the id column
         Handling null values
         Handling the duplicates data
         Heatmap for correlation between features
        
Data Preprocessing:

       Converting comments lowercase
       Removing special characters
       Splitting comments into words
       Removing stop words
       Joining the comments
       Tokenize the comments
       Convert text into uniform features
       Padding the sequences

Prepare embedding matrix using GloVe embeddings ��:

       Glove is pretrained an unsupervised learning algorithm
       For obtaining vector representations for words
       Download pre-trained vectors from online
       load them into a dictionary mapping word to vectors
       Splitting the data into train, test and validation

Model Building (LSTM(Bidirectional) and RNNs) ��:

LSTM - long, short-term memory:

       Initialize the model
       Convert input integer sequence into dence vector of a fixed size
       (embedding dim)
       The layer uses pretrained embedding matrix
       A recurrent layer that processes the input sequence in both forward and
      backward
       This helps captures context from both sides of word within sequence
       Here we used bidirectional LSTM 32 units with tanh activation function
       The tanh (hyperbolic tangent) function is a non-linear activation function in
      neural networks that maps input values to a range between -1 and 1. It is
      zero-centered, which helps improve training efficiency

### RNNs - Recurrent neural network model building code:
     from tensorflow.keras.layers import SimpleRNN
     from tensorflow.keras.regularizers import l2
     # Building the model
     model2 = Sequential()
     model2.add(Embedding(vocab_size + 1, embedding_dim, input_length = max_len))
     model2.add(Bidirectional(SimpleRNN(100, activation='tanh')))
     model2.add(Dense(128, activation = 'relu'))
     #model2.add(Dropout(0.3))
     model2.add(Dense(256, activation = 'relu', kernel_regularizer=l2(0.001)))
     model2.add(Dense(64, activation = 'relu', kernel_regularizer=l2(0.001)))
     model2.add(Dense(6, activation = 'sigmoid'))

Model compilation ��:

LSTM and RNNs:

       Shuffle : True
       Batch_size : 64 for LSTM
       Batch_size : 64 for RNNs
       Optimizer :Adam
       Loss : Binary_cross_entropy
       Metrics : accuracy
      Adam-optimizer:
       Adam calculates the exponential moving average of gradients
       It is computationally efficient, requires little memory, is well-suited for high-
      dimensional/large-scale data
       Widely used in neural networks, particularly for natural language processing,
      computer vision, and generative models.

Training the model :

      Here we are going to fit the model using input data ,batch size, validation data and two
      epochs
      Visualization of the data:
      Plotting the loss and accuracy LSTM:
      Val loss : 0.0472
      Train loss : 0.0514
      Val accuracy : 0.9840
      Train accuracy : 0.9756
      Plotting the loss and accuracy RNNs :
      Val loss : 0.1379
      Train loss : 0.1390
      Val accuracy : 0.9940
      Train accuracy : 0.9956

Prediction:
       After training the model we saved the model using pickle file format.
      
       Load the model for prediction.
       Get the comment from user and detect that comments are toxic or non-toxic.
      Building the Streamlit Application :
      Text area : user input
      Button : predict button
      Showing the result message
      
Sidebar buttons :

       Data visualization
       Deployment
       Data

Word cloud:

      Generating the word cloud for toxic, severe toxic....etc
      Insights :
      Label distribution:
      Toxic, obscene, insult - high-count
      Severe toxic, identity_hate, threat -low count
      Normally target labels are balanced but in this case not balanced

precision recall f1-score support

      0 0.86 0.67 0.75 1554
      1 0.61 0.13 0.21 151
      2 0.81 0.78 0.80 842
      3 0.00 0.00 0.00 47
      4 0.69 0.71 0.70 802
      5 0.76 0.18 0.29 164

micro avg 0.80 0.65 0.72 3560
macro avg 0.62 0.41 0.46 3560
weighted avg 0.78 0.65 0.70 3560
samples avg 0.06 0.06 0.06 3560

Here we can target labels are not balanced, that’s why metrics does not perform well to
all labels

Toxic - precision performed well

The plot says the purple, green, yellow lines those lines model able to classified well
And light pink, light blue ,blue weakly classified by a model , this model is not suited for
this label because these labels are unbalanced 

Work flow of the streamlit app:

1> Get the comment from user 

2> predict the comment by the model, the given comment toxic or not 

3> Here I have done bult input prediction by model as well 
