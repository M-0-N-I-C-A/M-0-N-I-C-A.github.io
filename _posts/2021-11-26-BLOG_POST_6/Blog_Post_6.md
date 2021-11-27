---
layout: post
title: It's Fake News!
---



# §1. Acquire Training Data


```python
import numpy as np
import pandas as pd
import tensorflow as tf
import re
import string
from matplotlib import pyplot as plt
```


```python
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow import keras
from sklearn.decomposition import PCA  
```


```python
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers.experimental.preprocessing import StringLookup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import plotly.express as px 
import plotly.io as pio
from plotly.io import write_html
pio.templates.default = "plotly_white"
```


```python
train_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_train.csv?raw=true"
train_data = pd.read_csv(train_url)
```


```python
train_data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>title</th>
      <th>text</th>
      <th>fake</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>17366</td>
      <td>Merkel: Strong result for Austria's FPO 'big c...</td>
      <td>German Chancellor Angela Merkel said on Monday...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5634</td>
      <td>Trump says Pence will lead voter fraud panel</td>
      <td>WEST PALM BEACH, Fla.President Donald Trump sa...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>17487</td>
      <td>JUST IN: SUSPECTED LEAKER and “Close Confidant...</td>
      <td>On December 5, 2017, Circa s Sara Carter warne...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12217</td>
      <td>Thyssenkrupp has offered help to Argentina ove...</td>
      <td>Germany s Thyssenkrupp, has offered assistance...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5535</td>
      <td>Trump say appeals court decision on travel ban...</td>
      <td>President Donald Trump on Thursday called the ...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# §2. Make a Dataset

Write a function called make_dataset. This function should do two things:
Remove stopwords from the article text and title. 

1.   Remove stopwords from the article text and title. 
2.   Construct and return a tf.data.Dataset with two inputs and one output. The input should be of the form (title, text), and the output should consist only of the fake column. 


```python
import nltk
nltk.download('stopwords')
```

    [nltk_data] Downloading package stopwords to /root/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!





    True




```python
from nltk.corpus import stopwords
stop = stopwords.words('english')
```


```python
def make_dataset(train_data):
  # remove stopwords
  train_data['title_no_sw'] = train_data['title'] .apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
  train_data['text_no_sw'] = train_data['text'] .apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

  data = tf.data.Dataset.from_tensor_slices(
    (
        {
            "title" : train_data[["title_no_sw"]], 
            "text" : train_data[["text_no_sw"]],
        }, 
        {
            "fake" : train_data[["fake"]]
        }
    )
  )

  data = data.batch(100)
  return data
```


```python
# apply make_dataset function on the dataset
Dataset = make_dataset(train_data)
```

Validation Data - Split 20% of the primary dataset to use for validation




```python
train_size = int(0.8*len(Dataset))
train = Dataset.take(train_size)   
val   = Dataset.skip(train_size)   
len(train), len(val)
```




    (180, 45)



Base Rate (the accuracy of a model that always makes the same guess)



```python
labels_iterator= train.unbatch().map(lambda image,fake: fake).as_numpy_iterator()
True_Article = 0
Fake_Article = 0
for LABEL in labels_iterator:
    if LABEL["fake"] == 1:
      Fake_Article += 1
    else:
      True_Article += 1
```


```python
base_rate = Fake_Article / (Fake_Article + True_Article)
print(base_rate)
```

    0.5220555555555556


The base rate for this data set is approximately 0.522, which means if a model predicts "fake" for a random piece of news, the prediction accuracy is around 52.2%.

# §3. Create Models

Use TensorFlow models to offer a perspective on the following question:

When detecting fake news, is it most effective to focus on only the title of the article, the full text of the article, or both?

To address this question, create three (3) TensorFlow models.



1. In the first model, you should use only the article title as an input.
2. In the second model, you should use only the article text as an input.
3. In the third model, you should use both the article title and the article text as input.




### Model 1: Article title as input


```python
size_vocabulary = 2000

# function for text standardization
def standardization(input_data):    
    # change all letters to lowercase
    lowercase = tf.strings.lower(input_data)
    # remove punctuations
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

# function for text vectorization
vectorize_layer = TextVectorization(
    standardize = standardization, 
    max_tokens = size_vocabulary, 
    output_mode='int',
    output_sequence_length=500) 

# vectorization layer
vectorize_layer.adapt(train.map(lambda x, y: x["title"]))
```


```python
title_input = keras.Input(
    shape = (1,),
    name = "title",
    dtype = "string"
)
```


```python
title_features = vectorize_layer(title_input)
title_features = layers.Embedding(size_vocabulary, 3, name = "embedding")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(12, activation='relu')(title_features)
```


```python
output = layers.Dense(2, name = "fake")(title_features)
```


```python
# use article title as input for model 1
model1 = keras.Model(
    inputs = title_input,
    outputs = output)

```


```python
model1.summary()
```

    Model: "model"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     title (InputLayer)          [(None, 1)]               0         
                                                                     
     text_vectorization (TextVec  (None, 500)              0         
     torization)                                                     
                                                                     
     embedding (Embedding)       (None, 500, 3)            6000      
                                                                     
     dropout (Dropout)           (None, 500, 3)            0         
                                                                     
     global_average_pooling1d (G  (None, 3)                0         
     lobalAveragePooling1D)                                          
                                                                     
     dropout_1 (Dropout)         (None, 3)                 0         
                                                                     
     dense (Dense)               (None, 12)                48        
                                                                     
     fake (Dense)                (None, 2)                 26        
                                                                     
    =================================================================
    Total params: 6,074
    Trainable params: 6,074
    Non-trainable params: 0
    _________________________________________________________________



```python
# Plot model 1
keras.utils.plot_model(model1)  
```




    
![output_28_0.png](/_posts/output_28_0.png)
    




```python
model1.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
# Train model1
history = model1.fit(train,
                    validation_data=val,
                    epochs = 30)
```

    Epoch 1/30


    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning:
    
    Input dict contained keys ['text'] which did not match any model input. They will be ignored by the model.
    


    180/180 [==============================] - 2s 10ms/step - loss: 0.6923 - accuracy: 0.5158 - val_loss: 0.6898 - val_accuracy: 0.5266
    Epoch 2/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.6858 - accuracy: 0.5396 - val_loss: 0.6763 - val_accuracy: 0.5266
    Epoch 3/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.6564 - accuracy: 0.7041 - val_loss: 0.6259 - val_accuracy: 0.7420
    Epoch 4/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.5782 - accuracy: 0.8656 - val_loss: 0.5154 - val_accuracy: 0.9364
    Epoch 5/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.4528 - accuracy: 0.9201 - val_loss: 0.3808 - val_accuracy: 0.9443
    Epoch 6/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.3368 - accuracy: 0.9338 - val_loss: 0.2802 - val_accuracy: 0.9519
    Epoch 7/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.2607 - accuracy: 0.9413 - val_loss: 0.2166 - val_accuracy: 0.9580
    Epoch 8/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.2142 - accuracy: 0.9482 - val_loss: 0.1752 - val_accuracy: 0.9631
    Epoch 9/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.1814 - accuracy: 0.9539 - val_loss: 0.1470 - val_accuracy: 0.9674
    Epoch 10/30
    180/180 [==============================] - 2s 8ms/step - loss: 0.1581 - accuracy: 0.9580 - val_loss: 0.1276 - val_accuracy: 0.9724
    Epoch 11/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.1400 - accuracy: 0.9606 - val_loss: 0.1132 - val_accuracy: 0.9733
    Epoch 12/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.1266 - accuracy: 0.9648 - val_loss: 0.1029 - val_accuracy: 0.9753
    Epoch 13/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.1157 - accuracy: 0.9661 - val_loss: 0.0946 - val_accuracy: 0.9768
    Epoch 14/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.1069 - accuracy: 0.9692 - val_loss: 0.0857 - val_accuracy: 0.9771
    Epoch 15/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.1004 - accuracy: 0.9701 - val_loss: 0.0786 - val_accuracy: 0.9786
    Epoch 16/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0955 - accuracy: 0.9706 - val_loss: 0.0746 - val_accuracy: 0.9793
    Epoch 17/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0891 - accuracy: 0.9718 - val_loss: 0.0717 - val_accuracy: 0.9809
    Epoch 18/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0854 - accuracy: 0.9733 - val_loss: 0.0684 - val_accuracy: 0.9809
    Epoch 19/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0805 - accuracy: 0.9737 - val_loss: 0.0660 - val_accuracy: 0.9800
    Epoch 20/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0810 - accuracy: 0.9728 - val_loss: 0.0615 - val_accuracy: 0.9804
    Epoch 21/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0748 - accuracy: 0.9748 - val_loss: 0.0591 - val_accuracy: 0.9802
    Epoch 22/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0734 - accuracy: 0.9751 - val_loss: 0.0576 - val_accuracy: 0.9807
    Epoch 23/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.0695 - accuracy: 0.9759 - val_loss: 0.0617 - val_accuracy: 0.9804
    Epoch 24/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0679 - accuracy: 0.9763 - val_loss: 0.0550 - val_accuracy: 0.9809
    Epoch 25/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0659 - accuracy: 0.9770 - val_loss: 0.0548 - val_accuracy: 0.9816
    Epoch 26/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.0648 - accuracy: 0.9768 - val_loss: 0.0529 - val_accuracy: 0.9811
    Epoch 27/30
    180/180 [==============================] - 1s 8ms/step - loss: 0.0638 - accuracy: 0.9768 - val_loss: 0.0525 - val_accuracy: 0.9816
    Epoch 28/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.0612 - accuracy: 0.9787 - val_loss: 0.0514 - val_accuracy: 0.9816
    Epoch 29/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.0611 - accuracy: 0.9773 - val_loss: 0.0506 - val_accuracy: 0.9816
    Epoch 30/30
    180/180 [==============================] - 2s 9ms/step - loss: 0.0587 - accuracy: 0.9794 - val_loss: 0.0492 - val_accuracy: 0.9822



```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fe77c9a3650>





![output_31_1.png](/_posts/output_31_1.png)
    


The plot above shows that Model 1 (with article title as input) is able to achieve around 98% validation accuracy with no overfitting observed as validation accuracy is almost consistently above training accuracy due to the dropout layers.

# Model 2: Article text as input


```python
size_vocabulary = 2000

# function for text standardization
def standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    no_punctuation = tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation),'')
    return no_punctuation 

# function for text vectorization
vectorize_layer = TextVectorization(
    standardize=standardization,
    max_tokens=size_vocabulary, # only consider this many words
    output_mode='int',
    output_sequence_length=500) 

# vectorization layer
vectorize_layer.adapt(train.map(lambda x, y: x["text"]))
```


```python
text_input = keras.Input(
    shape = (1,),
    name = "text",
    dtype = "string"
)
```


```python
text_features = vectorize_layer(text_input)
text_features = layers.Embedding(size_vocabulary, 3, name = "embedding")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(12, activation='relu')(text_features)
# Set up the text layers as we discussed in the lecture
```


```python
output = layers.Dense(2, name = "fake")(text_features)
```


```python
# use article text as input for model 2
model2 = keras.Model(
    inputs = text_input,
    outputs = output)
```


```python
model2.summary()
```

    Model: "model_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     text (InputLayer)           [(None, 1)]               0         
                                                                     
     text_vectorization_1 (TextV  (None, 500)              0         
     ectorization)                                                   
                                                                     
     embedding (Embedding)       (None, 500, 3)            6000      
                                                                     
     dropout_2 (Dropout)         (None, 500, 3)            0         
                                                                     
     global_average_pooling1d_1   (None, 3)                0         
     (GlobalAveragePooling1D)                                        
                                                                     
     dropout_3 (Dropout)         (None, 3)                 0         
                                                                     
     dense_1 (Dense)             (None, 12)                48        
                                                                     
     fake (Dense)                (None, 2)                 26        
                                                                     
    =================================================================
    Total params: 6,074
    Trainable params: 6,074
    Non-trainable params: 0
    _________________________________________________________________



```python
# Plot model 2
keras.utils.plot_model(model2) 
```




    
![output_40_0.png](/_posts/output_40_0.png)
    




```python
model2.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```


```python
# Train model2
history = model2.fit(train,
                    validation_data=val,
                    epochs = 30)
```

    Epoch 1/30


    /usr/local/lib/python3.7/dist-packages/keras/engine/functional.py:559: UserWarning:
    
    Input dict contained keys ['title'] which did not match any model input. They will be ignored by the model.
    


    180/180 [==============================] - 4s 16ms/step - loss: 0.6886 - accuracy: 0.5223 - val_loss: 0.6771 - val_accuracy: 0.5298
    Epoch 2/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.6137 - accuracy: 0.7518 - val_loss: 0.4966 - val_accuracy: 0.9117
    Epoch 3/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.3894 - accuracy: 0.9085 - val_loss: 0.2897 - val_accuracy: 0.9434
    Epoch 4/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.2598 - accuracy: 0.9362 - val_loss: 0.2122 - val_accuracy: 0.9559
    Epoch 5/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.2072 - accuracy: 0.9501 - val_loss: 0.1744 - val_accuracy: 0.9616
    Epoch 6/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.1738 - accuracy: 0.9556 - val_loss: 0.1510 - val_accuracy: 0.9645
    Epoch 7/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.1535 - accuracy: 0.9614 - val_loss: 0.1352 - val_accuracy: 0.9685
    Epoch 8/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.1381 - accuracy: 0.9648 - val_loss: 0.1242 - val_accuracy: 0.9697
    Epoch 9/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.1280 - accuracy: 0.9666 - val_loss: 0.1152 - val_accuracy: 0.9719
    Epoch 10/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.1159 - accuracy: 0.9683 - val_loss: 0.1082 - val_accuracy: 0.9735
    Epoch 11/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.1092 - accuracy: 0.9713 - val_loss: 0.1021 - val_accuracy: 0.9739
    Epoch 12/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.1022 - accuracy: 0.9730 - val_loss: 0.0983 - val_accuracy: 0.9755
    Epoch 13/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0987 - accuracy: 0.9738 - val_loss: 0.0941 - val_accuracy: 0.9766
    Epoch 14/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0927 - accuracy: 0.9753 - val_loss: 0.0913 - val_accuracy: 0.9771
    Epoch 15/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0874 - accuracy: 0.9746 - val_loss: 0.0897 - val_accuracy: 0.9777
    Epoch 16/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0825 - accuracy: 0.9761 - val_loss: 0.0877 - val_accuracy: 0.9775
    Epoch 17/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0788 - accuracy: 0.9774 - val_loss: 0.0869 - val_accuracy: 0.9777
    Epoch 18/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0755 - accuracy: 0.9789 - val_loss: 0.0857 - val_accuracy: 0.9789
    Epoch 19/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0713 - accuracy: 0.9808 - val_loss: 0.0836 - val_accuracy: 0.9784
    Epoch 20/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0708 - accuracy: 0.9793 - val_loss: 0.0824 - val_accuracy: 0.9791
    Epoch 21/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0655 - accuracy: 0.9811 - val_loss: 0.0803 - val_accuracy: 0.9795
    Epoch 22/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0648 - accuracy: 0.9803 - val_loss: 0.0791 - val_accuracy: 0.9804
    Epoch 23/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0614 - accuracy: 0.9834 - val_loss: 0.0792 - val_accuracy: 0.9804
    Epoch 24/30
    180/180 [==============================] - 3s 16ms/step - loss: 0.0598 - accuracy: 0.9828 - val_loss: 0.0798 - val_accuracy: 0.9807
    Epoch 25/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0557 - accuracy: 0.9837 - val_loss: 0.0798 - val_accuracy: 0.9802
    Epoch 26/30
    180/180 [==============================] - 3s 17ms/step - loss: 0.0556 - accuracy: 0.9842 - val_loss: 0.0764 - val_accuracy: 0.9820
    Epoch 27/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0534 - accuracy: 0.9847 - val_loss: 0.0767 - val_accuracy: 0.9818
    Epoch 28/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0494 - accuracy: 0.9861 - val_loss: 0.0813 - val_accuracy: 0.9802
    Epoch 29/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0489 - accuracy: 0.9853 - val_loss: 0.0783 - val_accuracy: 0.9807
    Epoch 30/30
    180/180 [==============================] - 3s 15ms/step - loss: 0.0478 - accuracy: 0.9853 - val_loss: 0.0778 - val_accuracy: 0.9816



```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fe77a76bbd0>





![output_43_1png](/_posts/output_43_1.png)
    


The plot above shows that Model 2 (with article text as input) is able to achieve around 98% validation accuracy with no overfitting observed.

# Model 3: Article title and text as input


```python
# layer from Model 1
title_features = vectorize_layer(title_input)
title_features = layers.Embedding(size_vocabulary, 3, name = "embeddingTITLE")(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.GlobalAveragePooling1D()(title_features)
title_features = layers.Dropout(0.2)(title_features)
title_features = layers.Dense(12, activation='relu')(title_features)
```


```python
# layer from Model 2
text_features = vectorize_layer(text_input)
text_features = layers.Embedding(size_vocabulary, 3, name = "embeddingTEXT")(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.GlobalAveragePooling1D()(text_features)
text_features = layers.Dropout(0.2)(text_features)
text_features = layers.Dense(12, activation='relu')(text_features)
```


```python
# concatenate layers 
main = layers.concatenate([title_features, text_features], axis = 1)
# add one output layer
main = layers.Dense(32, activation='relu')(main)
output = layers.Dense(2, name = "fake")(main)

```


```python
# use article title and text as input for model 3
model3 = keras.Model(
    inputs = [title_input, text_input],
    outputs = output)
```


```python
model3.summary()
```

    Model: "model_2"
    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to                     
    ==================================================================================================
     title (InputLayer)             [(None, 1)]          0           []                               
                                                                                                      
     text (InputLayer)              [(None, 1)]          0           []                               
                                                                                                      
     text_vectorization_1 (TextVect  (None, 500)         0           ['title[0][0]',                  
     orization)                                                       'text[0][0]']                   
                                                                                                      
     embeddingTITLE (Embedding)     (None, 500, 3)       6000        ['text_vectorization_1[1][0]']   
                                                                                                      
     embeddingTEXT (Embedding)      (None, 500, 3)       6000        ['text_vectorization_1[2][0]']   
                                                                                                      
     dropout_4 (Dropout)            (None, 500, 3)       0           ['embeddingTITLE[0][0]']         
                                                                                                      
     dropout_6 (Dropout)            (None, 500, 3)       0           ['embeddingTEXT[0][0]']          
                                                                                                      
     global_average_pooling1d_2 (Gl  (None, 3)           0           ['dropout_4[0][0]']              
     obalAveragePooling1D)                                                                            
                                                                                                      
     global_average_pooling1d_3 (Gl  (None, 3)           0           ['dropout_6[0][0]']              
     obalAveragePooling1D)                                                                            
                                                                                                      
     dropout_5 (Dropout)            (None, 3)            0           ['global_average_pooling1d_2[0][0
                                                                     ]']                              
                                                                                                      
     dropout_7 (Dropout)            (None, 3)            0           ['global_average_pooling1d_3[0][0
                                                                     ]']                              
                                                                                                      
     dense_2 (Dense)                (None, 12)           48          ['dropout_5[0][0]']              
                                                                                                      
     dense_3 (Dense)                (None, 12)           48          ['dropout_7[0][0]']              
                                                                                                      
     concatenate (Concatenate)      (None, 24)           0           ['dense_2[0][0]',                
                                                                      'dense_3[0][0]']                
                                                                                                      
     dense_4 (Dense)                (None, 32)           800         ['concatenate[0][0]']            
                                                                                                      
     fake (Dense)                   (None, 2)            66          ['dense_4[0][0]']                
                                                                                                      
    ==================================================================================================
    Total params: 12,962
    Trainable params: 12,962
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
# plot Model 3
keras.utils.plot_model(model3)
```




    
![output_51_0png](/_posts/output_51_0.png)
    




```python
model3.compile(optimizer = "adam",
              loss = losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

```


```python
# Train Model 3
history = model3.fit(train,
                    validation_data=val,
                    epochs = 30)

```

    Epoch 1/30
    180/180 [==============================] - 5s 21ms/step - loss: 0.6621 - accuracy: 0.6224 - val_loss: 0.5347 - val_accuracy: 0.9213
    Epoch 2/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.3148 - accuracy: 0.9144 - val_loss: 0.1746 - val_accuracy: 0.9593
    Epoch 3/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.1521 - accuracy: 0.9598 - val_loss: 0.1110 - val_accuracy: 0.9724
    Epoch 4/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.1048 - accuracy: 0.9722 - val_loss: 0.0827 - val_accuracy: 0.9777
    Epoch 5/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0755 - accuracy: 0.9813 - val_loss: 0.0612 - val_accuracy: 0.9827
    Epoch 6/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0565 - accuracy: 0.9844 - val_loss: 0.0469 - val_accuracy: 0.9849
    Epoch 7/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0453 - accuracy: 0.9879 - val_loss: 0.0368 - val_accuracy: 0.9881
    Epoch 8/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0375 - accuracy: 0.9894 - val_loss: 0.0309 - val_accuracy: 0.9906
    Epoch 9/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0318 - accuracy: 0.9909 - val_loss: 0.0257 - val_accuracy: 0.9928
    Epoch 10/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0273 - accuracy: 0.9920 - val_loss: 0.0241 - val_accuracy: 0.9933
    Epoch 11/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0262 - accuracy: 0.9917 - val_loss: 0.0216 - val_accuracy: 0.9939
    Epoch 12/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0205 - accuracy: 0.9945 - val_loss: 0.0239 - val_accuracy: 0.9926
    Epoch 13/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0183 - accuracy: 0.9943 - val_loss: 0.0199 - val_accuracy: 0.9942
    Epoch 14/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0178 - accuracy: 0.9952 - val_loss: 0.0183 - val_accuracy: 0.9939
    Epoch 15/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0160 - accuracy: 0.9952 - val_loss: 0.0183 - val_accuracy: 0.9939
    Epoch 16/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0137 - accuracy: 0.9966 - val_loss: 0.0187 - val_accuracy: 0.9942
    Epoch 17/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0138 - accuracy: 0.9960 - val_loss: 0.0222 - val_accuracy: 0.9926
    Epoch 18/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0117 - accuracy: 0.9972 - val_loss: 0.0169 - val_accuracy: 0.9948
    Epoch 19/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0101 - accuracy: 0.9973 - val_loss: 0.0169 - val_accuracy: 0.9951
    Epoch 20/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0102 - accuracy: 0.9974 - val_loss: 0.0185 - val_accuracy: 0.9946
    Epoch 21/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0099 - accuracy: 0.9968 - val_loss: 0.0193 - val_accuracy: 0.9939
    Epoch 22/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0108 - accuracy: 0.9962 - val_loss: 0.0180 - val_accuracy: 0.9944
    Epoch 23/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0104 - accuracy: 0.9963 - val_loss: 0.0162 - val_accuracy: 0.9964
    Epoch 24/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0085 - accuracy: 0.9973 - val_loss: 0.0169 - val_accuracy: 0.9951
    Epoch 25/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0086 - accuracy: 0.9973 - val_loss: 0.0174 - val_accuracy: 0.9955
    Epoch 26/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0070 - accuracy: 0.9979 - val_loss: 0.0165 - val_accuracy: 0.9953
    Epoch 27/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0074 - accuracy: 0.9979 - val_loss: 0.0175 - val_accuracy: 0.9955
    Epoch 28/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0082 - accuracy: 0.9971 - val_loss: 0.0167 - val_accuracy: 0.9964
    Epoch 29/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0080 - accuracy: 0.9975 - val_loss: 0.0178 - val_accuracy: 0.9957
    Epoch 30/30
    180/180 [==============================] - 4s 20ms/step - loss: 0.0071 - accuracy: 0.9977 - val_loss: 0.0177 - val_accuracy: 0.9951



```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```




    <matplotlib.legend.Legend at 0x7fe77f359f50>




    

![output_54_1.png](/_posts/output_54_1.png)
    


The plot above shows that Model 3 (with article title and text as input) is able to achieve around 99% validation accuracy with no overfitting observed due to the dropout layers.

# §4. Model Evaluation

Test best model (Model 3) performance on unseen test data


```python
test_url = "https://github.com/PhilChodrow/PIC16b/blob/master/datasets/fake_news_test.csv?raw=true"
test_data  = pd.read_csv(test_url)
```


```python
# convert this data using the make_dataset function defined in Part §2
test_data = make_dataset(test_data)
```


```python
model3.evaluate(test_data)
```

    225/225 [==============================] - 2s 10ms/step - loss: 0.0301 - accuracy: 0.9933





    [0.030065199360251427, 0.9933182001113892]



The accuracy of Model 3 on unseen test data is around an impressive 99%.

# §5. Embedding Visualization

Visualize and comment on the embedding that our model learned


```python
# weights from the embedding layer
weights = 0.5*(model3.get_layer('embeddingTITLE').get_weights()[0]) + 0.5*(model3.get_layer('embeddingTEXT').get_weights()[0])
vocab = vectorize_layer.get_vocabulary()  
```


```python
# convert data into 2 dimension 
pca = PCA(n_components=2)             
weights = pca.fit_transform(weights)
```


```python
embedding_df = pd.DataFrame({
    'word' : vocab, 
    'x0'   : weights[:,0],
    'x1'   : weights[:,1]
})
```


```python
fig = px.scatter(embedding_df, 
                 x = "x0", 
                 y = "x1", 
                 size = list(np.ones(len(embedding_df))),
                 size_max = 2,
                 hover_name = "word")

write_html(fig, "word_embed.html")
```


Since activists are usually firm on their opinions or stance, I find it quite reasonable to have activists[-0.0771, -0.0306] located closely to firm[-0.0872, -0.0326].

Also, population, employees and team are all used to describe a group of people and more often than not, employees in a company are required to work on teams for projects. Therefore, it is logical to have employees[0.121, -0.0280], team[0.137, -0.0158] and population[0.141, -0.0215] to be close in proximity.
