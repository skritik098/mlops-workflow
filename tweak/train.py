#!/usr/bin/env python
# coding: utf-8

# In[92]:


from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential, load_model, Model, model_from_json
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu
from keras.utils.np_utils import to_categorical
import numpy as np


# In[171]:


json_file = open('/model/architecture/model_num.json', 'r')

loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)


# In[172]:


loaded_model.summary()


# In[173]:


Layers = []
for i,layer in enumerate(loaded_model.layers):
    #input_output = {'name': layer.name,'input': int(layer.input_shape[1]),'output': int(layer.output_shape[1])}
    if i == int(0):
        (_, inp) = layer.input_shape[0]
        input_output = [layer.name,inp]
    else:
        input_output = [layer.name,int(layer.input_shape[1]),int(layer.output_shape[1])]
    layer_name = 'layer_{}'.format(i)
    input_output.insert(0,layer_name)
    Layers.append(input_output)


# In[167]:


def tweakTheModel(Layers):
    n_units = Layers[-2][-2]
    
    global x
    global inp_layer
    for l in Layers[:-1]:
        if '0' in l[0] :
            l_o = l[2]
            inp_layer = Input(shape=l_o)
            x = Dense(l_o,activation=relu)(inp_layer)
            #print(l_o,l_i,l_n,inp_layer,x)
        else:
            l_n = l[1]
            l_o = l[3]
            x = Dense(l_o,activation=relu)(x)
            #print(l_o,l_n)
    if n_units >= int(64):
        x = Dense(n_units // 2, activation=relu)(x)
    else:
        x = Dense(n_units*2, activation=relu)(x)
    x = Dense(Layers[-1][-1],activation='softmax')(x)
    return x


# In[168]:


out = tweakTheModel(Layers)


# In[169]:


model = Model(inputs=[inp_layer],outputs=[out])


# In[170]:


model.summary()


# In[154]:


from tensorflow.keras.datasets import mnist


# In[155]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[156]:


X_train = x_train.reshape(-1,28*28)
X_test = x_test.reshape(-1,28*28)


# In[157]:


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


# In[158]:


model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])


# In[159]:


history = model.fit(x=X_train,y=y_train_cat,epochs=2)


# In[160]:


results = model.evaluate(x=X_test,y=y_test_cat)


# In[161]:


p = "{}".format(results[1])


# In[162]:


with open('/results/tweak_model_data.txt', 'w') as f:
    f.write(p)


# In[163]:


model_json = model.to_json()


with open("/model/architecture/model_num.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save("/model/model_weight/model_mnist_{}.h5".format(p))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




