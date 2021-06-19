#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
path="C:/Users/USER/Downloads/Compressed/speech_commands_v0.02/_background_noise_"
master_list=os.listdir(path)


# In[2]:


master_list


# In[9]:


import numpy as np
import soundfile as sf

# read into a numpy array
for item in master_list:
    data, sr = sf.read('C:/Users/USER/Downloads/Compressed/speech_commands_v0.02/_background_noise_/'+item)

    # split
    split = []
    noSections = int(np.ceil(len(data) / sr))

    for i in range(noSections):
        # get 1 second
        temp = data[i*sr:i*sr + sr] # this is for mono audio
        # temp = data[i*sr:i*sr + sr, :] # this is for stereo audio; uncomment and comment line above
        # add to list
        split.append(temp)

    for i in range(noSections):
        # format filename
        filename = '{}_{}.wav'.format(item[:-4],i)
        # write to file
        sf.write(filename, split[i], sr)


# In[ ]:




