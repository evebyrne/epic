#!/usr/bin/env python
# coding: utf-8

# In[79]:


import csv
import pandas as pd
import ast
import cv2
# from fastai.vision import *


FILENAME = 'EPIC_train_object_labels.csv'
df = pd.read_csv(FILENAME)         
print(df.head())
# df_P01 = df[df['video_id']=='P01_01']
# # df_P01.to_csv('EPIC_train_object_labels_P01_01.csv')
# df['frame'] = df[['video_id', 'frame']].apply(lambda x: 'P01_01/'+str(x[0])+'/'+str(x[1]).zfill(10)+'.jpg')
df['frame'] = df['frame'].astype('str')
df['frame'] = '/home/eve/Downloads/EPIC_KITCHENS_2018/object_detection_images/train/'+df['participant_id']+'/'+df['video_id']+'/'+df['frame'].str.zfill(10)+'.jpg'

# print(f'num of rows in P01_01: {df_new.shape}')
df_new = df[df['participant_id']=='P01']
df_new = df_new[['frame', 'bounding_boxes', 'noun']]

df_new


# In[81]:


df_new['bounding_boxes'] = df_new['bounding_boxes'].apply(lambda x: list(ast.literal_eval(x)))
print('num rows with more than one bb')
series = df_new['bounding_boxes'].apply(len)
print(series[series>1].shape)

df_new = df_new.set_index(['frame', 'noun'], append=True).apply(lambda x: x.explode()).reset_index()

print('after exploding')
print(df_new.head())
print(f'shape after exploding: {df_new.shape}')


# In[46]:


df_new['frame'][0]


# In[17]:


img = cv2.imread(str(df_new['frame'][0]))


# In[20]:


img.shape


# In[30]:


df_new['height'] = df_new['frame']
df_new['width']=df_new['frame']


# In[31]:


# df_new['shape'] = df_new['frame'].apply(lambda x:cv2.imread(x).shape)
# df_new['shape'] = df_new['frame'].apply(lambda x:print(x))
for i in range(0, len(df_new['frame'])):
#     print(i)
    img = cv2.imread(df_new['frame'][i])
    if img is not None:
        print(img.shape)
        df_new['height'][i]=img.shape[0]
        df_new['width'][i]=img.shape[1]
    else: print(df_new['frame'][i])
# cv2.imread(df['frame']).shape
# >>> height, width, channels = img.shape


# In[52]:


df_new


# In[82]:


# keras_retianet: boxes` are shaped `(None, None, 4)` (for `(x1, y1, x2, y2)`)
# EPIC_KITCHENS: (<top:int>,<left:int>,<height:int>,<width:int>).

#  YOLO  <object-class> <x> <y> <width> <height>
# <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
# atention: <x> <y> - are center of rectangle (are not top-left corner)
# <x> <y> <width> <height> - float values relative to width and height of image, 
# it can be equal from 0.0 to 1.0

img_height = 1080
img_width = 1920

df_new = df_new[df_new['bounding_boxes'].apply(pd.notnull)]

df_new['top'] = df_new['bounding_boxes'].str[0].astype(int)
df_new['left'] = df_new['bounding_boxes'].str[1].astype(int)
df_new['ah'] =df_new['bounding_boxes'].str[2].astype(int)
df_new['aw'] = df_new['bounding_boxes'].str[3].astype(int)

# new
df_new['x1']=df_new['left']
df_new['y1']=df_new['top']
df_new['x2']=df_new['left'] + df_new['aw']
df_new['y2']=df_new['top'] + df_new['ah']

df_new['cx'] = ((df_new['x1'] + df_new['x2'])/2)/img_width
df_new['cy'] = ((df_new['y1'] + df_new['y2'])/2)/img_height
df_new['rh'] = df_new['ah'] / img_height
df_new['rw'] = df_new['aw'] / img_width

df_new.head()


print('number of rows w invalid bb:')
# invalid_bb_mask = (df_new['x2']<=df_new['x1']) |(df_new['y2']<=df_new['y1'])|(df_new['x1']<0)|(df_new['y1']<0)|(df_new['x2']>img_width)|(df_new['y2']>img_height)
invalid_bb_mask = (df_new['cx']>1) |(df_new['cy']>1)|(df_new['rh']<0)|(df_new['rw']<0)|(df_new['rh']>1)|(df_new['rw']>1)|(df_new['cx']<0)|(df_new['cy']<0)

invalid_bb = df_new[invalid_bb_mask]
df_new = df_new[~invalid_bb_mask]

print(invalid_bb.shape)
df_new


# In[83]:


invalid_bb


# In[90]:


df_new = df_new[['frame', 'noun', 'cx', 'cy', 'rw', 'rh']]


# In[91]:


df_new


# In[118]:


df_new.sort_values(by='frame')


# In[152]:


imgs = df_new['frame'].unique()


# In[154]:


def convert_to_single_noun(x):
    if len(x.split(' '))>1: 
        return x.split(' ')[0]+x.split(' ')[1]
    else: return x


# In[155]:


df_new['noun'] = df_new['noun'].apply(lambda x : convert_to_single_noun(x))


# In[156]:


for img in imgs: 
#     write all gt bb to file called img.txt
    print(img)
    df_ = df_new[df_new['frame']==img]
    print(img.split('.')[0]+'.txt')
    df_[['noun', 'cx', 'cy', 'rw', 'rh']].to_csv(img.split('.')[0]+'.txt', header=False, index = False, sep=' ')


# In[54]:


invalid_bb


# In[ ]:



# y1, x1, y2, x2
im = df_new.iloc[133]
print(im)
img = open_image(im['frame'])
print(img)

bboxes =[im['y1'], im['x1'], im['y2'], im['x2']]
print(bboxes)
# print([bboxes[:,0][:,None], bboxes[:,3][:,None]], 1)
bbox = ImageBBox.create(*img.size, [bboxes], [0], classes=[im['noun']])
img.show(figsize=(6,4), y=bbox)


# In[62]:


df_new.head()


# In[126]:


# CLASSES

classes = df_new['noun'].unique()
# print('classes in P01')
# print(len(classes))
# print(classes)

FILENAME = 'EPIC_noun_classes.csv'
df_classes = pd.read_csv(FILENAME)   

df_classes

classes_to_change = []
for c in classes:
    df_classes['mask'] = df_classes['class_key'].apply(lambda x: c == x)
    if df_classes[df_classes['mask']].shape[0]==0:
        classes_to_change.append(c)
print(classes_to_change)

nuck = []
for x in classes_to_change:
#     print(x)
    arr = x.split(' ')
    n = x
    if len(arr)>1:
        n = arr[1]+':'+arr[0]
#         print(n)
    nuck.append((x,n))
print(nuck)

replacements = []
for n in nuck:
    print(n)
    df_classes['mask'] = df_classes['nouns'].apply(lambda x: n[1] in x)
#     print(df_classes[df_classes['mask']][['class_key']])
    match = df_classes[df_classes['mask']]['class_key'].tolist()
    if(len(match)>0):
        replacements.append((n[0], n[1], match))
    else: 
        df_classes['mask'] = df_classes['class_key'].apply(lambda x: n[1][:-1] == x)
        if df_classes[df_classes['mask']].shape[0]==1:
            match = df_classes[df_classes['mask']]['class_key'].tolist()
            replacements.append((n[0],n[1], match))

replacements
# df_classes


# In[145]:


for r in replacements:
    original = r[0]
    new = r[2][0]
    print(original)
    print(new)
    rows = df_new[df_new['noun']==original]['noun']
    print(rows.shape)
    df_new.loc[df_new['noun']==original,['noun']] = new


# In[146]:


df_new['noun'].unique()


# In[95]:


df_classes[['nouns']]


# In[147]:



# df_new["noun"].loc[df_new['noun'] =='carrots'] = "carrot"

# unknown_class_nouns = ['curry powder', 'cutting board', 'door', 'drainer' ,'powder' ,'rice bag', 'sauce pan', 'saucepan' ,'tofu container' ,'vegetables']
# print('removing unknown class nouns')
# print(df_new.shape)
# # print(df_new[df_new['noun']=="curry powder"])
# print('num rows with unknown classes')
# print(df_new[df_new['noun'].isin(unknown_class_nouns)].shape)
# print('removing')
# df_new = df_new[~df_new['noun'].isin(unknown_class_nouns)]

# print(df_new.shape)

df_new = df_new[['frame', 'x1', 'y1', 'x2', 'y2', 'noun']]
df_new.to_csv('EPIC_train_object_labels_P01_01_annotations.csv', header=False, index=False)
df_P01.to_csv('EPIC_train_object_labels_P01_01.csv')


# In[ ]:



FILENAME = 'EPIC_noun_classes.csv'
df = pd.read_csv(FILENAME)   
df.head()

unknown_class_nouns = ['curry powder', 'cutting board', 'door', 'drainer' ,'powder' ,'rice bag', 'sauce pan', 'saucepan' ,'tofu container' ,'vegetables']
uck = unknown_class_nouns[0]
nuck = []
for x in unknown_class_nouns: 
    arr = x.split(' ')
    if len(arr)>1:
        x = arr[1]+':'+arr[0]
    nuck.append(x)
nuck

