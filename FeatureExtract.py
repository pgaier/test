#!/usr/bin/env python
#Filename: feature_ext.py
"""
======================================================================
Feature extraction module.
Input:     subject zstat map and prior brain image
           zstate.nii.gz and MNI_brain_standard.nii.gz
Output:    train/test sample set file per Subject
           train_sid.txt
           test_sid.txt
           Feature table:
           1.r         (spherical coordinates)
           2.theta     (spherical coordinates)
           3.phi       (spherical coordinates)
           4.t-value   (activity intensity)
           5.s-value   (MNI brain intensity)
           6.p-value   (ROI prob prior)
           7.C1        (mean intensity within 1 level local cuboids)
           8.D1(x)     (mean intensity diff within 1 level local cuboids)
           9.D2(x)     (mean intensity diff within 2 level local cuboids)
           10.D3(x)    (mean intensity within 3 level local cuboids)
           11.CB       (mean intensity)
           12.DB1(x)   (mean intensity diff)
           13.DB2(x)   (mean intensity diff)
           14.DB3(x)   (mean intensity diff)
           15.Dvi(x)   (difference of local intensity and offset cuboid mean)
             ...       (18 offset vectors)
           32.DBi(x)   (difference of local intensity and offset cuboid mean)
             ...       (18 offser vectors)
           51. y       (label value)
Author: dangxiaobin@gmail.com
Date:   2013.12.25
=======================================================================
"""

print (__doc__)

import os
import sys
import re
import time
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mul
import functools 
from sklearn.cross_validation import ShuffleSplit

def main():
    #data reposition and mask constraint
    database = '/nfs/t2/atlas/database'
    contrast = 'face-object'
    sess_list = read_sess_list('./sess')
    roi_mask = nib.load('./mask.nii.gz')
    mask_data = roi_mask.get_data()
    
    #get cubiods mask by max/min the spatical range.
    non = mask_data.nonzero()
    coor_r = np.zeros(6)
    coor_r[0] = np.min(non[0])
    coor_r[1] = np.max(non[0])
    coor_r[2] = np.min(non[1])
    coor_r[3] = np.max(non[1])
    coor_r[4] = np.min(non[2])
    coor_r[5] = np.max(non[2])
    mask_index,mask_img = get_mask(coor_r,mask_data)
    print mask_index.shape

    '''
    brain_m = nib.load("brain_mask.nii.gz")
    data = brain_m.get_data()
    mask_f = data*mask_img
    mask_index = np.transpose(mask_f.nonzero())
    print mask_index.shape
    '''
    ri = roi_mask
    ri._data = mask_img
    nib.save(ri,'final_mask.nii.gz')

    #output the mask coordinate indexs
    writeout(mask_index)

    st = time.time()
    subject_num = len(sess_list)
    sample_num = len(mask_index) # per subject
    
    #random split the train set and test set.
    ss = ShuffleSplit(subject_num, n_iter=1, test_size=0.2,random_state=0)
    for train,test in ss:
        #print("%s \n %s" % (train, test))
        train_list,test_list = sess_list[train],sess_list[test]
    print "train session",train_list
    print "test session",test_list
    train_subjects = "./train_split.sess"
    test_subjects = "./test_split.sess"
    output_sess_list(train_subjects,train_list)
    output_sess_list(test_subjects,test_list)


    #get neighbor offset 1,2,3 radiud cubois.
    of_1 = get_neighbor_offset(1)
    of_2 = get_neighbor_offset(2)
    of_3 = get_neighbor_offset(3)
    #print offset_1.shape, offset_2.shape, offset_3.shape


    ##functions test
    #img = nib.load("MNI_brain.nii.gz")
    #data = img.get_data()
    #m = [[44,22,33],[44,23,33],[44,23,22]]
    #u = get_mean(data,m)
    #print m,u
    #offset = get_neighbor_offset(1)
    #print offset,offset.shape
    #offset = get_neighbor_offset(3)
    #print offset,offset.shape
    #########################################################################
    
    #feature extraction on train set.
    #save the every subject sample per file named sample_sid.txt
    
    pool = mul.Pool(30)
    train_r = pool.map(functools.partial(ext,database=database,mask_index=mask_index,
                      of_1=of_1,of_2=of_2,of_3=of_3,flag="train"), train_list)
    print train_r

    #feature extraction on test set.
    pool = mul.Pool(30)
    test_r = pool.map(functools.partial(ext,database=database,mask_index=mask_index,
                      of_1=of_1,of_2=of_2,of_3=of_3,flag="test"), test_list)
    print test_r

    print "Feature extraction total time:%s"%(time.time()-st)
    return

def ext(sub=None,database=None,mask_index=None,of_1=None,of_2=None,of_3=None,flag=None):
    """
    Wrapper function.
    Tested
    """
    print "I am %s"%sub
    contrast = 'face-object'
    sub_dir =  os.path.join(database,sub)
    sub_ct_dir = os.path.join(sub_dir,contrast)
    f_list = os.listdir(sub_ct_dir)
    for file in f_list:
        if re.search('zstat1.nii.gz',file):
            x_img =  os.path.join(sub_ct_dir,file)
            print x_img
        if re.search('_ff.nii.gz',file): 
            y_img =  os.path.join(sub_ct_dir,file)
            print y_img

    #initial the feature array.
    #3+3+8+18+18+1=51
    feat_buff = np.zeros((len(mask_index),51))
    samples = feature_ext(x_img,y_img,mask_index,
                          of_1,of_2,of_3,feat_buff)
    #print samples[10],samples.shape
    #output samples as a file
    if flag == "train":
        np.savetxt('repo/train/sample_%s'%sub,samples,fmt='%10.5f',delimiter='    ',newline='\n')
    else:
        np.savetxt('repo/test/sample_%s'%sub,samples,fmt='%10.5f',delimiter='    ',newline='\n')
    return True

def writeout(mask_index):
    """
    Write mask coordinates out.
    Tested
    """
    coorf = open('coordinates','w')
    for c in mask_index:
        coorf.write("%d %d %d\n"%(c[0],c[1],c[2]))
    coorf.close()
    return 0

def get_neighbor_offset(radius):
    """
    get neighbor offset for generating cubiods.
    Tested
    """
    offsets = [] 
    for x in np.arange(-radius,radius+1):
        for y in np.arange(-radius,radius+1):
            for z in np.arange(-radius,radius+1):
                offsets.append([x,y,z])
    offsets = np.array(offsets)
    return offsets

def feature_ext(sub_x,sub_y,mask_index,os1,os2,os3,feat):
    """
    Feature extraction fucntion.
    """
    ###offset vectors 
    vecs = np.array([[2,0,0],
            [-2,0,0],
            [0,2,0],
            [0,-2,0],
            [0,0,2],
            [0,0,-2],
            [3,0,0],
            [-3,0,0],
            [0,3,0],
            [0,-3,0],
            [0,0,3],
            [0,0,-3],
            [4,0,0],
            [-4,0,0],
            [0,4,0],
            [0,-4,0],
            [0,0,4],
            [0,0,-4]])
    #print vecs.shape
    #load the zstat image, labeled image and standard brain image
    x = nib.load(sub_x)
    y = nib.load(sub_y)
    MNI = nib.load("./MNI_brain.nii.gz")
    prior = nib.load("./prob_allsub_sthr0.nii.gz")
    
    x_data = x.get_data()
    y_data = y.get_data()
    s_data = MNI.get_data()
    p_data = prior.get_data()

    #center of the region.
    c = np.array([24,23,28])
    
    #one coordinate one feature vec
    for i,coor in enumerate(np.array(mask_index)):
        #print i,coor
        x = coor[0]-c[0]
        y = coor[1]-c[1]
        z = coor[2]-c[2]
        r,t,p = spherical_coordinates(x,y,z)
        feat[i][0] = r
        feat[i][1] = t
        feat[i][2] = p
        
        I = x_data[tuple(coor)]
        S = s_data[tuple(coor)]
        P = p_data[tuple(coor)]
        
        feat[i][3] = I
        feat[i][4] = S
        feat[i][5] = P
        #print I,P

        neigh1 = coor+os1
        #print neigh1,neigh1.shape
        neigh2 = coor+os2
        neigh3 = coor+os3
        
        feat[i][6] = get_mean(x_data,neigh1)
        feat[i][7] = I-get_mean(x_data,neigh1)
        feat[i][8] = I-get_mean(x_data,neigh2)
        feat[i][9] = I-get_mean(x_data,neigh3)
        
        feat[i][10] = get_mean(s_data,neigh1)
        feat[i][11] = S-get_mean(s_data,neigh1)
        feat[i][12] = S-get_mean(s_data,neigh2)
        feat[i][13] = S-get_mean(s_data,neigh3)
        #print feat[i]
        for j,v in enumerate(vecs):
            # print j,v
            p = v + coor
            pn = p + os1
            # print coor,p,pn
            
            feat[i][14+j] = I - get_mean(x_data,pn)
            feat[i][32+j] = S - get_mean(s_data,pn)
            #print I,get_mean(x_data,pn),P,get_mean(s_data,pn)
            #print feat[i]
            #break
        #break
        #Label 1/0
        feat[i][50] = y_data[tuple(coor)] == 1.0
    return  feat

def output_sess_list(sess,list):
    """
    Output session list for data split
    Tested
    """
    file = open(sess,'w')
    for l in list:
        file.write("%s\n"%l)
    file.close()
    return True

def is_inside(v, shape):
    """
    Is coordinate inside the image.
    Tested
    """
    return ((v[0] >= 0) & (v[0] < shape[0]) &
            (v[1] >= 0) & (v[1] < shape[1]) &
            (v[2] >= 0) & (v[2] < shape[2]))

def get_mean(img,indexs):
    """
    Get mean intensity with in mask. 
    Tested
    """
    intensity = 0
    for c in indexs:
        #print c,img[tuple(c)]
        intensity+= img[tuple(c)] 
    return intensity/len(indexs)

def get_mask(c_r,img):
    """
    Get cuboids region mask as spacial constraint.
    Tested
    """
    indexs = []
    mask = np.zeros_like(img)
    shape = mask.shape
    for x in np.arange(c_r[0],c_r[1]+1):
        for y in np.arange(c_r[2], c_r[3]+1):
            for z in np.arange(c_r[4], c_r[5]+1):    
                tmp = [x,y,z]
                inside = is_inside(tmp,shape)
                if inside :
                    indexs.append(tmp)
                    mask[x][y][z]=1
    return np.array(indexs),mask

def spherical_coordinates(x,y,z):
    """
    Rectangular coordinate to spherical coordinate.
    Tested
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if z==0:
        theta = np.arctan(np.sqrt(x**2+y**2)/(z+0.000001))
    else:
        theta = np.arctan(np.sqrt(x**2+y**2)/z)
    if x==0:
        phi = np.arctan(float(y)/(x+0.000001))
    else:
        phi = np.arctan(float(y)/x)
       
    return r,theta,phi

def distance(c,t):
    """
    Compute the spacial distance.
    Tested
    """
    return np.sqrt((c[0]-t[0])**2+(c[1]-t[1])**2+(c[2]-t[2])**2)

def read_sess_list(sess):
    """
    Load subject Id list.
    Tested
    """
    sf = open(sess,'r')
    sess = sf.readlines()
    sess = [line.rstrip('\r\n') for line in sess]
    sess = np.array(sess)
    print sess
    return sess

if __name__=="__main__":
    main()