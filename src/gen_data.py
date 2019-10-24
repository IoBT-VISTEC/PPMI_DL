import cv2
import numpy as np
import nibabel as nib
import pydicom as dicom

def image_resize(img,nx=128,ny=128,nz=128):
    width = ny
    height = nz
    img1 = np.zeros((img.shape[0], width, height))
    
    for idx in range(img.shape[0]):
        tmp = img[idx, :, :]
        img_sm = cv2.resize(tmp, (width, height), interpolation=cv2.INTER_LINEAR)
        img1[idx, :, :] = img_sm
    
    width = nx
    height = ny
    img2 = np.zeros((width, height, img1.shape[2]))
    
    for idx in range(img1.shape[2]):
        tmp = img1[:, :, idx]
        img_sm = cv2.resize(tmp, (width, height), interpolation=cv2.INTER_LINEAR)
        img2[:, :, idx] = img_sm
    return img2

def name_to_data(name, n_count, df, group):
    name=np.trim_zeros(np.sort(name))
    n_dat=len(name)
    if np.sum(n_count)!=n_dat:
        print("---- ERROR NUMBER OF DATA -----",n_count,n_dat)
    count=0
    init_data=True
    sex=['M','F']
    vis=['negative','positive']
    for i in range(n_dat):
        ii=df.loc[df['Image Data ID'] == int(name[i])].index[0]
        dat_ii=df.loc[ii, :]
        
        filename=dat_ii['SPECT File Path']
        cls=group.index(dat_ii['Group'])
        age=int(dat_ii['Age'])
        n_sex=sex.index(dat_ii['Sex'])
        mds3_0=dat_ii['MDS3_0']
        mds3_1=dat_ii['MDS3_1']
        sbr_ratio=dat_ii[['CAUDATE_R', 'CAUDATE_L', 'PUTAMEN_R', 'PUTAMEN_L']].values
        n_vis=vis.index(dat_ii['VISINTRP'])
        
        nlabels=cls
        #if mds3_1-mds3_0 > 0:
        #    nlabels=1
        #else:
        #    nlabels=0
        
        ds = dicom.read_file(filename)
        img=ds.pixel_array[:,:,:]

        min_pixel=float(np.amin(img))
        max_pixel=float(np.amax(img))
        img=img.astype(float)
        img=(img-min_pixel)/(max_pixel-min_pixel)

        ##img_mirror=img[::-1,:,:]
        
        ## Crop images ------------------------------------------
        ###  x_sum=np.where(np.sum(np.sum(img, axis=1), axis=1) > 0)
        ###  y_sum=np.where(np.sum(np.sum(img, axis=0), axis=1) > 0)
        ###  z_sum=np.where(np.sum(np.sum(img, axis=0), axis=0) > 0)
        ###  
        ###  x1, x2 = x_sum[0][0], x_sum[0][-1]
        ###  y1, y2 = y_sum[0][0], y_sum[0][-1]
        ###  z1, z2 = z_sum[0][0], z_sum[0][-1]
        ###  
        ###  img = img[x1:x2, y1:y2, z1:z2]
        ###  
        ###  tmp=img.shape
        ###  pad1 = np.zeros((180-tmp[0],tmp[1], tmp[2]), dtype = np.int)
        ###  img = np.concatenate((img, pad1), axis = 0)
        ###  tmp=img.shape
        ###  pad1 = np.zeros((tmp[0],180-tmp[1], tmp[2]), dtype = np.int)
        ###  img = np.concatenate((img, pad1), axis = 1)
        ###  tmp=img.shape
        ###  pad1 = np.zeros((tmp[0],tmp[1], 210-tmp[2]), dtype = np.int)
        ###  img = np.concatenate((img, pad1), axis = 2)

        ###------------ Pad Image        
        ### tmp=img.shape
        ### pad1 = np.zeros((int((109-tmp[0])/2), tmp[1], tmp[2]), dtype = np.int)
        ### img = np.concatenate((pad1, img, pad1), axis = 0)
        ### 
        ### tmp=img.shape
        ### pad1 = np.zeros((tmp[0], tmp[1], (int((109-tmp[2])/2))), dtype = np.int)
        ### img = np.concatenate((pad1, img, pad1), axis = 2)
        
        if init_data:
            img_new = img
            n_dat_tot=n_dat
            img_shape=(*img_new.shape,1)
            data=np.zeros((n_dat_tot,*img_shape))
            labels=np.zeros(n_dat_tot, dtype=int)
            aux_inp=np.zeros((n_dat_tot,2))
            dat_sbr=np.zeros((n_dat_tot,4))
            dat_vis=np.zeros(n_dat_tot)
            print(data.shape)
            init_data=False

        img_new = img
        data[count]=np.reshape(img_new, img_shape)
        labels[count]=nlabels
        aux_inp[count][0]=age
        aux_inp[count][1]=n_sex
        dat_sbr[count]=sbr_ratio
        dat_vis[count]=n_vis
        count=count+1
            
    return data, aux_inp, labels, dat_sbr, dat_vis
