import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

### Function for generate xmask
def overlay_bin(binary,heatmap,mask_percent):
    ### --- Sort for the top n% of highest value
    upper=np.sort(heatmap, axis=None)[:int(heatmap.size*mask_percent)][-1]
    
    ### --- Generate binary heatmap that remove pixels below n%
    if np.max(heatmap)==np.min(heatmap):
        upper=np.max(heatmap)+1
    heatmap_bin=np.where(heatmap<upper,0,1)
    
    ### --- Sum both heat map and overlay map to get a new map
    TN_bin=np.where((binary+heatmap_bin)==0,1,0)
    TP_bin=np.where((binary+heatmap_bin)==2,1,0)
    
    FN_bin=np.where((binary-heatmap_bin)==1,1,0)
    FP_bin=np.where((binary-heatmap_bin)==-1,1,0)
    
    TN_num=np.count_nonzero(TN_bin)
    TP_num=np.count_nonzero(TP_bin)
    FN_num=np.count_nonzero(FN_bin)
    FP_num=np.count_nonzero(FP_bin)
    return heatmap_bin, TP_bin, FP_bin, FN_bin, TN_bin, TP_num, FP_num, FN_num, TN_num


### Function for generate xmask
def overlay_bin_a(binary,heatmap,mask_percent):
    ### --- Sort for the top n% of highest value
    upper=np.max(heatmap)*mask_percent
    
    ### --- Generate binary heatmap that remove pixels below n%
    if np.max(heatmap)==np.min(heatmap):
        upper=np.max(heatmap)+1
    heatmap_bin=np.where(heatmap<upper,0,1)
    
    ### --- Sum both heat map and overlay map to get a new map
    TN_bin=np.where((binary+heatmap_bin)==0,1,0)
    TP_bin=np.where((binary+heatmap_bin)==2,1,0)
    
    FN_bin=np.where((binary-heatmap_bin)==1,1,0)
    FP_bin=np.where((binary-heatmap_bin)==-1,1,0)
    
    TN_num=np.count_nonzero(TN_bin)
    TP_num=np.count_nonzero(TP_bin)
    FN_num=np.count_nonzero(FN_bin)
    FP_num=np.count_nonzero(FP_bin)
    return heatmap_bin, TP_bin, FP_bin, FN_bin, TN_bin, TP_num, FP_num, FN_num, TN_num


#Generate the binary image
def binary_img(x_tmp, nlabels):
    if nlabels==0:
        ratio=0.63
    elif nlabels==1:
        ratio=0.69
    
    binary = np.where(x_tmp<ratio,0,1)
    return binary

def gen_contour(binary):
    tmp1=np.zeros(binary.shape)
    binary=binary.astype(int)
    nx=tmp1.shape[0]
    ny=tmp1.shape[1]
    
    width=1
    
    for i in range(width,nx-width):
        for j in range(width,ny-width):
            if binary[i,j]!=0:
                if binary[i-1,j]!=1 or binary[i+1,j]!=1 or binary[i,j+1]!=1 or binary[i,j-1]!=1:
                    tmp1[i,j]=1
    return tmp1

def slice_ave(x_tmp):
    x_tmp=x_tmp[35:49,:,:]
    for i in range(x_tmp.shape[0]):
        min_pixel=float(np.amin(x_tmp[i]))
        max_pixel=float(np.amax(x_tmp[i]))
        img=x_tmp[i]
        if max_pixel!=min_pixel:
            img=(img-min_pixel)/(max_pixel-min_pixel)
        elif max_pixel==min_pixel:
            img=img-max_pixel
        x_tmp[i]=img
    
    x_tmp=np.mean(x_tmp, axis=0)
    min_pixel=float(np.amin(x_tmp))
    max_pixel=float(np.amax(x_tmp))
    
    x_tmp=(x_tmp-min_pixel)/(max_pixel-min_pixel)
    
    return x_tmp


def draw_spect_contour(input_img, binary):
    ## Convert to black and white
    tmp=(input_img-np.amin(input_img))*255/(np.amax(input_img)-np.amin(input_img))
    im = cv2.cvtColor(tmp.astype(np.uint8),cv2.COLOR_GRAY2RGB)
    
    
    tmp=np.argwhere(gen_contour(binary))
    tmp=tmp[:,::-1]
    contour_new=np.reshape(tmp, (tmp.shape[0],1,2))
    
    # Contoured image
    cv2.drawContours(im, contour_new, -1, (255, 0, 0), 1)
    return im

def image_resize(img,nx,ny,nz):
    width = ny
    height = nz
    img1 = np.zeros((img.shape[0], width, height))

    for idx in range(img.shape[0]):
        tmp = img[idx, :, :]
        img_sm = cv2.resize(tmp, (height, width), interpolation=cv2.INTER_LINEAR)
        img1[idx, :, :] = img_sm

    width = nx
    height = ny
    img2 = np.zeros((width, height, img1.shape[2]))

    for idx in range(img1.shape[2]):
        tmp = img1[:, :, idx]
        img_sm = cv2.resize(tmp, (height, width), interpolation=cv2.INTER_LINEAR)
        img2[:, :, idx] = img_sm
    return img2


### Plot image using matplotlib without boundary
def plot_without_boundary(imgtoplot, outname):
    fig = plt.figure(figsize=[6,6])
    ax = fig.add_subplot(111)
    ax.imshow(imgtoplot)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.savefig(outname, bbox_inches='tight', pad_inches=0)
    plt.close('all')
