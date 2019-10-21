import pandas as pd
import os
import sys

def read_input():
    if len(sys.argv)!=2:
        read_file='input_tmp'
    else:
        read_file=sys.argv[1]

    f = open(read_file, 'r')
    lines = f.read().split('\n')
    for i in range(len(lines)-1):
        print(lines[i])
        lines[i]=lines[i].split('=')[1]

    device, PathOutput, group, n_model, epochs, batch_size, fold, init_train =lines[0:8]
    group = int(group)
    n_model = int(n_model)
    epochs= int(epochs)
    batch_size= int(batch_size)
    fold = int(fold)
    init_train = int(init_train)
    if group==0:
        group=['Control','PD']
    elif group==1:
        group=['Control','SWEDD']

    f.close()
    return device, PathOutput, group, n_model, epochs, batch_size, fold, init_train


def generate_input_label_file(input_file):
    ## Acquire data of download images file from CSV file
    PathMRI = "../dat_mri_T1_gz/"
    PathBrain = "../dat_brainmask/"
    PathSpect = "../dat_spect/"
    ### df = pd.read_csv('../dat_csv/MRI_T1_180528_6_14_2018.csv')
    ### df=df[df['Description'].isin(['T1-anatomical'])]
    ### df=df.drop(['Downloaded', 'Format','Type','Description','Modality'], axis=1)
    ### df=df.reset_index(drop=True)
    
    df = pd.read_csv('../dat_csv/SPECT_180430_4_30_2018.csv')
    df=df.drop(['Downloaded', 'Format','Type','Description','Modality'], axis=1)
    df=df.reset_index(drop=True)
    
    df2 = pd.read_csv('../dat_csv/MDS_UPDRS_Part_III.csv')
    #Sum only NP3 -----------------
    tmp=df2.iloc[:,8:41]
    tmp=tmp.sum(axis=1)
    ##-------------------------------
    df2=df2.iloc[:, [2,5,43]]
    df2.insert(loc=2, column='MDS3', value=tmp)
    
    ## ---------- Gathering date and time -----------------
    df['Acq Date']=pd.to_datetime(df['Acq Date'])
    df2['INFODT_tmp']=pd.to_datetime(df2['INFODT'])
    
    for i in range(len(df)):
        ii=df.loc[i, 'Subject']
        nd1=df.loc[i, 'Acq Date']
        tmp=df2.loc[df2['PATNO'] == ii]
        e=[]
        for j in range(len(tmp)):
            ii=tmp.index[j]
            nd2=df2.loc[ii, 'INFODT_tmp']
            e.append(abs(nd1-nd2))
        if len(e) != 0:
            ii=tmp.index[e.index(min(e))]
            df.loc[i, 'INFODT_0']= df2.loc[ii, 'INFODT']
            df.loc[i, 'MDS3_0']= df2.loc[ii, 'MDS3']
            df.loc[i, 'NHY_0']= df2.loc[ii, 'NHY']
            ii_0=ii
        
        ###----ADD 1 YEAR TO THE ACQ DATE ---------------------------------------
        nd1=nd1+pd.Timedelta(365, unit='d')
        e=[]
        for j in range(len(tmp)):
            ii=tmp.index[j]
            nd2=df2.loc[ii, 'INFODT_tmp']
            e.append(abs(nd1-nd2))
        if len(e) != 0:
            ii=tmp.index[e.index(min(e))]
            df.loc[i, 'INFODT_1']= df2.loc[ii, 'INFODT']
            df.loc[i, 'MDS3_1']= df2.loc[ii, 'MDS3']
            df.loc[i, 'NHY_1']= df2.loc[ii, 'NHY']
    
    df=df.reset_index(drop=True)

    ###----------- Add more data of SBR ratio and Visual interpretation
    df_vis = pd.read_csv('../dat_csv/DaTSCAN_SPECT_Visual_Interpretation_Assessment.csv')
    df_img= pd.read_csv('../dat_csv/DaTscan_Imaging.csv')
    df_sbr=pd.read_csv("../dat_csv/DATScan_Analysis.csv")
    
    for i in range(len(df_sbr)):
        ii=df_sbr.loc[i, 'PATNO']
        event_id=df_sbr.loc[i, 'EVENT_ID']
        
        df_img.loc[(df_img['PATNO'] == ii) & (df_img['EVENT_ID'] == event_id), 'CAUDATE_R']=df_sbr.loc[i, 'CAUDATE_R']
        df_img.loc[(df_img['PATNO'] == ii) & (df_img['EVENT_ID'] == event_id), 'CAUDATE_L']=df_sbr.loc[i, 'CAUDATE_L']
        df_img.loc[(df_img['PATNO'] == ii) & (df_img['EVENT_ID'] == event_id), 'PUTAMEN_R']=df_sbr.loc[i, 'PUTAMEN_R']
        df_img.loc[(df_img['PATNO'] == ii) & (df_img['EVENT_ID'] == event_id), 'PUTAMEN_L']=df_sbr.loc[i, 'PUTAMEN_L']
    
    ## ---------- Gathering date and time -----------------
    df_vis['SCANDATE_tmp']=pd.to_datetime(df_vis['SCANDATE'])
    df_img['INFODT_tmp']=pd.to_datetime(df_img['INFODT'])
    
    count=0
    for i in range(len(df)):
        patno=df.loc[i, 'Subject']
        nd0=df.loc[i, 'Acq Date']
    
        ###---ADD SBR RATIO--------------------------------------
        tmp=df_img.loc[df_img['PATNO'] == patno]
        e=[]
        for j in range(len(tmp)):
            ii=tmp.index[j]
            nd1=df_img.loc[ii, 'INFODT_tmp']
            e.append(abs(nd0-nd1))
        if len(e) != 0:
            ii=tmp.index[e.index(min(e))]
            df.loc[i, 'INFODT_SBR']= df_img.loc[ii, 'INFODT']
            df.loc[i, 'CAUDATE_R'] = df_img.loc[ii, 'CAUDATE_R']
            df.loc[i, 'CAUDATE_L'] = df_img.loc[ii, 'CAUDATE_L']
            df.loc[i, 'PUTAMEN_R'] = df_img.loc[ii, 'PUTAMEN_R']
            df.loc[i, 'PUTAMEN_L'] = df_img.loc[ii, 'PUTAMEN_L']
    
        ###---ADD Vis Interpretation--------------------------------------
        tmp=df_vis.loc[df_vis['PATNO'] == patno]
        e=[]
        for j in range(len(tmp)):
            ii=tmp.index[j]
            nd1=df_vis.loc[ii, 'SCANDATE_tmp']
            e.append(abs(nd0-nd1))
        if len(e) != 0:
            ii=tmp.index[e.index(min(e))]
            df.loc[i, 'INFODT_VIS']= df_vis.loc[ii, 'SCANDATE']
            df.loc[i, 'VISINTRP']  = df_vis.loc[ii, 'VISINTRP']
            
    
    df=df.reset_index(drop=True)
   

    ##--------Check for duplicate image ID
    ### seen = set()
    ### 
    ### for dirName, subdirList, fileList in os.walk(PathSpect):
    ###     for filename in fileList:
    ###         if ".dcm" in filename.lower():
    ###             tmp=filename.split("_")
    ###             tmp1=tmp[1]
    ###             tmp2=tmp[9].split(".")[0][1:]
    ###             number=int(tmp2)
    ###             if number in seen:
    ###                 print("SPECT image ID repeated!",number)
    ###                 print(df.loc[df['Image Data ID'] == number])
    ###             seen.add(number)
    ### 
    ### ---------- SELECT DATA WITH SAME SHAPE and filepath to csv file----------------
    count=0
    lst=[]
    for dirName, subdirList, fileList in os.walk(PathSpect):
        for filename in fileList:
            if ".dcm" in filename.lower():
                tmp=filename.split("_")
                tmp1=tmp[1]
                tmp2=tmp[9].split(".")[0][1:]
                os.path.join(dirName,filename)
                ii=df.loc[df['Image Data ID'] == int(tmp2)].index[0]
                name=os.path.join(dirName,filename)
                count=count+1
                #if img.shape[1]==109:
                df.loc[ii, 'SPECT File Path']=name
                           
    df=df.reset_index(drop=True)
    df.to_csv(input_file, index=0)
