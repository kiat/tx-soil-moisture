import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''
Rain_mm_Tot: total rain precipitation (mm)
VWC means volumetric water content or soil moisture (m^3/m^3)
1 means sensor depth is 5 cm
2 means sensor depth is 10 cm
3 sensor depth is 20 cm
4 sensor depth is 50 cm

VWC_1_Avg means average volumetric water content at depth 5 cm

EC is electrical conductivity and is used to calculate VWC

T is Temperature of soil (deg C)
'''
def Remove_Outliers(dfsmc,columnname):
    dfsmc[columnname]=dfsmc[columnname].astype(float)
    medianc = dfsmc[columnname].median()
    #std = dfsmc[columnname].std()
    outliers = (dfsmc[columnname] - medianc).abs() > 100
    dfsmc[outliers] = np.nan
    #print(outliers)
    dfsmc[columnname].fillna(medianc, inplace=True)
    return dfsmc[columnname]

def preprocess_TxSON(smfile):
    dfsm = pd.read_csv(smfile, sep=",", header=1)
    dfsm = dfsm.drop(labels=[0,1], axis=0)

    dfsm = dfsm.reset_index(drop=True)

    dfsm['TIMESTAMP']= pd.to_datetime(dfsm['TIMESTAMP'])
    dfsm['TIMESTAMP']=dfsm['TIMESTAMP'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    dfsm['TIMESTAMP']=pd.to_datetime(dfsm['TIMESTAMP'])
    dfsm['date']=dfsm['TIMESTAMP'].apply(lambda x: int(x.strftime("%Y%m%d")))

    #dfsm['VWC_1_Avg']=dfsm['VWC_1_Avg'].astype(float)
    #dfsm['VWC_2_Avg']=dfsm['VWC_2_Avg'].astype(float)
    #dfsm['VWC_3_Avg']=dfsm['VWC_3_Avg'].astype(float)
    #dfsm['VWC_4_Avg']=dfsm['VWC_4_Avg'].astype(float)
    dfsm=dfsm.dropna()
    dfsm= dfsm.replace(-99,np.nan)
    if 'Rain_mm_Tot' in dfsm.columns:
        dfsm['Rain_mm_Tot']=dfsm['Rain_mm_Tot'].astype(float)
    else:
        dfsm['Rain_mm_Tot']=0
    h=[]
    hvwc=[]
    ht=[]
    if 'EC_1_Avg' in dfsm.columns:
        dfsm['EC_1']=Remove_Outliers(dfsm,'EC_1_Avg')
        #plt.plot(dfsm['EC_1_Avg'].astype(float))
        h.append('EC_1')
        if 'EC_2_Avg' in dfsm.columns:
            dfsm['EC_2_Avg']=Remove_Outliers(dfsm,'EC_2_Avg')
            h.append('EC_2')
        if 'EC_3_Avg' in dfsm.columns:
            dfsm['EC_3']=Remove_Outliers(dfsm,'EC_3_Avg')
            h.append('EC_3')
        if 'EC_4_Avg' in dfsm.columns:
            dfsm['EC_4']=Remove_Outliers(dfsm,'EC_4_Avg')
            h.append('EC_4')
    elif 'ECsoil_1_Avg' in dfsm.columns:
        h.append('EC_1_Avg')
        dfsm['EC_1_Avg']=Remove_Outliers(dfsm,'ECsoil_1_Avg')
        if 'ECsoil_2_Avg' in dfsm.columns:
            dfsm['EC_2']=Remove_Outliers(dfsm,'ECsoil_2_Avg')
            h.append('EC_2')
        if 'ECsoil_3_Avg' in dfsm.columns:
            dfsm['EC_3']=Remove_Outliers(dfsm,'ECsoil_3_Avg')
            h.append('EC_3')
        if 'ECsoil_4_Avg' in dfsm.columns:
            dfsm['EC_4']=Remove_Outliers(dfsm,'ECsoil_4_Avg')
            h.append('EC_4')
    if 'VWC_1_Avg' in dfsm.columns:
        dfsm['VWC_1']=Remove_Outliers(dfsm,'VWC_1_Avg')
        hvwc.append('VWC_1')
        if 'VWC_2_Avg' in dfsm.columns:
            dfsm['VWC_2']=Remove_Outliers(dfsm,'VWC_2_Avg')
            hvwc.append('VWC_2')
        if 'VWC_3_Avg' in dfsm.columns:
            dfsm['VWC_3']=Remove_Outliers(dfsm,'VWC_3_Avg')
            hvwc.append('VWC_3')
        if 'VWC_4_Avg' in dfsm.columns:
            dfsm['VWC_4']=Remove_Outliers(dfsm,'VWC_4_Avg')
            hvwc.append('VWC_4')
    if 'T_1_Avg' in dfsm.columns:
        dfsm['T_1']=dfsm['T_1_Avg'].astype(float)#Remove_Outliers(dfsm,'T_1_Avg')
        ht.append('T_1')
        if 'T_2_Avg' in dfsm.columns:
            dfsm['T_2']=dfsm['T_2_Avg'].astype(float)#Remove_Outliers(dfsm,'T_2_Avg')
            ht.append('T_2')
        if 'T_3_Avg' in dfsm.columns:
            dfsm['T_3']=dfsm['T_3_Avg'].astype(float)#Remove_Outliers(dfsm,'T_3_Avg')
            ht.append('T_3')
        if 'T_4' in dfsm.columns:
            dfsm['T_4']=dfsm['T_4_Avg'].astype(float)#Remove_Outliers(dfsm,'T_4_Avg')
            ht.append('T_4')
    hec=h
    
    
    
    return dfsm,hvwc,ht,hec
smfile='CR300_17_TRACER1_SubHourly_soil.dat'
dfsm,hvwc,ht,hec=preprocess_TxSON(smfile)
dfsm