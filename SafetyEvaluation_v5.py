# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
import math
from scipy.stats import norm 
pd.set_option('mode.chained_assignment',  None)
pd.set_option('use_inf_as_na', True)

#%% 
#Information_1min_list_TotFile : dataframe -> mint : minuate
def calcTTC(Speed, Speed_lead, Spacing, Type_lead):
    if ((Type_lead == 'VEHICLE')&(Speed_lead < Speed)&(Speed > 0)&(Spacing >= 0)):
        TTC = np.float(Spacing / (Speed- Speed_lead) * 3.6)
    else:
        TTC = 9999
    return TTC
    
def calc_yaw(dx,dy):
    if(dx==0):
        angle = 0
    else:
        angle = abs(math.atan(dy/dx)*180/np.pi)
        if(dy<=0): #횡선차 -
            if(dx>0): #종선차 +
                angle = 90 + angle #2상한
            else:     #종선차 -
                angle = 270 - angle #3상한
        else:      #횡선차 + 
            if(dx>0): #종선차 +
                angle = 90 - angle #2상한
            else:     #종선차 -
                angle = 270 + angle #3상한
    if angle >= 360 :
        angle -= 360
    if angle < 0   :
        angle += 360
    return angle
def Cal_Safety_section(link_section, Information_1min_list_TotFile_raw, section_Last_link, timestep):

    try:
        Information_1min_list_TotFile = Information_1min_list_TotFile_raw.copy()
        Information_1min_list_TotFile[['Speed', 'Pos', 'PosLat', 'Acceleration','CoordFrontX','CoordFrontY','Spacing']] = Information_1min_list_TotFile[['Speed', 'Pos', 'PosLat', 'Acceleration','CoordFrontX','CoordFrontY','Spacing']].astype(float)
        Information_1min_list_TotFile['Link']=Information_1min_list_TotFile['Lane'].str.split('-').str[0]  
        Information_1min_list_TotFile = Information_1min_list_TotFile.sort_values(['No','sec']) # No : vehicle num
        Information_1min_list_TotFile['VehType2'] = 0

        # volume
        vol_tmp = len(Information_1min_list_TotFile[Information_1min_list_TotFile.Link == section_Last_link].No.unique())

        Information_1min_list_TotFile[['VehType','VehType2']] = Information_1min_list_TotFile[['VehType','VehType2']].astype('int') 
        Information_1min_list_TotFile.loc[(Information_1min_list_TotFile['VehType'] == 100)|(Information_1min_list_TotFile['VehType'] == 601),'VehType2'] = 10 #Car
        Information_1min_list_TotFile.loc[(Information_1min_list_TotFile['VehType'] == 110)|(Information_1min_list_TotFile['VehType'] == 611),'VehType2'] = 11 #Car_CACC
        Information_1min_list_TotFile.loc[(Information_1min_list_TotFile['VehType'] == 200)|(Information_1min_list_TotFile['VehType'] == 701),'VehType2'] = 20 #HGV
        Information_1min_list_TotFile.loc[(Information_1min_list_TotFile['VehType'] == 300)|(Information_1min_list_TotFile['VehType'] == 801),'VehType2'] = 30 #Bus

        Information_1min_list_TotFile[['nxt_No','nxt_CoordFrontX','nxt_CoordFrontY','nxt_Speed','nxt_Acc']] = Information_1min_list_TotFile[['No','CoordFrontX','CoordFrontY','Speed','Acceleration']].shift(-1)
        TotFile_section=Information_1min_list_TotFile[(Information_1min_list_TotFile['Section']==link_section)&(Information_1min_list_TotFile['No']==Information_1min_list_TotFile['nxt_No'])].copy()

        # jerk calc first
        TotFile_section['Acceleration_abs'] = np.abs(TotFile_section['Acceleration'])
        TotFile_section['jerk'] = np.abs(TotFile_section['nxt_Acc']-TotFile_section['Acceleration'])
        TotFile_section['deltaX']  = TotFile_section['nxt_CoordFrontX'] - TotFile_section['CoordFrontX']
        TotFile_section['deltaY']  = TotFile_section['nxt_CoordFrontY'] - TotFile_section['CoordFrontY']
        TotFile_section['angle']   = 0 
        TotFile_section['angle']   = TotFile_section.apply(lambda row: calc_yaw(row.deltaX,row.deltaY), axis=1)
        TotFile_section[['nxt_No','nxt_angle']] = TotFile_section[['No','angle']].shift(-1)
        TotFile_section['yaw'] = np.abs(TotFile_section['nxt_angle'] - TotFile_section['angle'])
        # r calc _ Speed, Acceleration, jerk, yaw
        TotFile_section[['nnxt_No','nxt_Speed','nxt_jerk','nxt_yaw']] = TotFile_section[['nxt_No','Speed','jerk','yaw']].shift(-1)
        TotFile_section = TotFile_section[TotFile_section.No==TotFile_section.nnxt_No]
        TotFile_section_out = TotFile_section[['No','Speed','Acceleration_abs','jerk','yaw']].groupby('No').mean()

        mean_Speed = TotFile_section_out.Speed.mean()
        mean_Acc = TotFile_section_out.Acceleration_abs.mean()
        mean_Jerk = TotFile_section_out.jerk.mean()
        mean_Yaw = TotFile_section_out.yaw.mean()

        # calc TTC
        TotFile_section_2 = pd.merge(TotFile_section,TotFile_section[['No','Speed', 'Acceleration', 'sec']],left_on=['LeadTargNo','sec'],right_on=['No','sec'], how='left', suffixes=('_targ','_lead'))
        TotFile_section_2['TTC'] = TotFile_section_2.apply(lambda x: calcTTC(float(x.Speed_targ), float(x.Speed_lead), float(x.Spacing), x.LeadTargType),axis=1)

        min_veh_TTC = TotFile_section_2[['No_targ', 'TTC']].groupby('No_targ').min()
        count_TTC = len(min_veh_TTC[(min_veh_TTC.TTC < 4)])

        #섹션 내 TTC exp 평균
        TotFile_section_2['exp_TTC'] = np.exp(-TotFile_section_2['TTC'])
        #차량별
        mean_veh_TTC = TotFile_section_2[['No_targ','exp_TTC']].groupby('No_targ').max().exp_TTC.mean()
        #전체
        total_mean_TTC = TotFile_section_2['exp_TTC'].mean()

        # calc CPI
        TotFile_section_2['DRAC'] = (((TotFile_section_2.Speed_lead - TotFile_section_2.Speed_targ)/3.6)**2)/(2*TotFile_section_2.Spacing)
        TotFile_section_2[TotFile_section_2.Acceleration_targ == 0]['DRAC'] = np.NaN 
        TotFile_section_2[['No_nxt', 'DRAC_nxt']] = TotFile_section_2[['No_targ','DRAC']].shift(-1)
        mu, sigma = 4, 1.4
        TotFile_section_3 = TotFile_section_2[TotFile_section_2.No_targ==TotFile_section_2.No_nxt]
        TotFile_section_3['DRAC_z'] = (TotFile_section_3.DRAC_nxt-TotFile_section_3.DRAC - mu)/sigma
        TotFile_section_CPI = TotFile_section_3[(TotFile_section_3.Acceleration_targ < 0)&(TotFile_section_3.Acceleration_lead < 0)&(TotFile_section_3.Speed_targ > 0)]
        TotFile_section_CPI['CP'] = norm.cdf(TotFile_section_CPI['DRAC_z'], 0, 1)

        mean_CPI = TotFile_section_CPI.CP.mean()

        # total average 
        mean_Speed = TotFile_section_2.Speed_targ.mean()
        mean_Acc = np.abs(TotFile_section_2.Acceleration_targ).mean()
        # mean_Jerk = TotFile_section_out.jerk.mean()
        
        return mean_Speed, vol_tmp, mean_Acc, mean_veh_TTC, mean_CPI, mean_Jerk, mean_Yaw, total_mean_TTC, count_TTC

    except:
        print("Timestep"+str(timestep)+" error!!!")
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan