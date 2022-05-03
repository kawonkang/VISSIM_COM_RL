from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

import numpy as np
import random
import os

## 기본 모듈
import sys
import win32com.client as com
from datetime import datetime
import pandas as pd
import copy

## My Module
GET_CURRENT_DIR = os.getcwd() # 현재 디렉토리 위치 가져오기
# sys.path.append(os.getcwd() + '\\my_module') # 모듈 위치 추가

import SafetyEvaluation_v5 as SE # Updated File
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option('mode.chained_assignment',  None)

# %%=============================================================================
# Action 
# ===============================================================================
class DesSpeedController(): #Links 관련

    def __init__(self, Vissim):
        
        self.Vissim = Vissim

    # --------------------------------------------------Link Information-----------------------------------------------------
    def UpdateLinkInfo(self):

        self.linkInfo = self.Vissim.Net.Links.GetMultipleAttributes(('No','Name','LinkBehavType','DisplayType','Level',
                                                        'NumLanes','Length2D','IsConn','FromLink','ToLink','LnChgDistDistrDef','HasOvtLn'))
        return self.linkInfo
    # --------------------------------------------------DesSpeed Information-------------------------------------------------
    def UpdateDesSpeedInfo(self):

        self.desInfo = self.Vissim.Net.DesSpeedDecisions.GetMultipleAttributes(('No','Name','Lane','Pos','TimeFrom','TimeTo','DesSpeedDistr(10)','DesSpeedDistr(20)',
                                                        'DesSpeedDistr(30)', 'DesSpeedDistr(145)','DesSpeedDistr(146)','DesSpeedDistr(147)','DesSpeedDistr(148)'))
        return self.desInfo

    def act(self, action, control_Section_List, DefaultDesSpeed, CurrentDesSpeed, currentState_df): #########수정할 예정 
        # return action
        SubjectLinkDesSpeed = copy.deepcopy(CurrentDesSpeed) # array to save link speed output 
        for i in range(len(control_Section_List)): #6개 section에 대해 iteration
            trans_speed = int(CurrentDesSpeed[i,1])
            if action[i] == 0: #행동 1 일반 속도 유지(기존 속도 분포 유지)
                for a in range(1,8):
                    SubjectLinkDesSpeed[i,a] = trans_speed

            elif action[i] == 1:
                for a in range(1,8):
                    SubjectLinkDesSpeed[i,a] = trans_speed - 20

            elif action[i] == 2:
                for a in range(1,8):
                    SubjectLinkDesSpeed[i,a] = trans_speed + 20

            if int(SubjectLinkDesSpeed[i,1]) < 60:
                for a in range(1,8):
                    SubjectLinkDesSpeed[i,a] = 60

            if int(SubjectLinkDesSpeed[i,1]) > 100:
                for a in range(1,8):
                    SubjectLinkDesSpeed[i,a] = 100

        for i in range(len(control_Section_List)-1): #전방구간 속도차 코딩
            for j in range(8): # 차종별 전방속도차 코딩 동일 적용
                if ((int(SubjectLinkDesSpeed[i+1,1]) - int(SubjectLinkDesSpeed[i,1])) > 20):
                    trans_speed = int(SubjectLinkDesSpeed[i, 1]) + 20
                    for a in range(1,8):
                        SubjectLinkDesSpeed[i+1,a] = trans_speed

                elif ((int(SubjectLinkDesSpeed[i+1,1]) - int(SubjectLinkDesSpeed[i,1])) < -20):
                    trans_speed = int(SubjectLinkDesSpeed[i, 1]) - 20
                    for a in range(1,8):
                        SubjectLinkDesSpeed[i+1,a] = trans_speed

        print('subject' + str(SubjectLinkDesSpeed))
        print('Current' + str(CurrentDesSpeed))
        for i in range(0,len(control_Section_List)):
            realAction = int(SubjectLinkDesSpeed[i,1]) - int(CurrentDesSpeed[i,1])
            print('real_action'+str(realAction))
            if realAction == 0:
                action[i] = 0
            elif realAction == -20:
                action[i] = 1
            elif realAction == +20:
                action[i] = 2

        # 속도 산출값 적용 코드
        desInfo_Name = np.array(currentState_df) 
        for i in range(len(control_Section_List)): # of control section
            desInfo_Name_section= desInfo_Name[desInfo_Name[:,1]==control_Section_List[i],:] # 14: 'section' 1: 'Name'
            for j in range(len(desInfo_Name_section)):
                # MPR 도입 시 제어 항목 변경 필요
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(10)',SubjectLinkDesSpeed[i,1])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(20)',SubjectLinkDesSpeed[i,2])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(30)',SubjectLinkDesSpeed[i,3])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(145)',SubjectLinkDesSpeed[i,4])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(146)',SubjectLinkDesSpeed[i,5])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(147)',SubjectLinkDesSpeed[i,6])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(148)',SubjectLinkDesSpeed[i,7])
        return action

    def risk(self,list_section, StateDataset_1min, section_Last_link, timestep):
        StateDataset_temp=StateDataset_1min.copy(True)
        section_risk =np.array([SE.Cal_Safety_section(list_section, StateDataset_temp, section_Last_link, timestep)])
        return section_risk
