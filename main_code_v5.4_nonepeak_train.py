# -*- coding: utf-8 -*-
"""
2017 - Created @author: Hyunjin
2020 - Edited @Kawon

"""
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

import numpy as np
import random
import os

## 기본 모듈
import sys
import win32com.client as com # vissim
from datetime import datetime
import pandas as pd
import copy
## My Module
GET_CURRENT_DIR = os.getcwd() 
# sys.path.append(os.getcwd() + '\\my_module') 

# import MyLogger
import InitialSetting
import SafetyEvaluation_v5 as SE # Updated File
import DesSpeedController_v5 as DSC
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
pd.set_option('mode.chained_assignment',  None)
# Ignore method-hidden if the FunctionDef is below the assignment location in the mro
EPISODES = 50

# %%=============================================================================
# Agent
# ===============================================================================

class DQNAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = True

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.02
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999 # 0.999
        self.epsilon_min = 0.001 #0.01
        self.batch_size = 64
        self.train_start = 500
        # create replay memory using deque
        self.memory = deque(maxlen=1000)

        # create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # initialize target model
        self.update_target_model()

        # if self.load_model:
        #     self.model.load_weights("./save_model/Peak__Peak__V16.22.4.3_(reset_fix)_(50~200).h5")
            
    def load_model(self, load_weights):
        self.model.load_weights(load_weights)
        
    # approximate Q function using Neural Network
    # state is input and Q Value of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(256, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(256, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(256, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(256, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # get action from model using epsilon-greedy policy
    def get_action_for_test(self, state):
        q_value = self.model.predict(state)
        return np.argmax(q_value[0])

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(self.batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # Q Learning: get maximum Q value at s' from target model
            if done[i]:
                target[i][int(action[i])] = reward[i]
            else:
                target[i][int(action[i])] = reward[i] + self.discount_factor * (np.amax(target_val[i]))

        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)
        
#%%==============================================================================
# VISSIM parameter
# ===============================================================================
        
class VissimEnvironment():
    
    def __init__(self, network_name, file_name):
        
        # ===========================================================================
        # 초기 설정
        # =============================================================================
        
        ## 분석 TIME STEP
        self.SIM_TIME_STEP = 1 # 초당 Time Step ex. 10 -> 10 timestep/sec
        
        ## WARMUP 시간(초)
        self.WARMUP_PERIOD = 30*60 #1800
        
        ## 분석수행시간(초)
        # self.SIMULATION_PERIOD = 90*60 #5400
        # self.SIMULATION_PERIOD = 500*60 # for test
        self.SIMULATION_PERIOD = 2*60*60 #2시간
        ## 초기 변수 불러오기
        ## Risk Map 표시
        self.GRAPH = False
        self.ThreeD = False
        
        ## OUTPUT RESULT 저장 여부
        self.SAVE_OUTPUT = True
        
        # 차량생성유무
        self.VEHICLE_GENERATING = False
        self.CONTROLLED = False
        
        # For Debugging
        self.DEBUG = True
        


        #강화학습
        self.REINFORCEMENT_LEARNING  = True
        
        ## 로그파일 저장
        # MyLogger.filename(log_filename = datetime.today().strftime("%Y%m%d_%H%M%S") + '_Risk_Map(LOG).log')
        
        ## 결과 저장
        self.output_filename = r'' + datetime.today().strftime("%Y%m%d_%H%M%S") + '_Risk_Map(OUTPUT).txt'
        
        # Connecting the COM Server => Open a new Vissim Window:
        self.Vissim = com.Dispatch("Vissim.Vissim")
        
        
        self.GET_CURRENT_DIR = os.getcwd() # 현재 디렉토리 위치 가져오기
        # sys.path.append(self.GET_CURRENT_DIR+'//my_module') # 모듈 위치 추가
        
        ## 네트워크 불러오기
        InitialSetting.Network(self.Vissim,
                               Network_Path = network_name,
                               inpx_Name= file_name+'.inpx', # inpx_Name='ToyNetwork_ver2_for_License2.inpx',
                               layx_Name= file_name+'.layx') # layx_Name='ToyNetwork_ver2_for_License2.layx')

        self.GET_VEHICLE_DATA_LIST = ('No', 'VehType', 'Lane', 'Speed', 'Pos', 'PosLat', 'Length',
                                      'LnChg', 'DesLane', 'Acceleration', 'DesSpeed','InQueue', 'NextLink',
                                      'CoordRearX', 'CoordRearY','CoordRearZ','CoordFrontX','CoordFrontY','CoordFrontZ', 'RouteNo',
                                      'Spacing', 'LeadTargNo', 'LeadTargType' #for calc TTC
                                     )


    def reset(self, Controlled_Vehicle_MPR, control_Section_List, currentState_df):       
        self.Vissim.Simulation.Stop() 
        
        randseed = random.randrange(1, 100)
        
        self.S = (randseed, Controlled_Vehicle_MPR)

        ## 초기셋팅 INPUT
        InitialSetting.SimulationParameter(self.Vissim,
                                        #    Traffic_Volume = , # 교통량
                                           Random_Seed = self.S[0], # 랜덤시드
                                           Controlled_Vehicle_MPR = self.S[1], # MPR
                                           End_of_simulation = self.SIMULATION_PERIOD + self.WARMUP_PERIOD,  # Simulationsecond [s]
                                           SIM_TIME_STEP = self.SIM_TIME_STEP,  # Simulation Time Step [s]
                                           Sim_break_at = self.WARMUP_PERIOD,  # Simulationsecond [s]
                                           QuickMode = 1) # 0: True, 1: False   
                
        ## Logging
        print('★ Run Simulation : 랜덤시드 = ', self.S[0], ' , CONTROLL_MPR = ', self.S[1]) 
        # MyLogger.logger.info('★ Run Simulation : 랜덤시드 = %d , CONTROLL_MPR = %d ',self.S[0], self.S[1])
        ## deactivate QuickMode
        self.Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", 1)  

        desInfo_Name = np.array(currentState_df) 
        for i in range(len(control_Section_List)): # of control section
            desInfo_Name_section= desInfo_Name[desInfo_Name[:,1]==control_Section_List[i],:] # 14: 'section' 1: 'Name'
            defalut_DesSpeed = np.array([[100,90,100,90,90,90,100],
                                         [100,90,100,90,90,90,100],
                                         [100,90,100,90,90,90,100],
                                         [100,90,100,90,90,90,100],
                                         [100,90,100,90,90,90,100],
                                         [100,90,100,90,90,90,100]])

            for j in range(len(desInfo_Name_section)):
                # MPR 도입 시 제어 항목 변경 필요
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(10)',defalut_DesSpeed[i,0])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(20)',defalut_DesSpeed[i,1])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(30)',defalut_DesSpeed[i,2])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(145)',defalut_DesSpeed[i,3])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(146)',defalut_DesSpeed[i,4])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(147)',defalut_DesSpeed[i,5])
                self.Vissim.net.DesSpeedDecisions.ItemByKey(desInfo_Name_section[j,0]).SetAttValue('DesSpeedDistr(148)',defalut_DesSpeed[i,6])
        ## 시뮬레이션 수행
        self.Vissim.Simulation.RunContinuous() 
         

    def observe(self):
        
        # 차량정보 수집
        AllVehs = self.Vissim.Net.Vehicles.GetMultipleAttributes(self.GET_VEHICLE_DATA_LIST)
        self.AllVehiclesInformation = np.array(AllVehs)
        
    def RunSingleStep(self):
        
        self.Vissim.Simulation.RunSingleStep()    
    
        

# %%=============================================================================
# Training
# =============================================================================
def Training(model_version, network_name, file_name, save_folder, concent_unit, state_size, start, model_name):
    if __name__ == "__main__":

        link_avg_risk = [0.063962447, 0.081093311, 0.130856895,0.130716702, 0.034020632, 0.053524192, 0.080720485]
        save_folder = save_folder

        
        today_and_time = datetime.today().strftime("%Y%m%d_%H%M%S")
                
        env = VissimEnvironment(network_name,file_name)
        des = DSC.DesSpeedController(env.Vissim)
        
        des.UpdateLinkInfo()
        default_desSpeed = des.UpdateDesSpeedInfo()
        # link, section number matching list
        des.link_section_list=np.array(des.linkInfo)[:,0:2]
        link_section = pd.DataFrame(des.link_section_list).astype('str')
        link_section.columns = ['link_No', 'Section']

        # 환경으로부터 상태와 행동의 크기를 가져옴
        state_size = state_size 
        action_size = 5

        control_Section_List = ['section1','section2','section3','section4','section5','section6'] # 속도 컨트롤 구간
        data_collection_link_Name = ['section1','section2','section3','section4','section5','section6','section7'] #위험도 계산 구간
        section_Last_link = ['12', '18', '36', '37', '42', '44', '49'] # 교통량 집계 링크
        # DQN 에이전트의 생성
        agent = DQNAgent(state_size, action_size)

        # 기존의 모델 불러오기
        if model_name != 0:
            agent.model.load_weights(model_name) #2회 이상 수행 시 사용

        # 속도제어차량 비율 (훈련시 : 0)
        Controlled_Vehicle_MPR = 0

        DesSpeedControl_TAll = []

        # iter fot EPISODE 
        for e in range(start, EPISODES):
            print('@@@@@@@@@@@@@@@@@@에피소드: '+str(e))
            done = False
            part_timestep = 0       
            AvgRisk = 0
            AvgSpeed = 0
            AvgReward = 0
            
            ErrorCount = 0
            timestep = 0        
            
            # Controlled_Vehicle_MPR = random.randrange(0,100+1)
            Controlled_Vehicle_MPR = 0 #for test

            currentState = np.array(des.UpdateDesSpeedInfo())
            currentState_df = pd.DataFrame(currentState)
            env.reset(Controlled_Vehicle_MPR, control_Section_List, currentState_df)

            currentStateAll = []
            actionAll       = []
            NewRewardAll    = []
            nextStateAll    = []
            
            # State
            currentState_input_allSection = np.zeros(shape=(len(control_Section_List), state_size))
            nxtState_input_allSection = np.zeros(shape=(len(control_Section_List), state_size))

            SafetyRewardAll = [0 for i in range(len(control_Section_List))]
            weightForReward = []       
        
            StateDataset_1min = [] ##하나의 에피소드 내 1분마다 업데이트
            StateDataset_All  = [] ##하나의 에피소드 내 1분 자료기반 집계(평균)자료 모음

            sec_timer = 0 #0 1~60
            min_timer = 0 #0~4 - 5분단위

            concent_unit = concent_unit # 집계단위(분단위로 설정)

            first_state = True #집계단위 첫번째 완료 변수

            # for state, reward
            # 1. 1분 결과 (1x7)
            risk_list  = [0 for i in range(len(data_collection_link_Name))]
            risk_speed = [0 for i in range(len(data_collection_link_Name))]

            # new model 1-------------------------------------------------------

            vol  = [0 for i in range(len(data_collection_link_Name))]
            acc  = [0 for i in range(len(data_collection_link_Name))]
            ttc_veh  = [0 for i in range(len(data_collection_link_Name))]
            ttc_tot  = [0 for i in range(len(data_collection_link_Name))]
            ttc_cnt  = [0 for i in range(len(data_collection_link_Name))]
            cpi  = [0 for i in range(len(data_collection_link_Name))]
            jerk = [0 for i in range(len(data_collection_link_Name))]
            yaw  = [0 for i in range(len(data_collection_link_Name))]

            # 2. 집계단위(n) 결과 (nx7)
            risk_list_All  = []
            risk_speed_All = []

            vol_All = []
            acc_All = []
            ttc_veh_All = []
            ttc_tot_All = []
            ttc_cnt_All = []
            cpi_All = []
            jerk_All = []
            yaw_All = []

            # 3. 집계단위(n) 평균 (1x7)
            risk_list_T  = []
            risk_speed_T = []

            vol_T  = []
            acc_T  = []
            ttc_veh_T = []
            ttc_tot_T = []
            ttc_cnt_T = []
            cpi_T  = []
            jerk_T = []
            yaw_T = []

            # 4. 전체 에피소드 내 집계단위별 평균 (5400/n x 7)
            Risk_TAll       = []
            Risk_S_TAll     = []
            Speed_TAll      = []
            Reward_TAll     = []
            Action_TAll     = []

            vol_TAll  = []
            acc_TAll  = []
            ttc_veh_TAll = []
            ttc_tot_TAll = []
            ttc_cnt_TAll = []
            cpi_TAll  = []
            jerk_TAll = []
            yaw_TAll  = []

            action_list = [0 for i in range(len(control_Section_List))]
            
            #초기화
            currentDesSpeed = np.array([['section1', '0', '0', '0', '0', '0', '0', '0'],['section2', '0', '0', '0', '0', '0', '0', '0'],['section3', '0', '0', '0', '0', '0', '0', '0'],
                                        ['section4', '0', '0', '0', '0', '0', '0', '0'],['section5', '0', '0', '0', '0', '0', '0', '0'],['section6', '0', '0', '0', '0', '0', '0', '0']])        
            nxtDesSpeed     = np.array([['section1', '0', '0', '0', '0', '0', '0', '0'],['section2', '0', '0', '0', '0', '0', '0', '0'],['section3', '0', '0', '0', '0', '0', '0', '0'],
                                        ['section4', '0', '0', '0', '0', '0', '0', '0'],['section5', '0', '0', '0', '0', '0', '0', '0'],['section6', '0', '0', '0', '0', '0', '0', '0']])
            try:
                while not done:

                    # timestep 1 단위로 정보 수집, 1분(timestep60) 집계
                    if (first_state == False):
                        env.observe() # current state observe

                        # currentState => current DesSpeed of links
                        # current DesSpeed => S => get_action(S)
                        currentState = np.array(des.UpdateDesSpeedInfo())
                        currentState_df = pd.DataFrame(currentState)

                        currentState_df.columns = ['No','Name','Lane','Pos','TimeFrom','TimeTo','DesSpeedDistr(10)','DesSpeedDistr(20)',
                                                    'DesSpeedDistr(30)', 'DesSpeedDistr(145)','DesSpeedDistr(146)','DesSpeedDistr(147)','DesSpeedDistr(148)']
                        currentState_df['Link'] = currentState_df['Lane'].str.split('-').str[0]
                        currentState_df = pd.merge(currentState_df, link_section, left_on = 'Link', right_on = 'link_No',how = 'left')

                        # Update for section 1~6
                        for i in range(len(control_Section_List)):
                            currentDesSpeed[i]=currentState_df[currentState_df.Name == control_Section_List[i]][['Name','DesSpeedDistr(10)','DesSpeedDistr(20)', 'DesSpeedDistr(30)', 'DesSpeedDistr(145)','DesSpeedDistr(146)','DesSpeedDistr(147)','DesSpeedDistr(148)']].iloc[0].tolist()
                        
                        for c in range(len(control_Section_List)):
                            currentState_input = np.append(currentDesSpeed[c],np.array([Risk_S[c], vol_T[c+1], acc_T[c+1], ttc_cnt_T[c+1], cpi_T[c+1], jerk_T[c+1], yaw_T[c+1]]), axis=-1)
                            currentState_input = currentState_input[1:].reshape(1, state_size).astype(np.float64)

                            # action결과, State결과 저장
                            action_list[c] = agent.get_action(currentState_input)
                            currentState_input_allSection[c] = currentState_input[0]

                        new_action_list = des.act(action_list, control_Section_List, default_desSpeed, currentDesSpeed,currentState_df)

                        print("action_list in timestep" + str(timestep))
                        print(new_action_list)

                        min_timer = 0
                        sec_timer = 0

                        # for record
                        actionAll       = new_action_list      
                        currentStateAll = currentState_input_allSection  
                    # ==================================================================    
                    # get next State
                    # ==================================================================
                    nxtState_done = False

                    while not nxtState_done: #(Next state 5분동안 반복 수행)
                        part_timestep +=1
                        if part_timestep > env.SIMULATION_PERIOD:
                            raise Exception
                            
                        env.RunSingleStep()
                        timestep += 1       
                        sec_timer += 1                 
                        env.observe()
                        if (sec_timer == 1): 
                            Tot_file = pd.DataFrame(np.array(env.AllVehiclesInformation))
                            Tot_file['sec'] = 1
                        else:
                            StateDataset_1min = pd.DataFrame(np.array(env.AllVehiclesInformation)) #1초마다 업데이트
                            StateDataset_1min['sec'] = sec_timer #초 입력해주고
                            Tot_file = pd.concat([Tot_file,StateDataset_1min], ignore_index=True, axis=0) #Tot_file로 저장 ~1분까지
                        # 1min
                        if (sec_timer == 60):
                            print("@@@@@@@@@@@@@@@@@Episode : ", e)
                            print(str(min_timer), 'min record')
                            print("second: ",sec_timer,", min: ",min_timer)
                            sec_timer = 0
                            min_timer += 1 #1분 추가
                            # print(Tot_file)
                            Tot_file.columns = ['No', 'VehType', 'Lane', 'Speed', 'Pos', 'PosLat', 'Length',
                                                'LnChg', 'DesLane', 'Acceleration', 'DesSpeed','InQueue', 'NextLink',
                                                'CoordRearX', 'CoordRearY','CoordRearZ','CoordFrontX','CoordFrontY','CoordFrontZ', 'RouteNo',
                                                'Spacing', 'LeadTargNo', 'LeadTargType', 'sec']
                            Tot_file['Link'] = Tot_file.Lane.str.split('-').str[0]
                            Tot_file = pd.merge(Tot_file, link_section, left_on ='Link', right_on = 'link_No', how = 'left')

                            #Get Raw Trajectory Data
                            # Tot_file.to_excel(save_folder+'/raw_Episode' + str(e) + '_' + str(timestep) + '.xlsx')
                            
                            # fix
                            for c in range(len(data_collection_link_Name)):
                                des.UpdateLinkInfo()
                                current_Section_risk = des.risk(data_collection_link_Name[c],Tot_file,section_Last_link[c],timestep)
                                risk_value = current_Section_risk[0,8]
                                risk_list[c] = risk_value
                                risk_speed[c] = current_Section_risk[0,0]
                                
                                vol[c]  = current_Section_risk[0,1]
                                acc[c]  = current_Section_risk[0,2]
                                ttc_veh[c]  = current_Section_risk[0,3]
                                cpi[c]  = current_Section_risk[0,4]
                                jerk[c]  = current_Section_risk[0,5]
                                yaw[c]  = current_Section_risk[0,6]
                                ttc_tot[c] = current_Section_risk[0,7]
                                ttc_cnt[c] = current_Section_risk[0,8]

                                print('risk current section'+str(c+1))
                                print(current_Section_risk)
                            print('total risk list in timestep'+ str(timestep))
                            print(risk_list)

                            if (min_timer == 1):
                                risk_T_1min_All = [risk_list] 
                                risk_S_1min_All = [risk_speed]

                                vol_All  = [vol]
                                acc_All  = [acc]
                                ttc_veh_All  = [ttc_veh]
                                ttc_tot_All = [ttc_tot]
                                ttc_cnt_All = [ttc_cnt]
                                cpi_All  = [cpi]
                                jerk_All = [jerk]
                                yaw_All  = [yaw]

                            else: 
                                risk_T_1min_All = np.vstack([risk_T_1min_All,risk_list])
                                risk_S_1min_All = np.vstack([risk_S_1min_All,risk_speed])

                                vol_All  = np.vstack([vol_All,vol])
                                acc_All  = np.vstack([acc_All,acc])
                                ttc_veh_All = np.vstack([ttc_veh_All,ttc_veh])
                                ttc_tot_All = np.vstack([ttc_tot_All,ttc_tot])
                                ttc_cnt_All = np.vstack([ttc_cnt_All,ttc_cnt])
                                cpi_All  = np.vstack([cpi_All,cpi])
                                jerk_All = np.vstack([jerk_All,jerk])
                                yaw_All  = np.vstack([yaw_All,yaw])

                            if (min_timer == concent_unit):
                                # Next State 집계
                                nxtState = np.array(des.UpdateDesSpeedInfo())
                                nxtState_df = pd.DataFrame(nxtState)
                                nxtState_df.columns = ['No','Name','Lane','Pos','TimeFrom','TimeTo','DesSpeedDistr(10)','DesSpeedDistr(20)',
                                                        'DesSpeedDistr(30)', 'DesSpeedDistr(145)','DesSpeedDistr(146)','DesSpeedDistr(147)','DesSpeedDistr(148)']
                                nxtState_df['Link'] = nxtState_df['Lane'].str.split('-').str[0]
                                nxtState_df = pd.merge(nxtState_df, link_section, left_on = 'Link', right_on = 'link_No',how = 'left')

                                # 1분간 집계된 Section별 변수들 평균
                                if (concent_unit != 1):
                                    Risk_T = risk_T_1min_All.mean(axis=0)
                                    Risk_S = risk_S_1min_All.mean(axis=0)

                                    vol_T  =  vol_All.mean(axis=0)
                                    acc_T  =  acc_All.mean(axis=0)
                                    ttc_veh_T  =  ttc_veh_All.mean(axis=0)
                                    ttc_tot_T  =  ttc_tot_All.mean(axis=0)
                                    ttc_cnt_T  =  ttc_cnt_All.mean(axis=0)
                                    cpi_T  =  cpi_All.mean(axis=0)
                                    jerk_T =  jerk_All.mean(axis=0)
                                    yaw_T  =  yaw_All.mean(axis=0)

                                else: #concent unit -- 1min
                                    Risk_T = risk_T_1min_All[0]
                                    Risk_S = risk_S_1min_All[0]

                                    vol_T  =  vol_All[0]
                                    acc_T  =  acc_All[0]
                                    ttc_veh_T  =  ttc_veh_All[0]
                                    ttc_tot_T  =  ttc_tot_All[0]
                                    ttc_cnt_T  =  ttc_cnt_All[0]
                                    cpi_T  =  cpi_All[0]
                                    jerk_T =  jerk_All[0]
                                    yaw_T  =  yaw_All[0]                                    

                                # action 수행 후의 nxtDesSpeed 가져오기
                                for i in range(len(control_Section_List)):
                                    nxtDesSpeed[i]=nxtState_df[nxtState_df.Name == control_Section_List[i]][['Name','DesSpeedDistr(10)','DesSpeedDistr(20)', 'DesSpeedDistr(30)', 'DesSpeedDistr(145)','DesSpeedDistr(146)','DesSpeedDistr(147)','DesSpeedDistr(148)']].iloc[0].tolist()
                                
                                    # for print desSpeed
                                    DesSpeedControl_TAll.append(np.append(nxtDesSpeed[i],timestep))

                                print(nxtDesSpeed)

                                for c in range(len(control_Section_List)):
                                    nxtState_input = np.append(nxtDesSpeed[c],np.array([Risk_S[c], vol_T[c+1], acc_T[c+1], ttc_cnt_T[c+1], cpi_T[c+1], jerk_T[c+1], yaw_T[c+1]]), axis=-1)
                                    nxtState_input = nxtState_input[1:].reshape(1, state_size).astype(np.float64)
                                    # nxtState_input_all
                                    nxtState_input_allSection[c] = nxtState_input[0]

                                # 1분마다 STACK~ episode 끝까지
                                Risk_TAll.append(np.append(Risk_T,timestep))
                                Risk_S_TAll.append(np.append(Risk_S, timestep))
                                vol_TAll.append(np.append(vol_T, timestep))
                                acc_TAll.append(np.append(acc_T, timestep))

                                ttc_veh_TAll.append(np.append(ttc_veh_T, timestep))
                                ttc_tot_TAll.append(np.append(ttc_tot_T, timestep))
                                ttc_cnt_TAll.append(np.append(ttc_cnt_T, timestep))
                                cpi_TAll.append(np.append(cpi_T, timestep))

                                jerk_TAll.append(np.append(jerk_T, timestep))
                                yaw_TAll.append(np.append(yaw_T, timestep))

                                SpeedForReward = [0 for i in range(len(data_collection_link_Name))]
                                NewRewardAll = [0 for i in range(len(control_Section_List))]

                                Risk_pref=np.array(Risk_TAll).mean(axis=0)
                                # =============================================================================
                                # Training Model !!
                                # =============================================================================                        
                                # save the sample <s, a, r, s'> to the replay memory
                                if(not first_state):
                                    for i in range(len(control_Section_List)):

                                        # reward version 3
                                        NewRewardAll[i] = - Risk_T[i]

                                    for i in range(len(control_Section_List)):
                                        if(np.any(np.isnan(np.array(np.r_[currentStateAll[i],actionAll[i],NewRewardAll[i],nxtState_input_allSection[i]], dtype = np.float64)))):
                                            continue
                                        else: # @!@
                                            state         = currentStateAll[i] #(time - current) CurrentSection DesSpeed, nxtSection risk
                                            action        = actionAll[i]                     #(time - current) 
                                            reward        = NewRewardAll[i]                  #(time - nxt) nxtSection risk
                                            next_state    = nxtState_input_allSection[i]     #(time - nxt) CurrentSection DesSpeed, nxtSection risk
                                            agent.append_sample(state, action, reward, next_state, done)
                                            print("@@@@@@@@@@@@@@@@ append done!!! @@@@@@@@@@@@@@@@@@")
                                    
                                    Action_TAll.append(np.append(actionAll, timestep))
                                    Reward_TAll.append(np.append(NewRewardAll, timestep))
                
                                # every time step do the training
                                    agent.train_model()

                                else:
                                    first_state = False #return first_state with Nxt_Tot_file, CurDesSpeed

                                Risk_T_tmp = copy.deepcopy(Risk_T)
                                Risk_S_tmp = copy.deepcopy(Risk_S)

                                min_timer = 0
                                nxtState_done = True

            except: 
                done          = True

                agent.update_target_model()

                # every episode, plot the play time
                print(" ")
                
                print("Episode : ", e+1," / ",  EPISODES, '\n'
                    ,"Memory length = ", len(agent.memory), '\n'
                    ,"Epsilon       = ", agent.epsilon, '\n'
                    )
                

                print("----------------------------------------------------------")
                
                print(" ")

                pd.DataFrame(Risk_TAll).to_csv(save_folder+'Episode'+str(e)+'_Risk_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                pd.DataFrame(Risk_S_TAll).to_csv(save_folder+'Episode'+str(e)+'_Risk_S_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                pd.DataFrame(vol_TAll).to_csv(save_folder+'Episode'+str(e)+'_vol_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                pd.DataFrame(acc_TAll).to_csv(save_folder+'Episode'+str(e)+'_acc_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                
                pd.DataFrame(ttc_veh_TAll).to_csv(save_folder+'Episode'+str(e)+'_ttc_veh_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                pd.DataFrame(ttc_tot_TAll).to_csv(save_folder+'Episode'+str(e)+'_ttc_tot_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                pd.DataFrame(ttc_cnt_TAll).to_csv(save_folder+'Episode'+str(e)+'_ttc_cnt_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                pd.DataFrame(cpi_TAll).to_csv(save_folder+'Episode'+str(e)+'_cpi_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                pd.DataFrame(jerk_TAll).to_csv(save_folder+'Episode'+str(e)+'_jerk_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                pd.DataFrame(yaw_TAll).to_csv(save_folder+'Episode'+str(e)+'_yaw_TAll.csv',header=['section1','section2','section3','section4','section5','section6','section7','timestep'])
                                                 
                pd.DataFrame(Action_TAll).to_csv(save_folder+'Episode'+str(e)+'_action_TAll.csv',header=['section1','section2','section3','section4','section5','section6','timestep'])
                pd.DataFrame(Reward_TAll).to_csv(save_folder+'Episode'+str(e)+'_reward_TAll.csv',header=['section1','section2','section3','section4','section5','section6','timestep'])

                pd.DataFrame(DesSpeedControl_TAll).to_csv(save_folder+'Episode'+str(e)+'_DesSpeedControl_TAll.csv', header = ['section','DesSpeedDistr(10)','DesSpeedDistr(20)', 'DesSpeedDistr(30)', 'DesSpeedDistr(145)','DesSpeedDistr(146)','DesSpeedDistr(147)', 'DesSpeedDistr(148)','timestep'])
                # save the model
                agent.model.save_weights("C:/../save_model/"+model_version+".h5") 
        
        if e % 50 == 0:
            agent.model.save_weights("C:/../save_model/"+model_version+".h5")   



Training(model_version = "Nonepeak_V5.3.1_train_MPR100"
        ,network_name = 'C:\\..\\testnetwork\\V5.3.1\\Nonepeak_crash_train\\Nonepeak__MPR100'
        ,file_name = 'calibration-Nonepeaktime_MPR100'
        ,save_folder = './evaluation_results/V5.3.1/Nonepeak_crash_train/MPR100/'
        ,concent_unit = 1
        ,state_size = 14
        ,start = 0
        ,model_name = 0)

