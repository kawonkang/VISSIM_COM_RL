# -*- coding: utf-8 -*-
"""
2017 - Created @author: Hyunjin
2020 - Edited @Kawon

"""

#%% 네트워크 불러오기
def Network(Vissim, Network_Path, inpx_Name, layx_Name):

    import os

    ## Load a Vissim Network:
    Filename                = os.path.join(Network_Path, inpx_Name)
    flag_read_additionally  = False # you can read network(elements) additionally, in this case set "flag_read_additionally" to true
    Vissim.LoadNet(Filename, flag_read_additionally)
    
    ## Load a Layout:
    Filename = os.path.join(Network_Path, layx_Name)
    Vissim.LoadLayout(Filename)
     

#%% 시뮬레이션 파라미터 설정

def SimulationParameter(Vissim, Random_Seed, Controlled_Vehicle_MPR, End_of_simulation, SIM_TIME_STEP, Sim_break_at, QuickMode):  
    
    
    Vissim.Simulation.SetAttValue('SimRes', SIM_TIME_STEP)
    
    Vissim.Simulation.SetAttValue('RandSeed', Random_Seed)
    
    Vissim.Simulation.SetAttValue('SimPeriod', End_of_simulation)
    
    Vissim.Simulation.SetAttValue('SimBreakAt', Sim_break_at)
    
    Vissim.Simulation.SetAttValue('UseMaxSimSpeed', True)
    
    Vissim.Graphics.CurrentNetworkWindow.SetAttValue("QuickMode", QuickMode) 
    
    # Hint: to change the speed use: Vissim.Simulation.SetAttValue('SimSpeed', 10) # 10 => 10 Sim.sec. / s

#%% Link 길이, 키값 반환함수
def GetLinksLength(Vissim):

    Vissim = Vissim
    LINK_key = Vissim.Net.Links.GetMultiAttValues('No') 
    # LINKS LENGTH
    # LINK_10_LENGTH = Vissim.Net.Links.ItemByKey(10).AttValue('Length2D')
    # LINK_20_LENGTH = Vissim.Net.Links.ItemByKey(20).AttValue('Length2D')
    # LINK_30_LENGTH = Vissim.Net.Links.ItemByKey(30).AttValue('Length2D')
    
    # return [LINK_10_LENGTH, LINK_20_LENGTH, LINK_30_LENGTH]
    return LINK_key

#%% Vehicle생성

def VehicleGenerating(Vissim, VEHICLE_GENERATING, CONTROLLED):
    
    if CONTROLLED == True:
        sub = 101
    else:
        sub = 100
    
    if VEHICLE_GENERATING == True:
            
        a = 500
        
        interaction = True
        
        # vehicle_type, link, lane, xcoordinate, desired_speed, interaction
        
        # subject
        Vissim.Net.Vehicles.AddVehicleAtLinkPosition(  sub, 20, 1, a - 20 , 105, interaction)
        
        # front
        Vissim.Net.Vehicles.AddVehicleAtLinkPosition( 100, 20, 1, a + 70 , 95, interaction)
        
        # rear
        Vissim.Net.Vehicles.AddVehicleAtLinkPosition( 100, 20, 1, a - 70 , 105, interaction)
        
        # lead
        Vissim.Net.Vehicles.AddVehicleAtLinkPosition( 100, 20, 2, a + 30 , 110, interaction)
        
        # lag
        Vissim.Net.Vehicles.AddVehicleAtLinkPosition( 100, 20, 2, a - 60 , 115, interaction)    
        
        # Dummy
        Vissim.Net.Vehicles.AddVehicleAtLinkPosition( 400, 20, 1, a - 200 , 0, interaction)
        
        VehicleDataset = Vissim.Net.Vehicles.GetAll()
        
        rearn = VehicleDataset[-4].AttValue('No')
        
        lagn = VehicleDataset[-2].AttValue('No')
        
        
        Vissim.Net.Vehicles.ItemByKey(rearn).SetAttValue('Speed', 105)
        Vissim.Net.Vehicles.ItemByKey(lagn).SetAttValue('Speed', 115)
        
        Vissim.Simulation.RunSingleStep()