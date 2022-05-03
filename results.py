
import pandas as pd
from pandas import DataFrame, read_csv, Series
import numpy as np
import os
import math
import glob
import openpyxl

#데이터 '폴더' 경로
read = r'E:/../Non-peaktime with VSL/MPR0/' 

file_name = '_Risk_TAll'
file_name2 = '_reward_TAll'
extract_col = ['section1','section2','section3', 'section4', 'section5', 'section6', 'section7']
total_output = []
total_reward_output = []
#변경X
extension = 'csv'
os.chdir(read)
arr = glob.glob('*' + file_name+'.{}'.format(extension))
arr2 = glob.glob('*' + file_name2+'.{}'.format(extension))

for i in range(len(arr)): # 뒤에서 10개 에피소드 결과 추출
    file_tmp = pd.read_csv(read+'Episode'+str(i)+'_Risk_TAll.csv')[extract_col]
    file_tmp_out = file_tmp.mean()
    total_output.append(np.append(file_tmp_out, i))


for j in range(len(arr2)): # 뒤에서 10개 에피소드 결과 추출
    reward_raw = pd.read_csv(read+'Episode'+str(j)+'_reward_TAll.csv')[extract_col[:-1]]
    if j == 0:
        reward_raw_all = reward_raw
    else:
        reward_raw_all = pd.concat([reward_raw_all, reward_raw], axis=0)

total_output_mean = np.nanmean(np.array(total_output[-1:]), axis=0)

writer = pd.ExcelWriter(read+'results_mean.xlsx', engine='xlsxwriter')

pd.DataFrame(total_output_mean.transpose()).to_excel(writer, sheet_name = 'total_mean')
pd.DataFrame(total_output, columns = np.append(extract_col,'episode')).to_excel(writer, sheet_name = 'total')
pd.DataFrame(reward_raw_all, columns = extract_col[:-1]).to_excel(writer, sheet_name = 'total_reward_raw')

writer.save()