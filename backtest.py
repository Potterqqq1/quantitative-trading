import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from datetime import timedelta

factor = 'GB_factor'
FACTOR = pd.read_csv('/Users/haoyue/爱科研/Skill_Project/Factor_Return/{0}.csv'.format(factor), index_col = 0, parse_dates = True)
FMRWN1W = pd.read_csv('/Users/haoyue/爱科研/Skill_Project/Factor_Return/Return.csv', index_col = 0, parse_dates = True)

# long-short strategy
rt_long_short = []
for i in range(83):
    FACTOR_col = FACTOR.iloc[:,[i]]
    FMRWN1W_col = FMRWN1W.iloc[:,[i]]
    FACTOR_FMRWN1W = pd.merge(FACTOR_col, FMRWN1W_col, left_index = True, right_index = True)
    FACTOR_FMRWN1W.columns = ['FACTOR', 'FMRWN1W']
    FACTOR_FMRWN1W = FACTOR_FMRWN1W.dropna()
    FACTOR_FMRWN1W_sorted = FACTOR_FMRWN1W.sort_values('FACTOR', ascending = False)
    FACTOR_FMRWN1W_sorted_1 = FACTOR_FMRWN1W_sorted.iloc[:int(len(FACTOR_FMRWN1W_sorted)*(1/5)),:]
    FACTOR_FMRWN1W_sorted_5 = FACTOR_FMRWN1W_sorted.iloc[int(len(FACTOR_FMRWN1W_sorted)*(4/5))+1:,:]
    r1 = np.sum(FACTOR_FMRWN1W_sorted_1['FMRWN1W'])/len(FACTOR_FMRWN1W_sorted_1)
    r5 = np.sum(FACTOR_FMRWN1W_sorted_5['FMRWN1W'])/len(FACTOR_FMRWN1W_sorted_5)
    delta_r = r1 - r5
    rt_long_short.append(delta_r)

cum_perform_long_short = []
for j in range(len(rt_long_short)):
    product = 1
    for k in range(j+1):
        product = product * (1+rt_long_short[k])
    cum_perform_long_short.append(product)

result_long_short = pd.Series(cum_perform_long_short, index = FMRWN1W.columns)

result_long_short.plot(label = 'SKILL_'+factor+' L/S')
plt.title('Wealth Curve', fontsize = 15)
plt.ylabel('L/S Return', fontsize = 15)
plt.xticks(fontsize = 5, ha = 'center', rotation = -30)
plt.legend()


# long strategy
rt_long = []
for i in range(83):
    FACTOR_col = FACTOR.iloc[:,[i]]
    FMRWN1W_col = FMRWN1W.iloc[:,[i]]
    FACTOR_FMRWN1W = pd.merge(FACTOR_col, FMRWN1W_col, left_index = True, right_index = True)
    FACTOR_FMRWN1W.columns = ['FACTOR', 'FMRWN1W']
    FACTOR_FMRWN1W = FACTOR_FMRWN1W.dropna()
    FACTOR_FMRWN1W_sorted = FACTOR_FMRWN1W.sort_values('FACTOR', ascending = False)
    FACTOR_FMRWN1W_sorted_1 = FACTOR_FMRWN1W_sorted.iloc[:int(len(FACTOR_FMRWN1W_sorted)*(1/5)),:]
    FACTOR_FMRWN1W_sorted_5 = FACTOR_FMRWN1W_sorted.iloc[int(len(FACTOR_FMRWN1W_sorted)*(4/5))+1:,:]
    r1 = np.sum(FACTOR_FMRWN1W_sorted_1['FMRWN1W'])/len(FACTOR_FMRWN1W_sorted_1)
    r5 = np.sum(FACTOR_FMRWN1W_sorted_5['FMRWN1W'])/len(FACTOR_FMRWN1W_sorted_5)
    delta_r = r1
    rt_long.append(delta_r)

cum_perform_long = []
for j in range(len(rt_long)):
    product = 1
    for k in range(j+1):
        product = product * (1+rt_long[k])
    cum_perform_long.append(product)

result_long = pd.Series(cum_perform_long, index = FMRWN1W.columns)

result_long.plot(label = 'SKILL_'+factor+' L')
plt.title('Wealth Curve', fontsize = 15)
plt.ylabel('L Return', fontsize = 15)
plt.xticks(fontsize = 5, ha = 'center', rotation = -30)
plt.legend()
plt.savefig('Result/{0} Wealth Curve.png'.format(factor), dpi=400, bbox_inches='tight')
