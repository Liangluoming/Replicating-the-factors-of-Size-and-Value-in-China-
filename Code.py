import numpy as np 
import pandas as pd 
import datetime as dt
import matplotlib.pyplot as plt
from functools import reduce
from tqdm import tqdm
import warnings 
warnings.filterwarnings('ignore')
import os
def get_all_path(dirpath, *suffix):
    """
    获得指定目录下所有指定后缀的文件路径
    
    @param dirpath: 要搜索的目录路径
    @param *suffix: 可变参数，指定文件的后缀，如 '.txt', '.csv' 等
    
    @return: 包含所有符合条件文件路径的列表
    """
    PathArray = []
    # 遍历目录及其子目录
    for r, ds, fs in os.walk(dirpath):
        for fn in fs:
            # 检查文件后缀是否符合要求
            if os.path.splitext(fn)[1] in suffix:
                # 拼接完整的文件路径
                fname = os.path.join(r, fn)
                PathArray.append(fname)
    return PathArray

def datalag(x, index, column, value, n, rename):
    ## 数据滞后
    return x.pivot_table(index=index, columns=column, values=value).shift(n).unstack().reset_index().rename(columns={0:rename})

def data_tidy(data):
    ## CSMAR的财报整理函数
    data = data[data['Typrep']=='A'].reset_index(drop=True) # 合并财务报表
    ## 可以验证差错更正披露日期在4个月内的样本占比很低，忽略对结果影响不大
    data.drop(columns=['ShortName', 'Typrep', 'IfCorrect', 'DeclareDate'], inplace=True)
    data['Stkcd'] = data['Stkcd'].astype(int) ## 转成数值的形式
    data['Accper'] = pd.to_datetime(data['Accper']).dt.date ## 转成日期格式
    return data

## 季报披露时间文件
df_anttime = pd.read_csv("CSMAR年、中、季报基本情况文件.csv")
df_anttime['Stkcd'] = df_anttime['Stkcd'].astype(int) ## 转成数值的形式
df_anttime.drop_duplicates(subset=['Stkcd', 'Accper'], ignore_index=True, inplace=True) # 去重
df_anttime = df_anttime[['Stkcd', 'Accper', 'Annodt']] ## 保留需要的列
df_anttime['Accper'] = pd.to_datetime(df_anttime['Accper']).dt.date ## 转成日期格式
df_anttime['Annodt'] = pd.to_datetime(df_anttime['Annodt']).dt.date 

## CSMAR财报数据
df_asset = data_tidy(pd.read_csv("CSMAR资产负债表.csv"))
df_profit = data_tidy(pd.read_csv("CSMAR利润表.csv"))
df_cash = data_tidy(pd.read_csv("CSMAR现金流量表.csv"))
df_cash_indirect = data_tidy(pd.read_csv("CSMAR现金流量表(间接法).csv"))
df_indicator = pd.read_csv("CSMAR披露财务指标.csv")
df_indicator['Stkcd'] = df_indicator['Stkcd'].astype(int)
df_indicator['Accper'] = pd.to_datetime(df_indicator['Accper']).dt.date
df_indicator.rename(columns={'F020101':'非经常性损益'}, inplace=True)
df_indicator = df_indicator[['Stkcd', 'Accper', '非经常性损益']]
## 合并数据
df_fiscal = reduce(lambda left, right:pd.merge(left, right, on=['Stkcd', 'Accper'], how='outer'), [df_asset, df_profit, df_cash, df_cash_indirect, df_indicator])
df_fiscal = pd.merge(df_anttime, df_fiscal, on=['Stkcd', 'Accper'], how='left')
df_fiscal['Year'] = pd.to_datetime(df_fiscal['Accper']).dt.year
df_fiscal['Month'] = pd.to_datetime(df_fiscal['Accper']).dt.month
## 筛选需要的样本区间
df_fiscal = df_fiscal[(df_fiscal['Accper']>=dt.date(2004,1, 1)) & (df_fiscal['Accper']<=dt.date(2024, 9, 30))].reset_index(drop=True)
## 参照Stambaugh个人主页的 “Online Appendix for “Size and Value in China”，保留所需要的字段
df_fiscal.rename(columns={
    'A001101000':'货币资金', 
    'A001107000':'交易性金融资产',
    'A001100000':'流动资产合计',
    'A001000000':'资产总计',
    'A002101000':'短期借款',
    'A002107000':'应付票据',
    'A002113000':'应交税费',
    'A002125000':'一年内到期的非流动负债',
    'A002201000':'长期借款',
    'A002100000':'流动负债合计',
    'A003200000':'少数股东权益',
    'A003000000':'所有者权益合计', 
    'B002000000':'净利润', 
    'C005001000':'期初现金及现金等价物余额', 
    'C006000000':'期末现金及现金等价物余额',
    'D000103000':'固定资产折旧、油气资产折耗、生产性生物资产折旧',
    'D000104000':'无形资产摊销',
    'A002000000':'负债合计'
}, inplace=True)

## 保留所需的列
keepcol = [
    'Stkcd', 'Accper', 'Annodt', 'Year', 'Month', '货币资金', '交易性金融资产', '流动资产合计', '资产总计', '短期借款',
    '应付票据', '应交税费', '一年内到期的非流动负债', '长期借款', '流动负债合计', '少数股东权益', '所有者权益合计',
    '净利润', '期初现金及现金等价物余额', '期末现金及现金等价物余额', '固定资产折旧、油气资产折耗、生产性生物资产折旧',
    '无形资产摊销', '负债合计', '非经常性损益'
]
df_fiscalfactor = df_fiscal[keepcol]

## 资产、负债、净利润这种关键变量，缺失值用上一期填补，如果仍有缺失则删去数据。
df_fiscalfactor.loc[df_fiscalfactor['资产总计']==0, '资产总计'] = np.nan
df_fiscalfactor['资产总计'] = df_fiscalfactor.groupby(['Stkcd'])['资产总计'].fillna(method='ffill')

df_fiscalfactor = df_fiscalfactor[df_fiscalfactor['资产总计'].notnull()]
df_fiscalfactor.loc[df_fiscalfactor['负债合计']==0, '负债合计'] = np.nan
df_fiscalfactor['负债合计'] = df_fiscalfactor.groupby(['Stkcd'])['负债合计'].fillna(method='ffill')
df_fiscalfactor = df_fiscalfactor[df_fiscalfactor['负债合计'].notnull()]

df_fiscalfactor.loc[df_fiscalfactor['净利润']==0, '净利润'] = np.nan
df_fiscalfactor['净利润'] = df_fiscalfactor.groupby(['Stkcd'])['净利润'].fillna(method='ffill')
df_fiscalfactor = df_fiscalfactor[df_fiscalfactor['净利润'].notnull()]

df_fiscalfactor['期初现金及现金等价物余额'] = df_fiscalfactor.groupby(['Stkcd'])['期初现金及现金等价物余额'].fillna(method='ffill')
df_fiscalfactor['期末现金及现金等价物余额'] = df_fiscalfactor.groupby(['Stkcd'])['期末现金及现金等价物余额'].fillna(method='ffill')
df_fiscalfactor = df_fiscalfactor[df_fiscalfactor['期初现金及现金等价物余额'].notnull()]
df_fiscalfactor = df_fiscalfactor[df_fiscalfactor['期末现金及现金等价物余额'].notnull()]

## 对于其他列缺失默认为0（部分行业的流动资产也可能为0，如平安银行）
columns=['流动资产合计', '流动负债合计', '货币资金', '交易性金融资产', '短期借款', '应付票据', '应交税费', '一年内到期的非流动负债', '长期借款', '期初现金及现金等价物余额', '期末现金及现金等价物余额',
         '固定资产折旧、油气资产折耗、生产性生物资产折旧', '无形资产摊销', '非经常性损益', '少数股东权益']
for col in columns:
    df_fiscalfactor[col] = df_fiscalfactor[col].replace(np.nan, 0)

## 生成一些构建变量所需要的财务指标
df_fiscalfactor = pd.merge(df_fiscalfactor, datalag(df_fiscalfactor, 'Accper', 'Stkcd', '资产总计', 1, '上一季度资产总计'), on=['Stkcd', 'Accper'], how='left')
df_fiscalfactor = pd.merge(df_fiscalfactor, datalag(df_fiscalfactor, 'Accper', 'Stkcd', '资产总计', 4, '上一年度资产总计'), on=['Stkcd', 'Accper'], how='left')
df_fiscalfactor = pd.merge(df_fiscalfactor, datalag(df_fiscalfactor, 'Accper', 'Stkcd', '流动资产合计', 1, '上一季度流动资产合计'), on=['Stkcd', 'Accper'], how='left')
df_fiscalfactor = pd.merge(df_fiscalfactor, datalag(df_fiscalfactor, 'Accper', 'Stkcd', '流动负债合计', 1, '上一季度流动负债合计'), on=['Stkcd', 'Accper'], how='left')
df_fiscalfactor = pd.merge(df_fiscalfactor, datalag(df_fiscalfactor, 'Accper', 'Stkcd', '一年内到期的非流动负债', 1, '上一季度一年内到期的非流动负债'), on=['Stkcd', 'Accper'], how='left')
df_fiscalfactor = pd.merge(df_fiscalfactor, datalag(df_fiscalfactor, 'Accper', 'Stkcd', '应付票据', 1, '上一季度应付票据'), on=['Stkcd', 'Accper'], how='left')
df_fiscalfactor = pd.merge(df_fiscalfactor, datalag(df_fiscalfactor, 'Accper', 'Stkcd', '应交税费', 1, '上一季度应交税费'), on=['Stkcd', 'Accper'], how='left')

## 对于披露日期缺失的，根据财报披露规则，以限制内的最后一天做填补
indices = df_fiscalfactor[df_fiscalfactor['Annodt'].isnull()].index
for idx in indices:
    yr, mt = df_fiscalfactor['Year'][idx], df_fiscalfactor['Month'][idx]
    if mt == 3:
        df_fiscalfactor.loc[idx, 'Annodt'] = dt.date(yr, 4, 30)
    elif mt == 9:
        df_fiscalfactor.loc[idx, 'Annodt'] = dt.date(yr, 10, 31)
    elif mt == 6:
        df_fiscalfactor.loc[idx, 'Annodt'] = dt.date(yr, 8, 31)
    elif mt == 12:
        df_fiscalfactor.loc[idx, 'Annodt'] = dt.date(yr+1, 4, 30)

## 存在某些年报的披露时间晚于新一期季报披露时间，对于这部分观测保留最新一期季报的数据
# 将上一季财报的披露时间向新一季shift
df_annt_pivot = df_fiscalfactor.pivot_table(index=['Accper'], columns=['Stkcd'], values=['Annodt'], aggfunc="first").shift(1).unstack().droplevel(0).reset_index().rename(columns={0:'L_Annodt'})
df_fiscalfactor = pd.merge(df_fiscalfactor, df_annt_pivot, on=['Stkcd', 'Accper'], how='left')
## 比较当季披露时间和上一季财报披露时间，保证使用最新一季财报数据
df_fiscalfactor = df_fiscalfactor[~(df_fiscalfactor['Annodt'] < df_fiscalfactor['L_Annodt'])]
df_fiscalfactor = df_fiscalfactor.drop_duplicates(subset=['Stkcd', 'Annodt'], keep='last').reset_index(drop=True)

## 参照Stambaugh个人主页的 “Online Appendix for “Size and Value in China”，构建变量
df_fiscalfactor['经常性净利润'] = df_fiscalfactor['净利润'] - df_fiscalfactor['非经常性损益']
df_fiscalfactor['股东权益合计(不含少数股东权益)'] = df_fiscalfactor['所有者权益合计'] - df_fiscalfactor['少数股东权益']
df_fiscalfactor['ROE'] = df_fiscalfactor['经常性净利润'] / df_fiscalfactor['股东权益合计(不含少数股东权益)']
df_fiscalfactor['Investment'] = df_fiscalfactor['资产总计'] / df_fiscalfactor['上一年度资产总计'] - 1
df_fiscalfactor['DeltaCA'] = df_fiscalfactor['流动资产合计'] - df_fiscalfactor['上一季度流动资产合计']
df_fiscalfactor['DeltaCash'] = df_fiscalfactor['期末现金及现金等价物余额'] - df_fiscalfactor['期初现金及现金等价物余额']
df_fiscalfactor['DeltaCL'] = df_fiscalfactor['流动负债合计'] - df_fiscalfactor['上一季度流动负债合计']
df_fiscalfactor['DeltaSTD'] = (df_fiscalfactor['一年内到期的非流动负债'] + df_fiscalfactor['应付票据']) - (df_fiscalfactor['上一季度一年内到期的非流动负债'] + df_fiscalfactor['上一季度应付票据'])
df_fiscalfactor['DeltaTP'] = df_fiscalfactor['应交税费'] - df_fiscalfactor['上一季度应交税费']
df_fiscalfactor['DP'] = df_fiscalfactor['固定资产折旧、油气资产折耗、生产性生物资产折旧'] + df_fiscalfactor['无形资产摊销']
df_fiscalfactor['Accrual'] = (df_fiscalfactor['DeltaCA'] - df_fiscalfactor['DeltaCash']) - (df_fiscalfactor['DeltaCL'] - df_fiscalfactor['DeltaSTD'] - df_fiscalfactor['DeltaTP']) - df_fiscalfactor['DP']
df_fiscalfactor['AccrualComponent'] = 2 * df_fiscalfactor['Accrual'] / (df_fiscalfactor['上一年度资产总计'] + df_fiscalfactor['资产总计'])
df_fiscalfactor['OperatingAsset'] = df_fiscalfactor['资产总计'] - df_fiscalfactor['货币资金'] - df_fiscalfactor['交易性金融资产']
df_fiscalfactor['OperatingLiabilities'] = df_fiscalfactor['资产总计'] - df_fiscalfactor['短期借款'] - df_fiscalfactor['长期借款'] - df_fiscalfactor['所有者权益合计']
df_fiscalfactor['NOA'] = (df_fiscalfactor['OperatingAsset'] - df_fiscalfactor['OperatingLiabilities']) / df_fiscalfactor['上一季度资产总计']

## 保留所需要的列
keepcol = ['Stkcd', 'Accper', 'Annodt', 'Year', 'Month','经常性净利润', '股东权益合计(不含少数股东权益)', '期末现金及现金等价物余额', '期初现金及现金等价物余额',
 'ROE', 'Investment', 'Accrual', 'AccrualComponent', 'NOA']
df_factors = df_fiscalfactor[keepcol]
df_factors = df_factors[df_factors['Year']>=2005].reset_index(drop=True) ## 筛选样本区间

## CSMAR股票市场交易日历
df_calenda = pd.read_csv("中国股票市场交易日历.csv")
df_calenda = df_calenda[df_calenda['Markettype'].isin([1, 4, 16, 32, 64])] #A股
df_calenda = df_calenda[df_calenda['State']=='O'] #开市
df_calenda = df_calenda.drop_duplicates(subset=['Clddt']).reset_index(drop=True) #去重
df_calenda['Clddt'] = pd.to_datetime(df_calenda['Clddt']).dt.date #转换成日期格式
df_calenda.sort_values(by='Clddt', ignore_index=True, inplace=True) #按先后顺序排序

## 披露日期映射到披露后的第一个交易日（不是当日）
annodt_unique = df_factors['Annodt'].unique()
tmap = {}
for ant in tqdm(annodt_unique):
    tmap[ant] = df_calenda[df_calenda['Clddt']>ant]['Clddt'].values[0]

df_factors['CloseTrddt'] = df_factors['Annodt'].map(tmap)
df_factors = df_factors.drop_duplicates(subset=['Stkcd', 'CloseTrddt'], keep='last').reset_index(drop=True)

## 读取CSMAR A股日度交易数据
paths = get_all_path("A股日度交易数据", '.pq')
df_trade = [pd.read_parquet(path) for path in paths]
df_trade = pd.concat(df_trade)
df_trade['Trddt'] = pd.to_datetime(df_trade['Trddt']).dt.date
df_trade['Year'] = pd.to_datetime(df_trade['Trddt']).dt.year
df_trade = df_trade[(df_trade['Year']>=2005) & (df_trade['Year']<=2024)] ##筛选样本
df_trade.sort_values(by=['Stkcd', 'Trddt'], ignore_index=True, inplace=True)
df_trade = df_trade[df_trade['Trddt'].isin(df_calenda['Clddt'].unique())].reset_index(drop=True) ##确保交易日期正确

## 合并数据，数据被合并到映射到的交易日期
df_tradefactor = pd.merge(df_trade, df_factors, left_on=['Stkcd', 'Trddt', 'Year'], right_on=['Stkcd', 'CloseTrddt', 'Year'], how='left')

## 往后填补缺失值
columns=['经常性净利润', '股东权益合计(不含少数股东权益)', '期末现金及现金等价物余额', '期初现金及现金等价物余额', 'ROE', 'Investment', 'Accrual', 'AccrualComponent', 'NOA']
for col in columns:
    df_tradefactor[col] = df_tradefactor.groupby(['Stkcd'])[col].fillna(method='ffill')

## 参照Stambaugh个人主页的 “Online Appendix for “Size and Value in China”，构建变量
df_tradefactor['Size'] = df_tradefactor['Dsmvosd'] * 1000 ## 流通市值
df_tradefactor['EP'] = df_tradefactor['经常性净利润'] / (df_tradefactor['Dsmvtll'] * 1000) ## 这里除以总市值
df_tradefactor['BM'] = df_tradefactor['股东权益合计(不含少数股东权益)'] / (df_tradefactor['Dsmvtll'] * 1000) 
df_tradefactor['CP'] =  (df_tradefactor['期末现金及现金等价物余额'] - df_tradefactor['期初现金及现金等价物余额'])  / (df_tradefactor['Dsmvtll'] * 1000)
## 过去20个交易日收益率的标准差，至少要有15个交易日
df_vol1 = df_tradefactor.pivot_table(index='Trddt', columns='Stkcd', values='Dretwd').rolling(window=20, closed='left', min_periods=15).std().unstack().reset_index().rename(columns={0:'Volatility'})
## 过去20个交易日收益率的最大值，至少要有15个交易日
df_vol2 = df_tradefactor.pivot_table(index='Trddt', columns='Stkcd', values='Dretwd').rolling(window=20, closed='left', min_periods=15).max().unstack().reset_index().rename(columns={0:'VolatilityMax'})
df_tradefactor['Illiquidity'] = df_tradefactor['Dretwd'].abs() / (df_tradefactor['Dnvaltrd'] / 1000000)
## 过去250个交易日的日均换手率，至少要有120个交易日
df_tradefactor['Turnover'] = df_tradefactor['Dnvaltrd'] / (df_tradefactor['Dsmvtll'] * 1000 / df_tradefactor['Clsprc'])
df_turn = df_tradefactor.pivot_table(index='Trddt', columns='Stkcd', values='Dretwd').rolling(window=250, min_periods=120).mean().shift(20).unstack().reset_index().rename(columns={0:'AbnormalTurnover'})
## 过去20个交易日的累积收益率
df_reversal = df_tradefactor.pivot_table(index='Trddt', columns='Stkcd', values='Dretwd').rolling(window=20, closed='left', min_periods=15).apply(lambda x:(1+x).prod()).unstack().reset_index().rename(columns={0:'Reversal'})

df_tradefactor = reduce(lambda left, right:pd.merge(left, right, on=['Stkcd', 'Trddt'], how='left'), [df_tradefactor, df_vol1, df_vol2, df_turn, df_reversal])
df_tradefactor = df_tradefactor[df_trade['Year']>=2007].reset_index(drop=True)
df_tradefactor.to_parquet("Replicate_Size_and_Value_in_China20250418.pq", index=False)
