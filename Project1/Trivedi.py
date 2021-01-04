# -*- coding: utf-8 -*-
"""Trivedi.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1UrXt37MSNmXxlZ1SsPCbhCSdN2uZtPSf
"""

import pandas as pd

cgm_data = pd.read_csv("CGMData.csv")
cgm_df = cgm_data[['Date' ,'Time', 'Sensor Glucose (mg/dL)']]
insulin_data = pd.read_csv("InsulinData.csv")
insulin_df = insulin_data[['Date', 'Time', 'Alarm']]

reversed_cgmData = cgm_df.iloc[: :-1]
reversed_insulinData = insulin_df.iloc[: :-1]
reversed_cgmData['DateTime'] = pd.to_datetime(reversed_cgmData['Date']+' '+ reversed_cgmData['Time'])
Date = reversed_insulinData.loc[reversed_insulinData['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']

Auto_Date = Date['Date'].iloc[0]
Auto_Time = Date['Time'].iloc[0]
Auto_Start = pd.to_datetime(Auto_Date+' '+Auto_Time)

cgm_manual_df = reversed_cgmData.loc[reversed_cgmData['DateTime']<Auto_Start]
cgm_auto_df = reversed_cgmData.loc[reversed_cgmData['DateTime']>=Auto_Start]
cgm_manual_df['Sensor Glucose (mg/dL)'].fillna(method='ffill', inplace=True)
cgm_auto_df['Sensor Glucose (mg/dL)'].fillna(method='ffill', inplace=True)

def MetricVal(data, days):
  total = 288*days
  hyperglycemia = (data[data['Sensor Glucose (mg/dL)']>180].count()/total)*100
  hyperglycemia_critical = (data[data['Sensor Glucose (mg/dL)']>250].count()/total)*100
  lowRange = (data[(data['Sensor Glucose (mg/dL)']>=70)&(data['Sensor Glucose (mg/dL)']<=180)].count()/total)*100
  secondaryRange = (data[(data['Sensor Glucose (mg/dL)']>=70)&(data['Sensor Glucose (mg/dL)']<=150)].count()/total)*100
  hypoglycemiaLevel1 = (data[data['Sensor Glucose (mg/dL)']<70].count()/total)*100
  hypoglycemiaLevel2 = (data[data['Sensor Glucose (mg/dL)']<54].count()/total)*100
  return {'hyperglycemia':hyperglycemia, 'hyperglycemia_critical':hyperglycemia_critical, 'lowRange':lowRange, 'secondaryRange':secondaryRange, 'hypoglycemiaLevel1':hypoglycemiaLevel1, 'hypoglycemiaLevel2':hypoglycemiaLevel2}

mask = (pd.to_timedelta(cgm_manual_df['Time'].astype(str))
          .between(pd.Timedelta('00:00:00'), pd.Timedelta('06:00:00')))
manualNight = cgm_manual_df[mask]
manualDay = cgm_manual_df[~mask]

manualLength = len(cgm_manual_df['DateTime'].dt.normalize().unique())
fullDayManual = MetricVal(cgm_manual_df, manualLength)
nightManual = MetricVal(manualNight, manualLength)
dayManual = MetricVal(manualDay, manualLength)

mask = (pd.to_timedelta(cgm_auto_df['Time'].astype(str))
          .between(pd.Timedelta('00:00:00'), pd.Timedelta('06:00:00')))
autoNight = cgm_auto_df[mask]
autoDay = cgm_auto_df[~mask]

autoLength = len(cgm_auto_df['DateTime'].dt.normalize().unique())
fullDayAuto = MetricVal(cgm_auto_df, autoLength)
nightAuto = MetricVal(autoNight, autoLength)
dayAuto = MetricVal(autoDay, autoLength)

result = {'Column': [' ','Manual Mode','Auto Mode'],
          'a': ['Percentage time in hyperglycemia (CGM > 180 mg/dL)',nightManual['hyperglycemia']['Date'],nightAuto['hyperglycemia']['Date']],
          'b':['percentage of time in hyperglycemia critical (CGM > 250 mg/dL)', nightManual['hyperglycemia_critical']['Date'],nightAuto['hyperglycemia_critical']['Date']],
          'c':['percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)', nightManual['lowRange']['Date'],nightAuto['lowRange']['Date']],
          'd':['percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)', nightManual['secondaryRange']['Date'],nightAuto['secondaryRange']['Date']],
          'e':['percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)', nightManual['hypoglycemiaLevel1']['Date'],nightAuto['hypoglycemiaLevel1']['Date']],
          'f':['percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)', nightManual['hypoglycemiaLevel2']['Date'],nightAuto['hypoglycemiaLevel2']['Date']],
          'g': ['Percentage time in hyperglycemia (CGM > 180 mg/dL)',dayManual['hyperglycemia']['Date'],dayAuto['hyperglycemia']['Date']],
          'h':['percentage of time in hyperglycemia critical (CGM > 250 mg/dL)', dayManual['hyperglycemia_critical']['Date'],dayAuto['hyperglycemia_critical']['Date']],
          'i':['percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)', dayManual['lowRange']['Date'],dayAuto['lowRange']['Date']],
          'j':['percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)', dayManual['secondaryRange']['Date'],dayAuto['secondaryRange']['Date']],
          'k':['percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)', dayManual['hypoglycemiaLevel1']['Date'],dayAuto['hypoglycemiaLevel1']['Date']],
          'l':['percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)', dayManual['hypoglycemiaLevel2']['Date'],dayAuto['hypoglycemiaLevel2']['Date']],
          'm': ['Percentage time in hyperglycemia (CGM > 180 mg/dL)',fullDayManual['hyperglycemia']['Date'],fullDayAuto['hyperglycemia']['Date']],
          'n':['percentage of time in hyperglycemia critical (CGM > 250 mg/dL)', fullDayManual['hyperglycemia_critical']['Date'],fullDayAuto['hyperglycemia_critical']['Date']],
          'o':['percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)', fullDayManual['lowRange']['Date'],fullDayAuto['lowRange']['Date']],
          'p':['percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)', fullDayManual['secondaryRange']['Date'],fullDayAuto['secondaryRange']['Date']],
          'q':['percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)', fullDayManual['hypoglycemiaLevel1']['Date'],fullDayAuto['hypoglycemiaLevel1']['Date']],
          'r':['percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)', fullDayManual['hypoglycemiaLevel2']['Date'],fullDayAuto['hypoglycemiaLevel2']['Date']]
        }
df = pd.DataFrame(result, columns= ['Column','a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r'])

df.to_csv (r'Trivedi_Results.csv', index = False, header=False)