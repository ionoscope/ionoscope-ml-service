import pandas as pd

from scripts.restore_historical_features import read_drivers, station_grid


def test_each_driver_respects_its_own_period_and_other_indices_do_not_erase_daily_values(tmp_path):
    rows=[]
    for index, frequency in [('Kp','3h'),('ap','3h'),('Ap','1D'),('Fobs','1D'),('Fadj','1D')]:
        for stamp in pd.date_range('2025-08-30', '2025-09-02', freq=frequency, tz='UTC'):
            rows.append({'time_utc':stamp,'index':index,'value':4})
    pd.DataFrame(rows).to_csv(tmp_path/'geophysical_indices.csv',index=False)
    times=pd.date_range('2025-08-30','2025-09-02',freq='1h',tz='UTC')
    pd.DataFrame({'time_utc':times,'bz_gsm_nT':1,'dst_nT':-1,
                  'flow_speed_km_s':400,'proton_density_n_cc':5}).to_csv(tmp_path/'omni_solar_wind.csv',index=False)
    grid=pd.date_range('2025-08-30','2025-09-02',freq='15min',tz='UTC')
    result=read_drivers(tmp_path,grid)
    assert pd.isna(result.loc['2025-08-30T00:45Z','bz_gsm_nT'])
    assert result.loc['2025-08-30T01:00Z','bz_gsm_nT']==1
    assert pd.isna(result.loc['2025-08-30T02:45Z','gfz_Kp'])
    assert result.loc['2025-08-30T03:00Z','gfz_Kp']==4
    assert pd.isna(result.loc['2025-08-30T23:45Z','gfz_Ap'])
    assert result.loc['2025-08-31T18:00Z','gfz_Ap']==4


def test_low_and_unknown_cs_do_not_enter_any_station_measurements():
    times=pd.date_range('2025-09-01',periods=4,freq='15min',tz='UTC')
    raw=pd.DataFrame({'station':'TR169','time_utc':times,'CS':[70,20,-1,70],
                      'foF2':[5,99,99,6],'hmF2':250.,'TEC':20.})
    result=station_grid(raw,times,pd.DataFrame(index=times))
    assert result.foF2.notna().tolist()==[True,False,False,True]
    assert result.hmF2.notna().tolist()==[True,False,False,True]
