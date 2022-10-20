import xarray as xr
"""
This script combine individual .nc files for ensemble members
Ensemble members refer to p01, p02, p03, p04
For control member c00, refer to the other script
"""
# reading one FULL single varible folder
dsmerged1 = xr.open_mfdataset('./reforecast_v3_ens_apcp_pwat/pwat_eatm_*_p01.nc',concat_dim='time',combine='nested')
dsmerged2 = xr.open_mfdataset('./reforecast_v3_ens_apcp_pwat/pwat_eatm_*_p02.nc',concat_dim='time',combine='nested')
dsmerged3 = xr.open_mfdataset('./reforecast_v3_ens_apcp_pwat/pwat_eatm_*_p03.nc',concat_dim='time',combine='nested')
dsmerged4 = xr.open_mfdataset('./reforecast_v3_ens_apcp_pwat/pwat_eatm_*_p04.nc',concat_dim='time',combine='nested')
# dsmerged = xr.open_mfdataset('apcp_sfc_*_c00.nc')
# dsmerged = xr.open_mfdataset('C:\\Users\\bobby\\Desktop\\.vscode\\1 UROP Research\\UROP v2\\preprocessing\\reforecast_v3\\apcp_sfc_*_c00.nc')

# remove empty dimensions
def combine_reforecast_p(dsmerged, dirpath):
    dsmerged = dsmerged.drop_vars('number') 

    # combine into 1 time
    dsmerged = dsmerged.assign_coords(valid_time = dsmerged.time + dsmerged.step) # combine into 1 time
    stacked_ds_2 = dsmerged.stack(datetime=("time", "step"))
    ds_merged = (stacked_ds_2.drop_vars("datetime").rename_dims({"datetime": "time"}).rename_vars({"valid_time": "time"}))

    # Slicing through every 6 hours
    ds_merged = ds_merged.isel(time = slice(1,-1, 2))

    # Reorder to time, lat, lon
    ds_merged = ds_merged.transpose("time", "latitude", "longitude")
    # ds_merged

    ds_merged.to_netcdf(path=dirpath)

combine_reforecast_p(dsmerged1, 'GEFSv12-Reforecast_pwat_p01.nc')
combine_reforecast_p(dsmerged2, 'GEFSv12-Reforecast_pwat_p02.nc')
combine_reforecast_p(dsmerged3, 'GEFSv12-Reforecast_pwat_p03.nc')
combine_reforecast_p(dsmerged4, 'GEFSv12-Reforecast_pwat_p04.nc')


print('combined reforecast done')