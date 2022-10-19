import xarray as xr
"""
This script combine individual .nc files for control member
Control member refer to c00
For perturbed member p##, refer to the other script
"""
# reading one FULL single varible folder
dsmerged = xr.open_mfdataset('./reforecast_v3_ens_apcp/apcp_sfc_*_p01.nc')
# dsmerged = xr.open_mfdataset('apcp_sfc_*_c00.nc')
# dsmerged = xr.open_mfdataset('C:\\Users\\bobby\\Desktop\\.vscode\\1 UROP Research\\UROP v2\\preprocessing\\reforecast_v3\\apcp_sfc_*_c00.nc')

# remove empty dimensions
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

ds_merged.to_netcdf(path='GEFSv12-Reforecast_pwateatm_c00.nc')

print('combined reforecast done')