"""
Simulation with high resolution Stokes (1.5km) and no additional tidal forcing (standard case)
"""


from parcels import FieldSet, ParticleSet, JITParticle, ErrorCode, Field, VectorField, Variable
import numpy as np
from datetime import timedelta, datetime
import xarray as xr
from parcels.tools.converters import Geographic, GeographicPolar 
import math
import parcels.rng as ParcelsRandom
import time
import os

file_coast = 'Datafiles//datafile_coastMask_297x_375y'
coastMask  = np.genfromtxt(file_coast, delimiter=None)

startdate = '2020-01-01'
runtime_days = 730
#value for K based on Neumann (2014)
K = 13.39

current_data = xr.open_mfdataset('/data/oceanparcels/input_data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/*.nc')
lons = current_data.coords['longitude'].values
lats = current_data.coords['latitude'].values
fieldMesh_x,fieldMesh_y = np.meshgrid(lons,lats)

day_start = datetime(2020,1,1,12,00)

startplace = 'Domburg'
startlon = 3.493
startlat = 51.566
#fw = -1 means a backward simulation
fw = -1

outfile = str(startplace + '_'+ startdate + '_')

homedir = '/scratch/kaand004/BayesianAnalysis/'

#find nearest coastal cell to defined beaching location, to release in water        
def nearestcoastcell(lon,lat):
    dist = np.sqrt((fieldMesh_x - lon)**2 * coastMask + (fieldMesh_y - lat)**2 * coastMask)
    dist[dist == 0] = 'nan'
    coords = np.where(dist == np.nanmin(dist))
    startlon_release = fieldMesh_x[coords]
    endlon_release = fieldMesh_x[coords[0], coords[1] + 1]
    startlat_release = fieldMesh_y[coords]
    endlat_release = fieldMesh_y[coords[0] + 1, coords[1]]
    return startlon_release, endlon_release, startlat_release, endlat_release, coords

startlon_release, endlon_release, startlat_release, endlat_release, coords = nearestcoastcell(startlon,startlat)
#10x10 particles -> 100 particles homogeneously spread over grid cell
re_lons = np.linspace(startlon_release, endlon_release, 10)
re_lats = np.linspace(startlat_release, endlat_release, 10)
fieldMesh_x_re, fieldMesh_y_re = np.meshgrid(re_lons, re_lats)
#%%
variables_surface = {'U': 'uo',
             'V': 'vo'}
dimensions_surface = {'lat': 'latitude',
              'lon': 'longitude',
              'time': 'time'}

variables_stokes = {'U_Stokes': 'VSDX',
             'V_Stokes': 'VSDY'}
dimensions_stokes = {'lat': 'latitude',
              'lon': 'longitude',
              'time': 'time'}

fieldset = FieldSet.from_netcdf("/data/oceanparcels/input_data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/201*.nc", variables_surface, dimensions_surface, allow_time_extrapolation=True)

fieldset_Stokes = FieldSet.from_netcdf("/data/oceanparcels/input_data/CMEMS/NWSHELF_REANALYSIS_WAV/*.nc", variables_stokes, dimensions_stokes, allow_time_extrapolation=True)
fieldset_Stokes.U_Stokes.units = GeographicPolar()
fieldset_Stokes.V_Stokes.units = Geographic()

fieldset.add_field(fieldset_Stokes.U_Stokes)
fieldset.add_field(fieldset_Stokes.V_Stokes)

vectorField_Stokes = VectorField('UV_Stokes',fieldset.U_Stokes,fieldset.V_Stokes)
fieldset.add_vector_field(vectorField_Stokes)

class PlasticParticle(JITParticle):
    age = Variable('age', dtype=np.float32, initial=0., to_write=True)
    # beached : 0 sea, 1 beached, 2 after non-beach dyn, 3 after beach dyn, 4 please unbeach, 5 out of bounds
    beached = Variable('beached',dtype=np.int32,initial=0., to_write=False)

#%%
#---------------unbeaching
file_landCurrent_U = 'Datafiles/datafile_landCurrentU_%ix_%iy' % (len(lons),len(lats))
file_landCurrent_V = 'Datafiles/datafile_landCurrentV_%ix_%iy' % (len(lons),len(lats))

landCurrent_U = np.loadtxt(file_landCurrent_U)
landCurrent_V = np.loadtxt(file_landCurrent_V)

U_land = Field('U_land',landCurrent_U,lon=lons,lat=lats,fieldtype='U',mesh='spherical')
V_land = Field('V_land',landCurrent_V,lon=lons,lat=lats,fieldtype='V',mesh='spherical')


fieldset.add_field(U_land)
fieldset.add_field(V_land)

vectorField_unbeach = VectorField('UV_unbeach',U_land,V_land)
fieldset.add_vector_field(vectorField_unbeach)
#-----------------misc fields

K_m = K*np.ones(fieldMesh_x.shape)
K_z = K*np.ones(fieldMesh_x.shape)


Kh_meridional = Field('Kh_meridional', K_m,lon=lons,lat=lats,mesh='spherical')
Kh_zonal = Field('Kh_zonal', K_z,lon=lons,lat=lats,mesh='spherical')

fieldset.add_field(Kh_meridional)
fieldset.add_field(Kh_zonal)

#freeze out of bounds particles at discrete (randomly chosen) location instead of deleting; to prevent problems when concatenating in analysis
#not an elegant solution, but it works
def OutOfBounds(particle, fieldset, time): 
    particle.lon = -17.43
    particle.lat = 62.65
    #if particle.beached = 5, it is not advected anymore by any kernel and is thus frozen
    particle.beached = 5

def AdvectionRK4(particle, fieldset, time):
    """Advection of particles using fourth-order Runge-Kutta integration.
    Function needs to be converted to Kernel object before execution"""
    if particle.beached != 5:
        (u1, v1) = fieldset.UV[particle]
        lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
        (u2, v2) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat1, lon1, particle]
        lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
        (u3, v3) = fieldset.UV[time + .5 * particle.dt, particle.depth, lat2, lon2, particle]
        lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
        (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
        particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
        particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    else:
        particle.lon += 0
        particle.lat += 0

    
def StokesUV(particle, fieldset, time):
    if particle.beached == 0:
            (u_uss, v_uss) = fieldset.UV_Stokes[time, particle.depth, particle.lat, particle.lon]
            particle.lon += u_uss * particle.dt
            particle.lat += v_uss * particle.dt
            particle.beached = 3

def DiffusionUniformKh(particle, fieldset, time):
    """Kernel for simple 2D diffusion where diffusivity (Kh) is assumed uniform.
    Assumes that fieldset has constant fields `Kh_zonal` and `Kh_meridional`.
    These can be added via e.g.
        fieldset.add_constant_field("Kh_zonal", kh_zonal, mesh=mesh)
        fieldset.add_constant_field("Kh_meridional", kh_meridional, mesh=mesh)
    where mesh is either 'flat' or 'spherical'
    This kernel assumes diffusivity gradients are zero and is therefore more efficient.
    Since the perturbation due to diffusion is in this case isotropic independent, this
    kernel contains no advection and can be used in combination with a seperate
    advection kernel.
    The Wiener increment `dW` is normally distributed with zero
    mean and a standard deviation of sqrt(dt).
    """
    if particle.beached != 5:
        # Wiener increment with zero mean and std of sqrt(dt)
        dWx = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
        dWy = ParcelsRandom.normalvariate(0, math.sqrt(math.fabs(particle.dt)))
    
        bx = math.sqrt(2 * fieldset.Kh_zonal[particle])
        by = math.sqrt(2 * fieldset.Kh_meridional[particle])
    
        particle.lon += bx * dWx
        particle.lat += by * dWy
       
        particle.beached = 3 
    else:
        particle.lon += 0
        particle.lat += 0
            
def BeachTesting(particle, fieldset, time):
    if particle.beached == 2 or particle.beached == 3:
        (u, v) = fieldset.UV[time, particle.depth, particle.lat, particle.lon]
        if u == 0 and v == 0:
            if particle.beached == 2:
                particle.beached = 4
            else:
                particle.beached = 1
        else:
            particle.beached = 0
            
def UnBeaching(particle, fieldset, time):
    if particle.beached == 4 or particle.beached == 1:
        #-1 for backwards, otherwise the current is landward instead of seaward for unbeaching
        dtt = -1*particle.dt
        (u_land, v_land) = fieldset.UV_unbeach[time, particle.depth, particle.lat, particle.lon]
        particle.lon += u_land * dtt
        particle.lat += v_land * dtt
        particle.beached = 0
        
def Ageing(particle, fieldset, time):      
    particle.age += particle.dt   

#%%
#release particles for two years and advect for two years.
#releasing three times separately, to prevent having to advect once for 6 years resulting in long runtimes and large files with lots of redundant data
startdatelist = ['2020-01-01', '2018-01-01', '2016-01-01']

for i2 in range(2):
    
    for i in range(3):
        startdate_i = startdatelist[i]
        if i == 2:
            #the last release in 2016-01-01 is only releasing particles for one year. 
            #Since a particle released on 2015-01-01 is backtracked until 2013-01-01 (end of current data availability)
            runtime_releasedays = int(runtime_days/2)
        else: 
     
            runtime_releasedays = runtime_days
            
        pset = ParticleSet.from_list(fieldset=fieldset, pclass=PlasticParticle,
                                     time = np.datetime64(startdate_i),
                                     repeatdt=timedelta(hours=24).total_seconds(),
                                     lon = fieldMesh_x_re,
                                     lat = fieldMesh_y_re)
        
        filename_run1 = os.path.join(homedir,"{}.nc".format(outfile + 'r' + str(i+1) + '_run_' + str(i2)) )
        filename_run2 = os.path.join(homedir,"{}.nc".format(outfile + 'r' + str(i+1) + '_rerun_' + str(i2)) )
        
        output_file = pset.ParticleFile(name=filename_run1, outputdt=timedelta(hours=24))
          
         
        kernels = (pset.Kernel(AdvectionRK4) + pset.Kernel(StokesUV) + pset.Kernel(BeachTesting) + pset.Kernel(UnBeaching)
                    + pset.Kernel(DiffusionUniformKh)  +  pset.Kernel(Ageing)
                    + pset.Kernel(BeachTesting) + pset.Kernel(UnBeaching))       
        
        pset.execute(kernels,
                     #-1 because you repeat, first day is not taken into account. Prevents too many particles from being released.
                     runtime=timedelta(days=(runtime_releasedays - 1)),
                     dt=fw*timedelta(hours=2),
                     output_file=output_file,
                     recovery={ErrorCode.ErrorOutOfBounds: OutOfBounds})
        output_file.close()
        
        print('output written to %s' % filename_run1)
        print('Sleeping 30 secs...')
        time.sleep(30)
        
        #after particles have been released for two years, simulate these for two more years without releasing
        #to make sure that all particles have been advected for at least two years
        #redundant data (trajectories after more than 730 days) is deleted in post-processing
        pset = ParticleSet.from_particlefile(fieldset=fieldset, pclass=PlasticParticle,
                                              filename=filename_run1, restart=True, restarttime = np.nanmin)
        
        output_file2 = pset.ParticleFile(name=filename_run2, outputdt=timedelta(hours=24))
        
        pset.execute(kernels,
                     runtime=timedelta(days=runtime_days),
                     dt=fw*timedelta(hours=2),
                     output_file=output_file2,
                     recovery={ErrorCode.ErrorOutOfBounds: OutOfBounds})
        
        output_file2.close()
        print('output written to %s, ready for next date' % filename_run2)
        
