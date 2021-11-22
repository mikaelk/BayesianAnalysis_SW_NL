import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import timeit
import matplotlib.colors
import csv
import shapefile
import cartopy
import os

start = timeit.default_timer()


#load other data
# basepath = r'C:/Users/bramv/OneDrive - Universiteit Utrecht/UU/Jaar 3/BONZ/Datafiles/'
basepath = '/nethome/kaand004/BayesianAnalysis_SW_NL_v2/Datafiles/'
data_path = '/scratch/kaand004/BayesianAnalysis/'
suffix_results = '_tides2'

#load variables
x = np.load(data_path + 'x_tides.npy')
y = np.load(data_path + 'y_tides.npy')
t = (np.load(data_path + 't_tides.npy')).astype(int)
n_particles_day = 100

releasedays = len(x)//n_particles_day


file_landmask = basepath + '/datafile_landMask_297x_375y'
landdata = np.genfromtxt(file_landmask, delimiter=None)

file_coast = basepath + 'datafile_coastMask_297x_375y'
coastdata  = np.genfromtxt(file_coast, delimiter=None)

file_popmatrices = basepath + 'netcdf_populationInputMatrices_thres50_297x_375y.nc'
popmatrix = xr.open_mfdataset(file_popmatrices)
popmatrix_2020 = (popmatrix['pop_input'].values)[4,:,:]
c_prior = popmatrix_2020 * coastdata

# f_prior_week = np.load(basepath + 'fishingMatrix_week_20132020.npy')
nc_f_prior_week = xr.open_dataset(basepath + 'fishingMatrix_week.nc')
f_prior_week = nc_f_prior_week['Fishing activity'].values



#set fishing prior on land to zero (original data measures on some lakes, but are not in simulation)
for i in range(len(f_prior_week)):
    (f_prior_week[i])[landdata == 1] = 0
    
fisheryregions = np.load(basepath + 'fisheryregions.npy')
coastalregions = np.load(basepath + 'coastalregions.npy')

#load grid data
current_data = xr.open_dataset('/data/oceanparcels/input_data/CMEMS/NWSHELF_MULTIYEAR_PHY_004_009/201912.nc')
lons = current_data.coords['longitude'].values
lats = current_data.coords['latitude'].values
fieldMesh_x,fieldMesh_y = np.meshgrid(lons,lats)

xbins = np.linspace(-20, 13, 298)
ybins = np.linspace(40, 65, 376)  

#create colormap for plotting river data
num_colors = 9
cmap_r = plt.get_cmap('Greys', num_colors)
cmap_r2 = matplotlib.colors.ListedColormap(['white', 'black'])

stop = timeit.default_timer()
print('Time in cell loading variables: ', stop - start)  


#%% Load riverdata
start = timeit.default_timer()

file_river_prior = data_path + 'r_prior.npy'

if not os.path.exists(file_river_prior):
    def riverData():
        riverShapeFile    = basepath + 'Meijer2021_midpoint_emissions.shp'
        pollutionFile        = basepath + 'Meijer2021_midpoint_emissions.csv'
        dataArray_ID = 1 #column with yearly waste discharged by river

        sf = shapefile.Reader(riverShapeFile)

        #extract files within NorthSea
        plottingDomain = [-8.3, 5, 47, 57]

        rivers = {}
        rivers['longitude'] = np.array([])
        rivers['latitude'] = np.array([])
        rivers['ID'] = np.array([],dtype=int)
        rivers['dataArray'] = np.array([])

        for i1 in range(len(sf.shapes())):
            long = sf.shape(i1).points[0][0]
            lat = sf.shape(i1).points[0][1]

            if plottingDomain[0] < long <plottingDomain[1] and plottingDomain[2] < lat < plottingDomain[3]:
                rivers['longitude'] = np.append(rivers['longitude'],long)
                rivers['latitude'] = np.append(rivers['latitude'],lat)
                rivers['ID'] = np.append(rivers['ID'],i1)


        with open(pollutionFile, 'r',encoding='ascii') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            i1 = 0
            for row in filereader:
                if i1 > 0:
                    data_ID = i1-1 
                    if i1 == 1:
                        dataArray = [float(row[i2].replace(',','.')) for i2 in range(len(row))]
                        rivers['dataArray'] = dataArray
                    else:
                        if data_ID in rivers['ID']:
                            dataArray = [float(row[i2].replace(',','.')) for i2 in range(len(row))]
                            rivers['dataArray'] = np.vstack([rivers['dataArray'],dataArray])
                i1 += 1

        coastIndices = np.where(coastdata == 1)
        assert(np.shape(coastIndices)[0] == 2), "coastMask.data should be an array where the first dimension of the three is empty"

        # array containing indices of rivers not belonging to North Sea, which are to be deleted
        deleteEntries = np.array([],dtype=int)

        # matrix corresponding to fieldmesh, with per coastal cell the amount of river pollution
        riverInputMatrix = np.zeros(fieldMesh_x.shape)

        # for every river
        for i1 in range(len(rivers['longitude'])):   
            lon_river = rivers['longitude'][i1]
            lat_river = rivers['latitude'][i1]
            dist = 1e10
            # check which point is closest
            for i2 in range(np.shape(coastIndices)[1]):
                lon_coast = lons[coastIndices[1][i2]]
                lat_coast = lats[coastIndices[0][i2]]

                lat_dist = (lat_river - lat_coast) * 1.11e2
                lon_dist = (lon_river - lon_coast) * 1.11e2 * np.cos(lat_river * np.pi / 180)
                dist_tmp = np.sqrt(np.power(lon_dist, 2) + np.power(lat_dist, 2))

                # save closest distance
                if dist_tmp < dist:
                    dist = dist_tmp
                    lat_ID = coastIndices[0][i2]
                    lon_ID = coastIndices[1][i2]

            # if distance to closest point > threshold (3*approx cell length), delete entry
            if dist > 3*0.125*1.11e2:
                deleteEntries = np.append(deleteEntries,i1)
            # else: get pollution river, and add to releasematrix
            else:
                # add plastic input as obtained from the dataset
                riverInputMatrix[lat_ID,lon_ID] += rivers['dataArray'][i1,dataArray_ID]

        return riverInputMatrix

    r_prior = riverData()
    np.save(file_river_prior,r_prior)
else:
    r_prior = np.load(file_river_prior)
    
stop = timeit.default_timer()
print('Time in cell loading river data: ', stop - start)  


#%% Thicken landborder, for plotting
def thickenCoast(coastalprobs, thickness):
    def getLandBorder(landMask,lon,lat,val_add): 
        n_lat = landMask.shape[0]
        n_lon = landMask.shape[1]
            
        for i1 in range(n_lat):
            for i2 in range(n_lon):
                
                check_bot = True
                check_top = True
                check_left = True
                check_right = True
                
                # check whether land is located at boundary
                if i1 == 0:
                    check_top = False
                if i1 == n_lat-1:
                    check_bot = False
                if i2 == 0:
                    check_left = False
                if i2 == n_lon-1:
                    check_right = False
                    
                # check whether cell is land, if so look for coast
                if landMask[i1,i2] == 1:
                    
                    if check_top:
                        if (landMask[i1-1,i2] == 0) or (landMask[i1-1,i2] >= 2):
                            landMask[i1,i2] = -1
                    if check_bot:
                        if (landMask[i1+1,i2] == 0) or (landMask[i1+1,i2] >= 2):
                            landMask[i1,i2] = -1
                    if check_left:
                        if (landMask[i1,i2-1] == 0) or (landMask[i1,i2-1] >= 2):
                            landMask[i1,i2] = -1
                    if check_right:
                        if (landMask[i1,i2+1] == 0) or (landMask[i1,i2+1] >= 2):
                            landMask[i1,i2] = -1
        landMask[landMask == -1] = val_add         
        return landMask
    
    landMask = landdata.copy()
    coastMask = coastdata.copy()
    
    landBorder = landMask.copy()
    val_add = 2
    for i1 in range(thickness):
        landBorder = getLandBorder(landBorder,lons,lats,val_add)
        val_add += 1
    
    def closest_index(lat,lon,mask_test):
        distMat = 1e5 * np.ones(fieldMesh_x.shape)
        
        test_indices = np.where(mask_test == 1)
        
        distMat_lon = (lon - fieldMesh_x[test_indices[0],test_indices[1]])*1.11e2*0.63 #find distances coastal element w.r.t. ocean cells. 0.63 comes from lat=51deg (Zeeland)
        distMat_lat = (lat - fieldMesh_y[test_indices[0],test_indices[1]])*1.11e2
    
        distMat[test_indices[0],test_indices[1]] = np.sqrt(np.power(distMat_lon, 2) + np.power(distMat_lat, 2))    
        
        return np.where(distMat == distMat.min())[0][0],np.where(distMat == distMat.min())[1][0]
    
    ### interpolate beaching to closest coastal cell
    hist_beaching_coast = np.zeros(fieldMesh_x.shape)
    for i1 in range(len(lats)):
        for i2 in range(len(lons)):
            
            if coastalprobs[i1,i2] > 0:
                
                i_lat,i_lon = closest_index(lats[i1],lons[i2],coastMask)
                hist_beaching_coast[i_lat,i_lon] +=coastalprobs[i1,i2]
    
     ### go through the landborder defined above with increased width
    hist_beaching_extended = np.zeros(fieldMesh_x.shape)
    indices_border = np.where(landBorder > 1)            
    for i1 in range(len(indices_border[0])):
        lon_ = lons[indices_border[1][i1]]
        lat_ = lats[indices_border[0][i1]]
        i_lat,i_lon = closest_index(lat_,lon_,coastMask)
        
        hist_beaching_extended[indices_border[0][i1],indices_border[1][i1]] += hist_beaching_coast[i_lat,i_lon]
    return hist_beaching_extended


#%% Apply Bayesian framework to study temporal variability in sources
start = timeit.default_timer()

def find_sourceprobs_temp(x, y, lon, lat):
    #fishery_posterior_cells_notnormalized; cells are the grid cells, later we aggregate for bar charts 
    f_post_cells_nn = np.empty((52,5,375,297))
    #normalized
    f_post_cells_n = np.empty((52,5,375,297))
    #coastal
    c_post_cells_nn = np.empty((52,5,375,297))
    c_post_cells_n = np.empty((52,5,375,297))
    #river
    r_post_cells_nn = np.empty((52,5,375,297))
    r_post_cells_n = np.empty((52,5,375,297))
    f_post_list_n = np.empty((5,52,5))
    f_post_list_n_temp = np.empty((5,52))
    c_post_list_n = np.empty((10,52,5))
    c_post_list_n_temp = np.empty((10,52))
    total_p_week = np.zeros(52)
    likelihood_week = np.empty((52,5,375,297))
    #first calculate fishery probabilities using time-dependent prior
    for i in range(len(x)//n_particles_day):
        #id of first particle released on that day: releaseday*100, since 100 particles are released per day
        id1 = i*n_particles_day
        #release date of particle (actually the beaching date)
        #-1 since we are using it as an index
        #add week 53 (index 52) releases to week 1 (index 0), as week 53 is not there in every year
        releaseweek = int(t[id1,0]) - 1
        if releaseweek == 52:
            releaseweek = 0
        releaseyear = i//365
        for j in range(len(x.T)):
            #weeks in real time
            week = int(t[id1,j]) - 1
            if week == 52:
                week = 0
            hist_day = np.histogram2d(x[n_particles_day*i:n_particles_day*(i+1),j], y[n_particles_day*i:n_particles_day*(i+1),j], bins=[lon, lat])[0]
            hist_day = hist_day.T
            #unnormalized posterior, hist_day is the likelihood of all particles released on day j
            #multiplied with the right week (time-depedent prior)
            f_post_cells_nn[releaseweek, releaseyear :, :] += hist_day * f_prior_week[week,:,:] 
    print('1/5')

    #coastal and river don't have time-dependent prior; so just calculate likelihood per release week
    #loop over particles
    for i in range(len(x)):
        releaseweek = int(t[i,0]) - 1
        if releaseweek == 52:
            releaseweek = 0
        releaseyear = i//(365*n_particles_day)
        hist_particle = np.histogram2d(x[i,:], y[i,:], bins=[lon, lat])[0]
        hist_particle = hist_particle.T
        likelihood_week[releaseweek, releaseyear,:,:] += hist_particle
    
    print('2/5')
    for k in range(52):
        for l in range(5):
            #multiply with prior, likelihood depends on release week, prior is time-independent for coastal and river
            c_post_cells_nn[k,l,:,:] = likelihood_week[k,l,:,:] * c_prior
            r_post_cells_nn[k,l,:,:] = likelihood_week[k,l,:,:] * r_prior
            #normalize based on amount of fishing activity experienced by particles released in that week
            total_p_week[k] = np.sum(f_post_cells_nn[k,:,:,:])
    posterior_av = np.mean(total_p_week)
    posterior_rel = total_p_week / posterior_av
    print('3/5')

    #Normalize to 40% fishery avg over the year, 50% coastal and 10% river.
    for k in range(52):
        for l in range(5):
            f_post_cells_n[k, l,:,:] = 40*posterior_rel[k]*f_post_cells_nn[k, l,:,:]/np.nansum(f_post_cells_nn[k, l,:,:])
            r_post_cells_n[k, l,:,:] = (1/6)*(100-40*posterior_rel[k])*r_post_cells_nn[k, l,:,:]/np.nansum(r_post_cells_nn[k, l,:,:])
            c_post_cells_n[k, l,:,:] = (5/6)*(100-40*posterior_rel[k])*c_post_cells_nn[k, l,:,:]/np.nansum(c_post_cells_nn[k, l,:,:])
    #aggregate the grid cell probabilities per target region, for bar chart plotting
    print('4/5')
    for k in range(52):
        for l in range(5):
            for i in range(5):
                f_i = f_post_cells_n[k, l,:,:] * fisheryregions[:,:,i]
                f_post_list_n[i,k,l] = np.sum(f_i)
            for i in range(10):
                c_i = c_post_cells_n[k, l,:,:] * coastalregions[:,:,i]
                r_i = r_post_cells_n[k, l,:,:] * coastalregions[:,:,i]
                #sum river and coastal probabilities, since both use the same target regions
                cr_i = c_i + r_i
                c_post_list_n[i,k,l] = np.nansum(cr_i)
    print('5/5')
    
    #take average over the 5 years
    for k in range(52):
        for i in range(10):
            c_post_list_n_temp[i,k] = np.mean(c_post_list_n[i,k,:])  
        for i in range(5):
            f_post_list_n_temp[i,k] = np.mean(f_post_list_n[i,k,:])
            
    return f_post_list_n_temp, c_post_list_n_temp, f_post_cells_n, c_post_cells_n, r_post_cells_n

f_post_list_n_temp, c_post_list_n_temp, f_post_cells_n_temp, c_post_cells_n_temp, r_post_cells_n_temp = find_sourceprobs_temp(x,y,xbins,ybins)

stop = timeit.default_timer()
print('Time calculating temporal variability: ', stop - start)  


#%% Apply Bayesian framework to study influence of age assumption
start = timeit.default_timer()

def find_sourceprobs_age(x, y, lon, lat):
    f_post_cells_nn = np.empty((24, 375, 297))
    f_post_cells_n = np.empty((24, 375, 297))
    c_post_cells_nn = np.empty((24,375,297))
    c_post_cells_n = np.empty((24,375,297))
    r_post_cells_nn = np.empty((24,375,297))
    r_post_cells_n = np.empty((24,375,297))
    f_post_list_n = np.empty((5,24))
    c_post_list_n = np.empty((10,24))
    total_p_week = np.zeros(24)
    oob_pct = np.zeros(24)   
    start1 = timeit.default_timer()
    #again, first calculate fishery probabilities with time-dependent prior
    for i in range(releasedays):
        id1 = n_particles_day*i
        #-10 ugly solution, because 24*30 = 720, but I have 730 observations
        for j in range(len(x.T) - 10):
            #assuming 30 days per month
            age = j//30
            #-1 since you are using indices
            week = int(t[id1,j]) - 1
            if week == 52:
                week = 0
            hist_day = np.histogram2d(x[n_particles_day*i:n_particles_day*(i+1),j], y[n_particles_day*i:n_particles_day*(i+1),j], bins=[lon, lat])[0]
            hist_day = hist_day.T
            #unnormalized posterior
            f_post_cells_nn[age, :, :] += hist_day * f_prior_week[week,:,:] 
    stop1 = timeit.default_timer()
    print('Time calculating fishery probabilities age: ', stop1 - start1)
    #calculate coastal probabilities with constant prior
    #loop over particle age [months]
    for k in range(24):
        #calculate likelihood per assumed particle age (months)
        likelihood = np.zeros((375,297))
        #only consider part of trajectory with right age
        for i in range(len(x)): 
            hist_particle= np.histogram2d(x[i,30*k:30*(k+1)], y[i,30*k:30*(k+1)], bins=[lon, lat])[0]
            hist_particle = hist_particle.T
            likelihood += hist_particle
        #multiply with prior
        c_post_cells_nn[k,:,:] = likelihood * c_prior
        r_post_cells_nn[k,:,:] = likelihood * r_prior
        
        #check how many particles are out of bounds, located at NaN, NaN (lon,lat)
        #weigh fishing activity experienced by particles with that, to compensate for out-of-bounds behavior
        oob_count = np.isnan(x[:,30*k:30*(k+1)]).sum()
        oob_pct[k] = ((oob_count/(30*len(x)))*100)         
        w = 100 - oob_pct[k]
        #total unnormalized probability for age k
        total_p_week[k] = np.nansum(f_post_cells_nn[k,:,:]) / w
    posterior_av = np.mean(total_p_week)
    posterior_rel = total_p_week / posterior_av
    #again, to normalize to 40% avg over the year
    for k in range(24):
        f_post_cells_n[k] = 40*posterior_rel[k]*f_post_cells_nn[k]/np.nansum(f_post_cells_nn[k])
        r_post_cells_n[k] = (1/6)*(100-40*posterior_rel[k])*r_post_cells_nn[k]/np.nansum(r_post_cells_nn[k])
        c_post_cells_n[k] = (5/6)*(100-40*posterior_rel[k])*c_post_cells_nn[k]/np.nansum(c_post_cells_nn[k])
        for i in range(5):
            f_i = f_post_cells_n[k] * fisheryregions[:,:,i]
            f_post_list_n[i,k] = np.sum(f_i)
        for i in range(10):
            c_i = c_post_cells_n[k] * coastalregions[:,:,i]
            r_i = r_post_cells_n[k] * coastalregions[:,:,i]
            cr_i = c_i + r_i
            c_post_list_n[i,k] = np.nansum(cr_i)
    return f_post_list_n, c_post_list_n, f_post_cells_n, r_post_cells_n, c_post_cells_n

f_post_list_n_age, c_post_list_n_age, f_post_cells_n_age, r_post_cells_n_age, c_post_cells_n_age = find_sourceprobs_age(x,y,xbins,ybins)

stop = timeit.default_timer()
print('Time calculating age variability: ', stop - start)


#%%also calculate source probabilities without making any assumption about age, and averaging over all release dates
#most general view
start = timeit.default_timer()

def find_sourceprobs_avg(x, y, lon, lat):
    f_post_cells_nn = np.empty((375, 297))
    f_post_cells_n = np.empty((375, 297))
    c_post_cells_nn = np.empty((375,297))
    c_post_cells_n = np.empty((375,297))
    r_post_cells_nn = np.empty((375,297))
    r_post_cells_n = np.empty((375,297))
    #again, first calculate fishery probabilities with time-dependent prior
    for i in range(releasedays):
        id1 = n_particles_day*i
        for j in range(len(x.T)):
            week = int(t[id1,j]) - 1
            if week == 52:
                week = 0
            hist_day = np.histogram2d(x[n_particles_day*i:n_particles_day*(i+1),j], y[n_particles_day*i:n_particles_day*(i+1),j], bins=[lon, lat])[0]
            hist_day = hist_day.T
            #unnormalized posterior
            f_post_cells_nn += hist_day * f_prior_week[week,:,:] 
    #calculate probabilities for coastal and river, with constant priors
    likelihood = np.zeros((375,297))
    for i in range(len(x)): 
        hist_particle= np.histogram2d(x[i,:], y[i,:], bins=[lon, lat])[0]
        hist_particle = hist_particle.T
        likelihood += hist_particle
    #multiply with prior
    c_post_cells_nn = likelihood * c_prior
    r_post_cells_nn = likelihood * r_prior
    
    #normalize
    f_post_cells_n = 40*f_post_cells_nn/np.sum(f_post_cells_nn)
    r_post_cells_n = 10*r_post_cells_nn/np.sum(r_post_cells_nn)
    c_post_cells_n = 50*c_post_cells_nn/np.sum(c_post_cells_nn)
    return f_post_cells_n, r_post_cells_n, c_post_cells_n

f_post_cells_n_avg, r_post_cells_n_avg, c_post_cells_n_avg = find_sourceprobs_avg(x,y,xbins,ybins) 

stop = timeit.default_timer()
print('Time calculating avg sources: ', stop - start)


# f_post_list_n_temp, c_post_list_n_temp
# f_post_list_n_age, c_post_list_n_age
# f_post_cells_n_avg, r_post_cells_n_avg, c_post_cells_n_avg
import pickle 

def to_pickle(item,filename):
    outfile = open(filename,'wb')
    pickle.dump(item,outfile)
    outfile.close()   
    
  
def from_pickle(filename):
    with open(filename, 'rb') as f:
        item = pickle.load(f)
    return item


file_results = data_path + 'results' + suffix_results + '.pickle'

dict_results = {}
dict_results['f_post_list_n_temp'] = f_post_list_n_temp
dict_results['c_post_list_n_temp'] = c_post_list_n_temp
dict_results['f_post_list_n_age'] = f_post_list_n_age
dict_results['c_post_list_n_age'] = c_post_list_n_age
dict_results['f_post_cells_n_avg'] = f_post_cells_n_avg
dict_results['r_post_cells_n_avg'] = r_post_cells_n_avg
dict_results['c_post_cells_n_avg'] = c_post_cells_n_avg

to_pickle(dict_results,file_results)
