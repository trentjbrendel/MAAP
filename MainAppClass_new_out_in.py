# dr, dz is the tilt point. Tilt wrt y-axis which go through (dr, dz)
# Scanning range should be thought later.

import bz2
import csv
import os
import sys
import time
from itertools import product
from os.path import isfile
from shutil import copyfileobj, rmtree

import imageio
import matplotlib.pyplot as plt
import numpy as np
import win32com.client  # COM interface
import xlsxwriter
from PIL import Image
from scipy.interpolate import griddata
from scipy.optimize import least_squares

from basic import cart2pol, rotate
from import_seq import import_seq
from sag_theory import (asphere, asphere_xy, zernike, zernike_only,
                        zernike_polynomials_xy)


class MainApplication:
    # initiate    
    def __init__(self, **kwargs):
        # Field names to save
        self.fieldnames = ["Surface", \
                            "Zone", \
                            "Tan Factor", \
                            "Rad Factor", \
                            "dr(mm)", \
                            "d_theta", \
                            "dc (Deg)", \
                            "Num Rot", \
                            "dz(um)", \
                            "PV(um)", \
                            "RMSE(um)", \
                            "NA", \
                            "Time (sec)", \
                            "Total Time (sec)", \
                            "Fringe", \
                            "Slope_quiver", \
                            "Slope_angle", \
                            "Region (Orig. Coor.)"]
        
        self.seq = kwargs["seq"]        # Design file        
        #self.ih = ih # image height
        
        self.savfile_final = kwargs["savfile_final"]
        try:
            os.mkdir("D:/Programming/MAAP_Results")
        except:
            pass
        self.savfile_final = "D:/Programming/MAAP_Results/" + self.savfile_final
        
        self.fov = kwargs["fov"]
        self.sampling = kwargs["sampling"]            

        self.delta_x = self.fov/(self.sampling-1)
        self.delta_y = self.fov/(self.sampling-1)        

        # Aperture Size
        self.mav_list = kwargs["mav_list"]        

        # Objective NA
        self.ob_NA = kwargs["ob_NA"]

        # Run Code V Server
        if isfile(self.seq):
            try:
                self.CV_start()            
                self.cvserver.Command('in ' + self.seq) # Import lens design file
                self.cvserver.Command('in cv_macro:setvig')                
            except:
                pass
        else:
            print("Wrong file path.")
            sys.exit()     

    # Start Code V
    def CV_start(self):
        try:
            self.cvserver = win32com.client.Dispatch("CODEV.Command") # Run Code V Server     
            self.cvserver.SetStartingDirectory("c:/CVUSER") # Starting foler of CV
            self.cvserver.StartCodeV() # Run Code V
            self.cvserver.Command("in defaults.seq") # Initiate CV setting
            self.cvserver.Command("pth seq cv_macro:   C:/CVUSER/Macro")               
        except:
            print("You can't use Code V for now.")        

    # Stop Code V
    def CV_stop(self):
        self.cvserver.StopCodeV()

    # Tilt point cloud data (input tilt : degrees)
    def tilt_only(self, tilt, x, y, z):
        tilt_rad = tilt*np.pi/180 # Angle to Radian
        
        # Rotation (right-handed coordinate  -> the order of rotaion is x y z)
        [z,x] = np.dot(rotate(tilt_rad), [z,x])

        return x, y, z

    # Sag interpolation from x, y, z
    def sag_interpolation(self, x_new, y_new, z_new, x_center):
        t0_interpol = time.time()
        x_c = x_center
        y_c = 0
        x_datai = np.linspace(x_c-self.fov/2, x_c+self.fov/2, self.sampling)
        y_datai = np.linspace(y_c-self.fov/2, y_c+self.fov/2, self.sampling)                
        Z_sag = griddata((x_new, y_new), z_new, (x_datai[None,:], y_datai[:,None]), method='cubic') # method='cubic, linear, nearest'
        
        # interpolation time
        t_interpol = (time.time() - t0_interpol)
        print("--- Interpolation Time : %d min %d sec ---" %((t_interpol/60), (t_interpol%60)))
        
        X, Y = np.meshgrid(x_datai, y_datai)
        
        return X, Y, Z_sag

    # Sag interpolation for 0 degree tilt
    def sag_interpolation_zero_deg(self, CV_Coeff, x_center):
        t0_interpol = time.time()
        # If no tilt, no need to interpolation
        x = np.linspace(x_center-self.fov/2, x_center+self.fov/2, self.sampling, endpoint=True) # mm
        y = np.linspace(-self.fov/2, self.fov/2, self.sampling, endpoint=True) # mm
        X, Y = np.meshgrid(x, y)
        Z_sag = asphere_xy(CV_Coeff, X, Y)

        # interpolation time
        t_interpol = (time.time() - t0_interpol)
        print("--- Interpolation Time : %d min %d sec ---" %((t_interpol/60), (t_interpol%60)))
        
        return X, Y, Z_sag

    # Calculate slope from X, Y, Z (meshgrid data)
    def slope_cal(self, X, Y, Z_sag):
        t0_slope = time.time()
        
        # slope
        slope_x = np.zeros((self.sampling, self.sampling))
        slope_y = np.zeros((self.sampling, self.sampling))
        
        # 4 edges
        for i in [0, self.sampling-1]:
            j = 0
            if np.isnan(Z_sag[i,j+1]) or np.isnan(Z_sag[i,j]):
                slope_x[i,j] = np.nan
            else:
                slope_x[i,j] = (Z_sag[i,j+1]-Z_sag[i,j])/self.delta_x
            j = self.sampling-1
            if np.isnan(Z_sag[i,j]) or np.isnan(Z_sag[i,j-1]):
                slope_x[i,j] = np.nan
            else:
                slope_x[i,j] = (Z_sag[i,j]-Z_sag[i,j-1])/self.delta_x

        for j in [0, self.sampling-1]:
            i=0
            if np.isnan(Z_sag[i+1,j]) or np.isnan(Z_sag[i,j]):
                slope_y[i,j] = np.nan 
            else:
                slope_y[i,j] = (Z_sag[i+1,j]-Z_sag[i,j])/self.delta_y
            i = self.sampling-1
            if np.isnan(Z_sag[i,j]) or np.isnan(Z_sag[i-1,j]):
                slope_y[i,j] = np.nan 
            else:
                slope_y[i,j] = (Z_sag[i,j]-Z_sag[i-1,j])/self.delta_y

        # y = 0 ie j=0
        for i in range(1, self.sampling-1): # y-direction
            j=0
            if np.isnan(Z_sag[i,j+1]) or np.isnan(Z_sag[i,j]):
                slope_x[i,j] = np.nan
            else:
                slope_x[i,j] = (Z_sag[i,j+1]-Z_sag[i,j])/self.delta_x
            if np.isnan(Z_sag[i+1,j]) or np.isnan(Z_sag[i-1,j]):
                slope_y[i,j] = np.nan 
            else:
                slope_y[i,j] = (Z_sag[i+1,j]-Z_sag[i-1,j])/(2*self.delta_y)
        
        # x = 0 ie i=0
        for j in range(1, self.sampling-1): # y-direction
            i=0
            if np.isnan(Z_sag[i,j+1]) or np.isnan(Z_sag[i,j-1]):
                slope_x[i,j] = np.nan
            else:
                slope_x[i,j] = (Z_sag[i,j+1]-Z_sag[i,j-1])/(2*self.delta_x)
            if np.isnan(Z_sag[i+1,j]) or np.isnan(Z_sag[i,j]):
                slope_y[i,j] = np.nan 
            else:
                slope_y[i,j] = (Z_sag[i+1,j]-Z_sag[i,j])/self.delta_y

        # y = fov ie j=sampling
        for i in range(1, self.sampling-1): # y-direction
            j = self.sampling-1
            if np.isnan(Z_sag[i,j]) or np.isnan(Z_sag[i,j-1]):
                slope_x[i,j] = np.nan
            else:
                slope_x[i,j] = (Z_sag[i,j]-Z_sag[i,j-1])/self.delta_x
            if np.isnan(Z_sag[i+1,j]) or np.isnan(Z_sag[i-1,j]):
                slope_y[i,j] = np.nan 
            else:
                slope_y[i,j] = (Z_sag[i+1,j]-Z_sag[i-1,j])/(2*self.delta_y)
        
        # x = fov ie i=sampling
        for j in range(1, self.sampling-1): # y-direction
            i = self.sampling-1
            if np.isnan(Z_sag[i,j+1]) or np.isnan(Z_sag[i,j-1]):
                slope_x[i,j] = np.nan
            else:
                slope_x[i,j] = (Z_sag[i,j+1]-Z_sag[i,j-1])/(2*self.delta_x)
            if np.isnan(Z_sag[i,j]) or np.isnan(Z_sag[i-1,j]):
                slope_y[i,j] = np.nan 
            else:
                slope_y[i,j] = (Z_sag[i,j]-Z_sag[i-1,j])/self.delta_y

        # rest
        for i in range(1, self.sampling-1): # y-direction
            for j in range(1, self.sampling-1): #x-direction
                if np.isnan(Z_sag[i,j+1]) or np.isnan(Z_sag[i,j-1]):
                    slope_x[i,j] = np.nan
                else:
                    slope_x[i,j] = (Z_sag[i,j+1]-Z_sag[i,j-1])/(2*self.delta_x)
                if np.isnan(Z_sag[i+1,j]) or np.isnan(Z_sag[i-1,j]):
                    slope_y[i,j] = np.nan 
                else:
                    slope_y[i,j] = (Z_sag[i+1,j]-Z_sag[i-1,j])/(2*self.delta_y)
        
        slope_x = np.where(np.isnan(slope_x)|np.isnan(slope_y), np.nan, slope_x)
        slope_y = np.where(np.isnan(slope_x)|np.isnan(slope_y), np.nan, slope_y)        
            
        slope_mag = np.sqrt(slope_x**2+slope_y**2)

        # slope calculation time
        t_slope = (time.time() - t0_slope)
        print("--- Slope Calculation Time : %d min %d sec ---" %((t_slope/60), (t_slope%60)))        

        return slope_x, slope_y, slope_mag

    # Save point cloud in csv and compress
    def save_xyz(self, x_new, y_new, z_new, savefilename):
        # save csv
        with open(savefilename + ".csv", 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(['x', 'y', 'z'])  
            writer.writerows(np.transpose([x_new,y_new,z_new]))

        # Compression (to reduce file size)
        with open(savefilename + ".csv", 'rb') as input:
            with bz2.BZ2File(savefilename + '.csv.bz2', 'wb', compresslevel=9) as output:
                copyfileobj(input, output)

        # Delete csv after compression
        os.remove(savefilename + '.csv')

    # Fringe for wavelength calcuation from Z
    def fringe_cal(self, Z_sag, wavelength):
        phase = 2*np.pi/wavelength * np.array(Z_sag)*1000*2
        interference = np.cos(phase)        
        return interference

    # Whitelight Fringe calcuation from Z
    def fringe_cal_white(self, Z_sag, wavelength_i, wavelength_f):
        ################# make whitelight fringe by scanning wavelength 10nm step ###############
        # interference = 0
        # for wavelength in np.arange(0.450, 0.650, 0.01):
        #     interference += self.fringe_cal(Z_sag + depth, wavelength)
        #########################################################################################
        
        vc = 3e8 # speed of light
        constant = 2*np.pi * Z_sag * 1e-3 * 2 / vc # length dimension : m
        freq_i = vc/(wavelength_i * 1e-6) # length dimension : m
        freq_f = vc/(wavelength_f * 1e-6) # length dimension : m
        interference = (1/freq_f-freq_i) * 1/constant * (np.sin(constant * freq_f) - np.sin(constant * freq_i)) # Integral of cos(phase) w.r.t frequency        
        return interference

    # Save interference data in csv and compress
    def save_interference(self, interference, savefilename):
        # Save fringe values
        with open(savefilename + "_interference.csv", 'w+') as f:
            writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerows(interference)

        # Compress (to reduce file size)
        with open(savefilename + "_interference.csv", 'rb') as input:
            with bz2.BZ2File(savefilename + "_interference.csv.bz2", 'wb', compresslevel=9) as output:
                copyfileobj(input, output)

        # Remove csv file
        os.remove(savefilename + '_interference.csv')        

    # Save interference image in grey scale png
    def save_interference_image(self, interference, savefilename):
        # Save tiff
        # Image.fromarray(interference).save(savefilename + '.tiff')
        # Save png
        min_val = np.nanmin(interference)
        max_val = np.nanmax(interference)
        norm = np.where(interference == np.nan, np.nan, (interference.astype(np.float)-min_val)*255.0 / (max_val-min_val))
        Image.fromarray(norm.astype(np.uint8)).save(savefilename + '.png')
        
    # Save quiver plot
    def save_quiverplot(self, X, Y, slope_x, slope_y, quiver_num, savefilename):
        idx = np.around(np.linspace(0, self.sampling-1, quiver_num)).astype(int)
        # idx_slope = np.around(np.linspace(0, self.sampling-2, quiver_num)).astype(int)
        
        plt.xlim(np.min(X)-0.1, np.max(X)+0.1)
        plt.ylim(np.min(Y)-0.1, np.max(Y)+0.1)
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.quiver(X[tuple(np.meshgrid(idx,idx))], Y[tuple(np.meshgrid(idx,idx))], slope_x[tuple(np.meshgrid(idx_slope,idx_slope))], slope_y[tuple(np.meshgrid(idx_slope,idx_slope))])
        plt.quiver(X[tuple(np.meshgrid(idx,idx))], Y[tuple(np.meshgrid(idx,idx))], slope_x[tuple(np.meshgrid(idx,idx))], slope_y[tuple(np.meshgrid(idx,idx))])
        plt.tight_layout()
        plt.savefig(savefilename + ".png")
        plt.close()

    # Save pcolormesh plot
    def save_pcolormeshplot(self, X, Y, data, savefilename):
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))
        plt.gca().set_aspect('equal', adjustable='box')
        # plt.pcolormesh(X[1:, 1:]-self.delta_x/2, Y[1:, 1:]-self.delta_y/2, data)
        plt.pcolormesh(X, Y, data)
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(savefilename + ".png")
        plt.close()

    # Make movie for whitelight fringe using Z
    def make_movie(self, Z_sag, savefilename):
        # Make movie WLI
        t0_movie = time.time()
        try:
            os.mkdir("PNGs")
        except:
            pass

        Z_sag -= np.nanmean(Z_sag)
        
        # gif = []
        writer = imageio.get_writer(savefilename + ".mp4", fps=10)
        # for depth in np.arange(np.nanmin(Z_sag)-0.001, np.nanmax(Z_sag)+0.001, 0.0001): # scanning resolution 0.1um
        for depth in np.linspace(np.nanmin(Z_sag)-0.001, np.nanmax(Z_sag)+0.001, 101, endpoint=True): # 101 frames
            interference = self.fringe_cal_white(Z_sag - depth, 0.450, 0.650)
            self.save_interference_image(interference, "PNGs/white_" + f"{depth:.4f}")
            # gif.append(imageio.imread("PNGs/" + savefilename + "_white_" + f"{depth:.4f}" + ".png"))
            writer.append_data(imageio.imread("PNGs/white_" + f"{depth:.4f}" + ".png"))
        # imageio.mimsave(savefilename + ".gif", gif, loop=1)
        writer.close()
        try:
            rmtree("PNGs")
        except:
            pass
        # Time
        t_movie = (time.time() - t0_movie)
        print("--- Making movie Time : %d min %d sec ---" %((t_movie/60), (t_movie%60)))

    # Save sub-aperture map
    def save_scan_region_image(self, x_orig_zone, y_orig_zone, n_rot, dc, num_data, x_orig, y_orig, savefilename):
        ################## test #####################
        x_test_zone_i, y_test_zone_i = x_orig_zone, y_orig_zone
        x_test_zone, y_test_zone = x_test_zone_i, y_test_zone_i
        for i in range(n_rot-1):
            x_test_zone_i, y_test_zone_i = np.dot(rotate(-dc), [x_test_zone_i, y_test_zone_i])
            x_test_zone = np.append(x_test_zone, x_test_zone_i)
            y_test_zone = np.append(y_test_zone, y_test_zone_i)
            
        num_data_test = np.append(num_data, len(x_test_zone))

        # Total data
        x_test = np.append(x_orig, x_test_zone)
        y_test = np.append(y_orig, y_test_zone)
        
        # colorlist = ['r', 'g', 'b'] 
        plt.figure(figsize=(8,8), dpi=80)
        plt.gca().set_aspect('equal', adjustable='box')
        index_i = 0
        for i in range(len(num_data_test)):
            index = np.random.choice(np.arange(index_i, index_i+int(num_data_test[i])), min(self.sampling**2, int(num_data_test[i])), replace=False)
            plt.scatter(x_test[index], y_test[index], s=1)
            index_i += int(num_data_test[i])
        plt.tight_layout()
        plt.savefig(savefilename + "_in_xy_total_rot" + str(n_rot) + ".png")
        plt.close()
        # plt.show()

    # Model for tilt value search (Assuming r-translation and dc(z-rotation) algorithm)
    def model(self, d_parameters, x_center, x_max_limit, CV_Coeff, mav):
        d_theta = d_parameters[0]*np.pi/180 # Angle to Radian
        # dz = np.array(d_parameters[1])/1000 # Decenter from um to mm

        # Change x_center
        if np.isnan(x_max_limit) or (x_max_limit > mav): # call function -> no need to change x_center
            pass
        else:
            dz_tmp = asphere_xy(CV_Coeff, x_center, 0)
            sag_max_limit = asphere_xy(CV_Coeff, x_max_limit, 0)-dz_tmp # mm
            x_max_limit_rot, _, _ = self.tilt_only(d_parameters[0], x_max_limit-x_center, 0, sag_max_limit)
            if x_max_limit_rot-self.fov/2 > 0:
                x_center += (x_max_limit_rot-self.fov/2)*np.cos(d_theta)
            else:
                pass

        if x_max_limit >= mav:
            x_center += 0.05
        else:
            pass
        
        # original_coordinate
        add_region = 1/np.cos(d_theta)+0.05
        # Since high sampling is not necessary for optimization and interpolation, self.sampling -> 201 
        x = np.linspace(x_center-add_region*self.fov/2, x_center+add_region*self.fov/2, 201, endpoint=True) # mm
        y = np.linspace(-self.fov/2, self.fov/2, 201, endpoint=True) # mm
        xy = [x,y]
        
        x = np.array([k[0] for k in list(product(*xy))]) # mm
        y = np.array([k[1] for k in list(product(*xy))]) # mm  
        
        rho = np.sqrt(x**2 + y**2)

        x = x[(rho < mav)]
        y = y[(rho < mav)]

        dz = asphere_xy(CV_Coeff, x_center, 0)
        sag = asphere_xy(CV_Coeff, x, y)-dz # mm
        
        x_new, y_new, z_new = self.tilt_only(d_parameters[0], x-x_center, y, sag)
        # self.tip_tilt_3dof(d_parameters, x, y, z) # 3-DOF (da, db, dz)
        # self.tip_tilt_6dof(d_parameters, x, y, z) # 6-DOF (da, db, dz)

        x_new += x_center
        z_new += dz

        return [x_new, y_new, z_new], x_center, dz

    # Search tilt angle for zone and calculate everything for zone
    def run_on_zone(self, surf, zone, x_center, x_max_limit, CV_Coeff, mav):
        print("----------------------------------")
        print ("Surface", surf, "Zone", zone)    
        print("----------------------------------")      
        
        d_parameters_i = [0]
        
        t0_fit = time.time()
        # Optimization 
        def fun(d_parameters, x_center, x_max_limit, CV_Coeff, mav):
            [x, y, z], _, _ = self.model(d_parameters, x_center, x_max_limit, CV_Coeff, mav)
            # PV_sag = max(z)-min(z)
            # return PV_sag
            # RMSE = np.sqrt(np.mean((z-np.mean(z))**2))
            # return RMSE
            x_y0 = x[np.where(y == 0)]
            z_y0 = z[np.where(y == 0)]
            return np.nanmax(np.abs((z_y0[1:-1] - z_y0[0:-2])/(x_y0[1:-1] - x_y0[0:-2])))

        #dparams_bound = ([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # bounds=dparams_bound, 
        map_result = least_squares(
            fun, d_parameters_i, jac='3-point', method='trf', \
            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', \
            f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, \
            jac_sparsity=None, max_nfev=None, verbose=0, args=(x_center, x_max_limit, CV_Coeff, mav), kwargs={}
            )  
        # fitting time
        t_fit = (time.time() - t0_fit)
        print("--- Fitting Time : %d min %d sec ---" %((t_fit/60), (t_fit%60)))
        
        # Result
        d_parameters = map_result.x
        # d_theta = np.array(d_parameters[0])*np.pi/180 # radians
        # dz = np.array(d_parameters[1])/1000 # mm
        
        # RMSE = map_result.fun[0] # mm

        print ("d_theta(deg) : ", d_parameters[0])
        # print ("RMSE(um) : ", f"{RMSE*1000:.4f}")
        
        [x_new, y_new, z_new], x_center, dz = self.model(d_parameters, x_center, x_max_limit, CV_Coeff, mav)
        print ("dz(um) : ", f"{dz*1000:.4f}")   # 아직 dz 값은 의미없음

        print(len(~np.isnan(z_new)))

        if len(~np.isnan(z_new)) == 0:
            break_bool = True
        else:
            break_bool = False
        
        ############ Plot and Save #################

        self.savefilename = self.savfile_final + "_surf_"+ str(surf) + "_zone_" + str(zone)

        # self.save_xyz(x_new, y_new, z_new, self.savefilename)

        # Interpolation -> sag, slope in tilted view
        if d_parameters[0] == 0:
            X, Y, Z_sag = self.sag_interpolation_zero_deg(CV_Coeff, x_center)
        else:
            X, Y, Z_sag = self.sag_interpolation(x_new, y_new, z_new, x_center)
            
        slope_x, slope_y, slope_mag = self.slope_cal(X, Y, Z_sag)

        if len(np.where(~np.isnan(Z_sag))[0]) == 0 or len(np.where(~np.isnan(slope_mag))[0]) == 0:
            break_bool = True
        else:
            break_bool = False

        PV = np.nanmax(Z_sag) - np.nanmin(Z_sag)
        print ("PV(um) : ", f"{PV*1000:.4f}") 
        RMSE = np.sqrt(np.nanmean((Z_sag-np.nanmean(Z_sag))**2))
        print ("RMSE(um) : ", f"{RMSE*1000:.4f}")
        # center of scanning = np.mean(Z_sag_tmp)

        slope_angle = np.arctan(slope_mag)
        NA = np.sin(slope_angle)
        NA_limit = np.where(NA > self.ob_NA, np.nan, NA)
        Z_sag[np.where(np.isnan(NA_limit))] = np.nan        
        
        x = X[np.where(~np.isnan(NA_limit))].flatten()
        y = Y[np.where(~np.isnan(NA_limit))].flatten()
        z = Z_sag[np.where(~np.isnan(NA_limit))].flatten()

        xmax_orig = self.tilt_only(-d_parameters[0], x[np.where(x==x.max())]-x_center, y[np.where(x==x.max())], z[np.where(x==x.max())]-dz)[0].max()
        xmax_orig += x_center
        
        if zone == 1 and xmax_orig < (mav*0.99):
            print("The measurable area is smaller than effective aperture. Do recursion.")
            x_max_limit += (x_max_limit - xmax_orig)
            return self.run_on_zone(surf,zone,x_center, x_max_limit, CV_Coeff, mav)
        else:
            # quiver plot of slope in tilted view                                                
            self.save_quiverplot(X, Y, slope_x, slope_y, 20, self.savefilename + "_slope_quiver" + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF))

            # pcolormesh plot of slope magnitude in tilted view                
            # self.save_pcolormeshplot(X, Y, slope_mag, self.savefilename + "_slope_mag" + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF))

            # pcolormesh plot of angle in tilted view
            self.save_pcolormeshplot(X, Y, slope_angle*180/np.pi, self.savefilename + "_slope_angle" + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF))

            # Calculate numerical aperture of objective for the zone
            self.save_pcolormeshplot(X, Y, NA_limit, self.savefilename + "_NA" + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF))

            # Interference plot
            red = 0.6328 # um
            interference_red = self.fringe_cal(Z_sag, red)
            # self.save_interference(interference_red, self.savefilename + "_red")
            self.save_interference_image(interference_red, self.savefilename + "_red" + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF))
            
            # Make movie WLI
            # self.make_movie(Z_sag, self.savefilename)   

            return x, y, z, x_center, dz, d_parameters, PV, RMSE, np.nanmax(NA), break_bool

    # Find sub-apertures for surface using run_on_zone function
    def run_on_surf(self, surf, **kwargs):
        # Scanning Factor - Tangential
        self.TF = kwargs["TF"]
        if self.TF < 0:
            self.TF = 0
            print("Tangential Scanning Factor is set to 0.")
        elif self.TF > 1:
            self.TF = 1
            print("Tangential Scanning Factor is set to 1.")
        else:
            pass

        # Scanning Factor - Radial
        self.RF = kwargs["RF"]
        if self.RF < 0:
            self.RF = 0
            print("Radial Scanning Factor is set to 0.")
        elif self.RF > 1:
            self.RF = 1
            print("Radial Scanning Factor is set to 1.")
        else:
            pass

        # csv for each surface
        with open(self.savfile_final + '_surf_' + str(surf) + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

        # Excel for each surface
        xlsx_row = 0
        workbook = xlsxwriter.Workbook(self.savfile_final + '_surf_' + str(surf) + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.xlsx')
        worksheet = workbook.add_worksheet()
        worksheet.set_column('O:O', 15)
        worksheet.set_column('P:P', 17)
        worksheet.set_column('Q:Q', 17)
        worksheet.set_column('R:R', 17)
        worksheet.write_row('A1', self.fieldnames)

        # mav 
        try:
            self.cvserver.Command('eva (map s1)') # GetMaxAperture is working after some Command (I don't know why) 
            mav = self.cvserver.GetMaxAperture(surf,1)
        except:
            mav = self.mav_list[surf]

        _, CV_Coeff, _, _, _, _, _ = import_seq(self.seq, surf, mav)
        # print(CV_Coeff)

        # mav += 0.05 # 50um add

        overlap = self.RF*self.fov/2
        
        zone = 0
        x_center = mav - self.fov/2
        x_max_limit = mav
        totaltime = 0
        # x_center = 0
        # x_max_limit = self.fov/2
        # num_zone_est = mav/self.fov*2
        
        x_orig, y_orig, z_orig = [],[],[]
        num_data = [] # Number of data for each zone 
        n_rot = []
        
        while x_max_limit-overlap >= -overlap/2:
            print(x_max_limit-overlap, -overlap/2)
            zone += 1
            t0 = time.time()
            x, y, z, x_center, dz, d_parameters, PV, RMSE, NA_max, break_bool = self.run_on_zone(surf,zone,x_center, x_max_limit, CV_Coeff, mav)
            if break_bool == True:
                break
            else:
                pass

            # Measurement region in original coordinate
            x_orig_zone, y_orig_zone, z_orig_zone = self.tilt_only(-d_parameters[0], x-x_center, y, z-dz)
            x_orig_zone += x_center
            z_orig_zone += dz
            print("x : ", np.nanmin(x_orig_zone), "~", np.nanmax(x_orig_zone))
            print("z : ", np.nanmin(z_orig_zone), "~", np.nanmax(z_orig_zone))
            
            # collect left edge of x data to set the x_limit of next loop
            # x_left_edge = x_orig_zone[0:-1:self.sampling]
            # z_left_edge = z_orig_zone[0:-1:self.sampling]
            
            # if len(x_left_edge[~np.isnan(z_left_edge)]) == 0:
            #     # break
            #     x_max_limit = x_center - self.fov/2*np.cos(-d_parameters[0]*np.pi/180) + overlap
            # else:
            #     x_max_limit = np.max(x_left_edge[~np.isnan(z_left_edge)]) + overlap

            x_max_limit = x_orig_zone.min() + overlap

            # # Remove nan
            # x_orig_zone = x_orig_zone[np.where(~np.isnan(z_orig_zone))]
            # y_orig_zone = y_orig_zone[np.where(~np.isnan(z_orig_zone))]
            # z_orig_zone = z_orig_zone[np.where(~np.isnan(z_orig_zone))]

            # If you want to use only Edge
            # k1 = np.array(np.where(y_orig_zone==self.fov/2)[0])
            # k2 = np.array(np.where(y_orig_zone==-self.fov/2)[0])
            # k3 = np.array(np.where(x_orig_zone[0:-1] > x_orig_zone[1:])[0])
            # k4 = k3+1
            # # print(k1,k2,k3,k4)
            # k = np.unique(np.concatenate((k1,k2,k3,k4),0))

            # x_orig_zone = x_orig_zone[k]
            # y_orig_zone = y_orig_zone[k]
            # z_orig_zone = z_orig_zone[k]

            # Save zone image in original coordinate 
            # plt.figure(figsize=(8,8), dpi=80)
            # plt.gca().set_aspect('equal', adjustable='box')
            # plt.scatter(x_orig_zone, y_orig_zone, s=1)
            # plt.tight_layout()
            # plt.savefig(self.savefilename + "_in_xy.png")
            # plt.close()

            # Rorations along z-axis
            ##################
            if x_center == 0:
                n_rot.append(1)
            else:
                n_rot.append(0)
                y_tmp = -self.fov/2-1
                while n_rot[zone-1] < 4 or y_tmp < (-self.fov/2-(self.fov/self.sampling)): # 최소 회전 수는 현재 3 (n_rot 조건에 따라 결정)
                    n_rot[zone-1] += 1
                    dc = 2*np.pi/n_rot[zone-1]
                    _, y_tmp = np.dot(rotate(-dc), [x_orig_zone[-1], y_orig_zone[-1]])

                    # self.save_scan_region_image(x_orig_zone, y_orig_zone, n_rot, dc, num_data, x_orig, y_orig, self.savefilename)
                    # img = Image.open(self.savefilename + "_in_xy_total_rot" + str(n_rot) + ".png")
                    # img.show()

                    # Check image and determine go/stop
                    # yes = {'yes','y', 'ye', ''}
                    # no = {'no','n'}
                    # choice = input("Continue ? [y/n] : ").lower()
                    # if choice in yes:
                    #     pass
                    # elif choice in no:
                    #     break
                    # else:
                    #     print("You didn't input 'yes' or 'no'. It will be considered 'yes'.")
                    #     pass
                    
                if round(n_rot[zone-1]*self.TF) == 0:
                    n_rot[zone-1] = 1
                else:
                    n_rot[zone-1] = int(round(n_rot[zone-1]*self.TF))
                    # 3의 배수로
                    ###### case 1 올림 ################
                    # if n_rot[zone-1]%3 == 0:
                    #     pass
                    # elif n_rot[zone-1]%3 == 1:
                    #     n_rot[zone-1] += 2
                    # else:
                    #     n_rot[zone-1] += 1
                    ###### case 2 내림 ################
                    # n_rot[zone-1] = (n_rot[zone-1]//3)*3
                    ###### case 3 반올림 ################
                    n_rot[zone-1] = (round(n_rot[zone-1]/3))*3
                    if n_rot[zone-1] == 0:
                        n_rot[zone-1] = 3

                # # 엣지쪽만 있는 경우
                # if zone != 1:
                #     n_rot[zone-1] = 3
            
            print("n_rot : ", n_rot)
            
            if n_rot[zone-1] > 1:
                dc = 2*np.pi/n_rot[zone-1]
            else:
                dc = 0
                
            x_orig_zone_list, y_orig_zone_list = [[]] * n_rot[zone-1], [[]] * n_rot[zone-1]
            z_orig_zone_list = z_orig_zone * n_rot[zone-1]

            for i in range(n_rot[zone-1]):
                x_orig_zone_list[i], y_orig_zone_list[i] = np.dot(rotate(-dc*i), [x_orig_zone, y_orig_zone])

            x_orig_zone = np.array(x_orig_zone_list).flatten()
            y_orig_zone = np.array(y_orig_zone_list).flatten()
            z_orig_zone = np.array(z_orig_zone_list).flatten()

            # Save total zone image in original coordinate 
            # plt.figure(figsize=(8,8), dpi=80)
            # plt.gca().set_aspect('equal', adjustable='box')
            # index = np.random.choice(np.arange(0,len(x_orig_zone)), min(self.sampling**2, len(x_orig_zone)), replace=False)
            # plt.scatter(x_orig_zone[index], y_orig_zone[index], s=1)
            # plt.tight_layout()
            # plt.savefig(self.savefilename + "_in_xy_zone_total_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + ".png")
            # plt.close()

            num_data.append(len(x_orig_zone))

            # Total data
            x_orig.append(x_orig_zone_list)
            y_orig.append(y_orig_zone_list)
            z_orig.append(z_orig_zone_list)
            
            # colorlist = ['r', 'g', 'b'] 
            plt.figure(figsize=(8,8), dpi=80)
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(-mav*1.1, mav*1.1)
            plt.ylim(-mav*1.1, mav*1.1)
            # index_i = 0
            for i in range(zone):
                for j in range(len(x_orig[i])):
                    # index = np.random.choice(np.arange(0, int(num_data[i]/len(x_orig[i]))), int(min(self.sampling**2, int(num_data[i]/len(x_orig[i])))/100), replace=False)
                    index = np.random.choice(np.arange(0, int(num_data[i]/len(x_orig[i]))), int(10000/mav**2), replace=False)
                    plt.scatter(x_orig[i][j][index], y_orig[i][j][index], s=1)
                # index_i += int(num_data[i])
            ang = np.linspace(0, 2*np.pi, 360)
            xcircle = np.zeros(360)
            ycircle = np.zeros(360)
            xcircle = mav*np.cos(ang)
            ycircle = mav*np.sin(ang)
            plt.scatter(xcircle, ycircle, s=1, c='k')
            plt.tight_layout()
            plt.savefig(self.savefilename + "_in_xy_total_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + ".png")
            plt.close()

            totaltime += 0.16*PV*1000*n_rot[zone-1] 
            # csv
            with open(self.savfile_final + '_surf_' + str(surf) + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.csv', 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow({"Surface" : surf, \
                                "Zone" : zone, \
                                "Tan Factor" : self.TF, \
                                "Rad Factor" : self.RF, \
                                "dr(mm)" : x_center, \
                                "d_theta" : d_parameters[0], \
                                "dc (Deg)" : dc*180/np.pi, \
                                "Num Rot" : n_rot[zone-1], \
                                "dz(um)" : dz*1000, \
                                "PV(um)" : PV*1000, \
                                "RMSE(um)" : RMSE*1000, \
                                "NA" : NA_max, \
                                "Time (sec)" : 0.16*PV*1000*n_rot[zone-1],\
                                "Total Time (sec)" : totaltime})

            # Write Excel file
            xlsx_row += 1                    
            worksheet.set_row(xlsx_row, 90)
            worksheet.write_row(xlsx_row, 0, [surf, zone, self.TF, self.RF, x_center, d_parameters[0], dc*180/np.pi, n_rot[zone-1], dz*1000, PV*1000, RMSE*1000, NA_max, 0.16*PV*1000*n_rot[zone-1], totaltime])
            worksheet.insert_image('O' + str(xlsx_row+1), self.savefilename + '_red_TF_' + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.png', {'x_scale': 0.1, 'y_scale': 0.1})
            worksheet.insert_image('P' + str(xlsx_row+1), self.savefilename + '_slope_quiver_TF_' + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.png', {'x_scale': 0.2, 'y_scale': 0.2})
            worksheet.insert_image('Q' + str(xlsx_row+1), self.savefilename + '_slope_angle_TF_' + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.png', {'x_scale': 0.2, 'y_scale': 0.2})
            worksheet.insert_image('R' + str(xlsx_row+1), self.savefilename + '_in_xy_total_TF_' + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + ".png", {'x_scale': 0.15, 'y_scale': 0.15})
            
            # Time
            t = (time.time() - t0)
            print("--- Time : %d min %d sec ---" %((t/60), (t%60)))

            # Conditions for the next loop
            # if x_max_limit-overlap > self.fov/2:
            if x_max_limit > self.fov/2:
                x_center -= self.fov
            # elif x_max_limit-overlap <= 0:
            # elif x_max_limit <= 0:
            elif x_max_limit-overlap < -overlap/2: # zone 내의 영역들 간에 충분한 overlap 형성
                break
            elif x_center == 0:
                break
            else:
                x_max_limit = self.fov/2
                x_center = 0

        # Total number of scan, time
        self.num_scans = sum(n_rot) # for test
        xlsx_row += 1
        worksheet.write(xlsx_row, 7, sum(n_rot))
        worksheet.write_formula(xlsx_row, 13, "=sum(N2:N" + str(xlsx_row) + ")")
        
        workbook.close()

        return x_orig, y_orig, z_orig
    
    # Run run_on_surf function from surf_i to surf_f        
    def run(self, **kwargs):
        self.surf_i = kwargs["surf_i"]
        self.surf_f = kwargs["surf_f"]
        self.surf_num = self.surf_f - self.surf_i + 1

        # Scanning Factor - Tangential
        self.TF = kwargs["TF"]
        if self.TF < 0:
            self.TF = 0
            print("Tangential Scanning Factor is set to 0.")
        elif self.TF > 1:
            self.TF = 1
            print("Tangential Scanning Factor is set to 1.")
        else:
            pass

        # Scanning Factor - Radial
        self.RF = kwargs["RF"]
        if self.RF < 0:
            self.RF = 0
            print("Radial Scanning Factor is set to 0.")
        elif self.RF > 1:
            self.RF = 1
            print("Radial Scanning Factor is set to 1.")
        else:
            pass

        # csv
        with open(self.savfile_final + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
        
        # Excel for Total
        xlsx_row_total = 0 
        workbook_total = xlsxwriter.Workbook(self.savfile_final + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.xlsx')
        worksheet_total = workbook_total.add_worksheet()
        worksheet_total.set_column('O:O', 15)
        worksheet_total.set_column('P:P', 17)
        worksheet_total.set_column('Q:Q', 17)
        worksheet_total.set_column('R:R', 17)
        worksheet_total.write_row('A1', self.fieldnames)
        
        for surf in range(self.surf_i, self.surf_f+1):
            self.run_on_surf(surf, TF = self.TF, RF=self.RF)

            # csv
            with open(self.savfile_final + '_surf_' + str(surf) + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.csv', 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if float(row['Surface']) == surf:
                        zone = int(row['Zone'])
                        x_center = float(row['dr(mm)'])
                        d_theta = float(row["d_theta"]) # d_parameters[0]
                        d_parameters = [d_theta]
                        dc = float(row["dc (Deg)"])/180*np.pi
                        n_rot = int(row["Num Rot"])
                        dz = float(row["dz(um)"])/1000
                        PV = float(row["PV(um)"])/1000
                        RMSE = float(row["RMSE(um)"])/1000
                        NA_max = float(row["NA"])
                        totaltime = float(row["Total Time (sec)"])

                        self.savefilename = self.savfile_final + "_surf_"+ str(surf) + "_zone_" + str(zone)

                        # total csv write
                        with open(self.savfile_final + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.csv', 'a', newline='') as f:
                            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                            writer.writerow({"Surface" : surf, \
                                            "Zone" : zone, \
                                            "Tan Factor" : self.TF, \
                                            "Rad Factor" : self.RF, \
                                            "dr(mm)" : x_center, \
                                            "d_theta" : d_parameters[0], \
                                            "dc (Deg)" : dc*180/np.pi, \
                                            "Num Rot" : n_rot, \
                                            "dz(um)" : dz*1000, \
                                            "PV(um)" : PV*1000, \
                                            "RMSE(um)" : RMSE*1000, \
                                            "NA" : NA_max, \
                                            "Time (sec)" : 0.16*PV*1000*n_rot,\
                                            "Total Time (sec)" : totaltime})

                        xlsx_row_total += 1                    
                        worksheet_total.set_row(xlsx_row_total, 90)
                        worksheet_total.write_row(xlsx_row_total, 0, [surf, zone, self.TF, self.RF, x_center, d_parameters[0], dc*180/np.pi, n_rot, dz, PV, RMSE, NA_max, 0.16*PV*1000*n_rot, totaltime])
                        worksheet_total.insert_image('O' + str(xlsx_row_total+1), self.savefilename + '_red_TF_' + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.png', {'x_scale': 0.1, 'y_scale': 0.1})
                        worksheet_total.insert_image('P' + str(xlsx_row_total+1), self.savefilename + '_slope_quiver_TF_' + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.png', {'x_scale': 0.2, 'y_scale': 0.2})
                        worksheet_total.insert_image('Q' + str(xlsx_row_total+1), self.savefilename + '_slope_angle_TF_' + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + '.png', {'x_scale': 0.2, 'y_scale': 0.2})
                        worksheet_total.insert_image('R' + str(xlsx_row_total+1), self.savefilename + '_in_xy_total_TF_' + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + ".png", {'x_scale': 0.15, 'y_scale': 0.15})
            
            # self.save_xyz(x_orig, y_orig, z_orig, self.savfile_final + "_surf_"+ str(surf) + "_xyz")

        workbook_total.close()

    # Run run_on_surf function for every TF/RF cases in list
    def run_on_surf_for_TF_RF_list(self, **kwargs):
        t0 = time.time()
        self.surf = kwargs["surf"]
        TF_list = kwargs["TF_list"]
        RF_list = kwargs["RF_list"]

        try:
            os.mkdir("Results")
        except:
            pass
        xlsx_row_test = 0 
        workbook_test = xlsxwriter.Workbook('Results/Run_TF_RF_list.xlsx')
        worksheet_test = workbook_test.add_worksheet()
        worksheet_test.set_column('B:G', 17)
        worksheet_test.write_row('B1', np.arange(0, 1.01, 0.2))

        for i in TF_list:
            xlsx_row_test += 2
            worksheet_test.set_row(xlsx_row_test-1, 90)
            # worksheet_test.write_row(xlsx_row_test-1, 0, [i])
            worksheet_test.merge_range(xlsx_row_test-1, 0, xlsx_row_test, 0, i)
            xlsx_col_test = 0 
            for j in RF_list:
                print("----------------------------------")
                print("TF", "{:.2f}".format(i), "RF", "{:.2f}".format(j))    
                print("----------------------------------")
                xlsx_col_test += 1
                self.run_on_surf(self.surf, TF = i, RF = j)
                worksheet_test.insert_image(xlsx_row_test-1, xlsx_col_test, self.savefilename + '_in_xy_total_TF_' + "{:.2f}".format(i) + "_RF_" + "{:.2f}".format(j) + ".png", {'x_scale': 0.15, 'y_scale': 0.15})
                worksheet_test.write(xlsx_row_test, xlsx_col_test, self.num_scans)
                try:
                    app2.CV_stop()
                except:
                    pass
                
        workbook_test.close()

        # 수행시간
        t = (time.time() - t0)
        print("--- Total time : %d min %d sec ---" %((t/60), (t%60)))
    
    # Convert asphere surface to zernike polynomials using symmetric terms
    def asphere_to_zernike_pol_fit_sym(self, **kwargs):
        self.surf = kwargs["surf"]
        num_term = kwargs["num_term"]

        # mav 
        try:
            self.cvserver.Command('eva (map s1)') # GetMaxAperture is working after some Command (I don't know why) 
            mav = self.cvserver.GetMaxAperture(self.surf,1)
        except:
            mav = self.mav_list[self.surf]

        _, CV_Coeff, _, _, _, _, _ = import_seq(self.seq, self.surf, mav)

        rho = np.linspace(0, mav, 1000)
        sag = asphere(CV_Coeff, rho)

        rho /= mav # normalize wrt mav

        # Model to fit. Only using symmetric terms of zernike polynomials.
        def model(x, rho, num_term):
            CV_Coeff_zer = np.zeros(num_term)
            C_i = 0
            for i in range(len(x)):
                C_i += i*4
                CV_Coeff_zer[C_i] = x[i]
            sag = 0
            for j in range(len(CV_Coeff_zer)):
                sag += CV_Coeff_zer[j] * zernike_polynomials_xy(j, rho, 0)
            return sag

        # residual (y = sag)
        def fun(x, u, num_term, y):
            return model(x, u, num_term) - y

        # Initial values
        x_i = np.zeros(10)
        C_i = 0
        num_x = 0
        for i in range(len(x_i)):
            C_i += i*4
            if C_i < num_term:
                num_x += 1
            else:
                break
        print(num_x)
        x_i = np.zeros(num_x)

        fit_result = least_squares(
            fun, x_i, jac='3-point', method='trf', \
            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', \
            f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, \
            jac_sparsity=None, max_nfev=None, verbose=0, args=(rho,num_term,sag), kwargs={}
            )  

        # Result
        C_i = 0
        CV_Coeff_zer = np.zeros(num_term)
        for i in range(len(fit_result.x)):
            C_i += i*4
            CV_Coeff_zer[C_i] = fit_result.x[i]

        # CV_Coeff_zer = fit_result.x

        diff = model(fit_result.x, rho, num_term) - sag
        maxerror = max(diff)
        minerror = min(diff)

        print("Max Error :", maxerror*1000, "um")
        print("Min Error :", minerror*1000, "um")
        print("PV :", (maxerror-minerror)*1000, "um")
        
        return CV_Coeff_zer, mav

    # Convert asphere surface to zernike polynomials using every terms
    def asphere_to_zernike_pol_fit(self, **kwargs):
        self.surf = kwargs["surf"]
        num_term = kwargs["num_term"]

        # mav 
        try:
            self.cvserver.Command('eva (map s1)') # GetMaxAperture is working after some Command (I don't know why) 
            mav = self.cvserver.GetMaxAperture(self.surf,1)
        except:
            mav = self.mav_list[self.surf]

        _, CV_Coeff, _, _, _, _, _ = import_seq(self.seq, self.surf, mav)

        x = np.linspace(0, mav, 100)
        y = np.linspace(0, mav, 100)
        xy = [x,y]
        
        x = np.array([k[0] for k in list(product(*xy))]) # mm
        y = np.array([k[1] for k in list(product(*xy))]) # mm  
        
        rho = np.sqrt(x**2 + y**2)

        x = x[(rho < mav)]
        y = y[(rho < mav)]
        rho = rho[(rho < mav)]

        # # Ramdom Sampling
        # index = np.random.choice(np.arange(0, len(x)), 10000, replace=False)
        # x = x[index]
        # y = y[index]
        # rho = rho[index]
        
        sag = asphere(CV_Coeff, rho)

        x /= mav
        y /= mav
        rho /= mav # normalize wrt mav        

        # Model to fit. Using all terms of zernike polynomials.
        def model(CV_Coeff_zer, x, y):
            sag = 0
            for j in range(len(CV_Coeff_zer)):
                sag += CV_Coeff_zer[j] * zernike_polynomials_xy(j, x, y)
            return sag

        # residual (y = sag)
        def fun(CV_Coeff_zer, x, y, sag):
            return model(CV_Coeff_zer, x, y) - sag

        # Initial values
        x_i = np.zeros(num_term)

        print("Optimization 1 starts")

        fit_result = least_squares(
            fun, x_i, jac='3-point', method='trf', \
            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', \
            f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, \
            jac_sparsity=None, max_nfev=None, verbose=0, args=(x, y, sag), kwargs={}
            )  

        print("Optimization 1 complete")

        # Result
        CV_Coeff_zer = fit_result.x

        # diff = model(fit_result.x, x, y) - sag
        # maxerror = max(diff)
        # minerror = min(diff)

        # print("Max Error :", maxerror*1000, "um")
        # print("Min Error :", minerror*1000, "um")
        # print("PV :", (maxerror-minerror)*1000, "um")
        # print(CV_Coeff_zer)

        return CV_Coeff_zer, mav

    # Reference zernike surface from converting asphere and adding noise 
    def Reference_zernike_for_surf(self, **kwargs):
        self.surf = kwargs["surf"]
        num_term_1 = kwargs["num_term_1"] # fitting 할 텀
        num_term_2 = kwargs["num_term_2"] # 비교할 텀
        if num_term_1 < num_term_2:
            num_term_1 = num_term_2
            print ("num_term_1 should be larger than num_term_2. num_term_1 is set to", num_term_2)

        ############### Reference for surf #########################
        Zrn_Coeff, mav = self.asphere_to_zernike_pol_fit_sym(surf = self.surf, num_term = num_term_2)
        Zrn_Coeff = np.concatenate([Zrn_Coeff, np.zeros(num_term_1-num_term_2)], axis=None)
        Zrn_Coeff_noise = np.random.rand(num_term_1)*1e-5

        # noise
        x = np.linspace(0, mav, 100)
        y = np.linspace(0, mav, 100)
        xy = [x,y]
        
        x = np.array([k[0] for k in list(product(*xy))]) # mm
        y = np.array([k[1] for k in list(product(*xy))]) # mm  
        
        rho = np.sqrt(x**2 + y**2)

        x = x[(rho < mav)]
        y = y[(rho < mav)]
        rho = rho[(rho < mav)]

        # PV noise cal
        sag_noise = 0
        for j in range(num_term_2, num_term_1):
            sag_noise += Zrn_Coeff_noise[j] * zernike_polynomials_xy(j, x/mav, y/mav) # normalize to mav
        
        maxnoise = max(sag_noise)
        minnoise = min(sag_noise)
        noise_pv = maxnoise-minnoise

        print("PV Noise :", noise_pv*1000, "um")

        Zrn_Coeff = Zrn_Coeff + Zrn_Coeff_noise

        # csv for each surface
        with open(self.savfile_final + '_Zer_ref_surf_' + str(self.surf) + "_noise_" + "{:.3f}".format(noise_pv*1000) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(["Zernike Term", "Coeff"])
            for i in range(len(Zrn_Coeff)):
                writer.writerow([i+1, Zrn_Coeff[i]])
            
        return Zrn_Coeff, mav

    # Zernike fit from point cloud data
    def zernike_fit_from_zernike_pol_point_cloud(self, **kwargs):
        self.surf = kwargs["surf"]
        self.TF = kwargs["TF"]
        self.RF = kwargs["RF"]
        Zrn_Coeff = kwargs["Zrn_Coeff"]
        mav = kwargs["mav"]
        num_term_1 = kwargs["num_term_1"] # number of terms to fit
        num_term_2 = kwargs["num_term_2"] # number of terms to compare
        if num_term_1 < num_term_2:
            num_term_1 = num_term_2
            print ("num_term_1 should be larger than num_term_2. num_term_1 is set to", num_term_2)
        sample = kwargs["samples"] # number of samples of point clould which will be used to fit
        
        ############### x, y for TF/RF #########################
        filename = self.savfile_final + '_surf_' + str(self.surf) + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + "_sampling_" + str(sample) + '_xy.csv'
        if isfile(filename):
            x, y = [], []
            # read xy from csv
            with open(filename, 'r') as csvfile:
                reader = list(csv.reader(csvfile))
                for row in reader:
                    x.append(float(row[0]))
                    y.append(float(row[1]))
            print("Calling x,y complete.")
        else:
            x, y, _ = self.run_on_surf(self.surf, TF=self.TF, RF=self.RF)
            x = np.concatenate(x, axis=None)
            y = np.concatenate(y, axis=None)
            print("Concatenate Complete.")

            # Ramdom Sampling
            index = np.random.choice(np.arange(0, len(x)), sample, replace=False)
            # When I used numpy array instead of list, the fitting error is extremely increased
            x = list(x[index])
            y = list(y[index])

            # xy for each case
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                writer.writerows(np.transpose([x, y]))

        ###################################################################

        sag = 0
        for j in range(len(Zrn_Coeff)):
            sag += Zrn_Coeff[j] * zernike_polynomials_xy(j, x/mav, y/mav) # normalize to mav
        
        # Optimization 
        def model(Zrn_Coeff, mav, x, y):
            # normalize wrt mav
            x /= mav
            y /= mav
            sag_fit = 0
            for j in range(len(Zrn_Coeff)):
                sag_fit += Zrn_Coeff[j] * zernike_polynomials_xy(j,x,y)
            return sag_fit

        def fun(Zrn_Coeff, mav, x, y, sag):
            res = model(Zrn_Coeff, mav, x, y) - sag
            return res
        
        num_coef = num_term_1
        Zrn_Coeff_i = np.zeros(num_coef)
        
        print("Optimization 2 starts.")
        #dparams_bound = ([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]) # bounds=dparams_bound, 
        opt_result = least_squares(
            fun, Zrn_Coeff_i, jac='3-point', method='trf', \
            ftol=1e-08, xtol=1e-08, gtol=1e-08, x_scale=1.0, loss='linear', \
            f_scale=1.0, diff_step=None, tr_solver=None, tr_options={}, \
            jac_sparsity=None, max_nfev=None, verbose=0, args=(mav, x, y, sag), kwargs={}
            )  

        print("Optimization 2 Complete.")

        Zrn_Coeff_fitted = opt_result.x

        sag_fit = 0
        for j in range(len(Zrn_Coeff_fitted)):
            sag_fit += Zrn_Coeff_fitted[j] * zernike_polynomials_xy(j, x/mav, y/mav) # normalize to mav
        
        diff = sag - sag_fit
        maxerror = max(diff)
        minerror = min(diff)

        print("Max Error :", maxerror*1000, "um")
        print("Min Error :", minerror*1000, "um")
        print("PV :", (maxerror-minerror)*1000, "um")

        return Zrn_Coeff_fitted

    # Compare zernike coefficients    
    def compare_zernike_coefficients(self, Zrn_Coeff, Zrn_Coeff_fitted, **kwargs):
        num_term_1 = kwargs["num_term_1"] # number of terms to fit
        num_term_2 = kwargs["num_term_2"] # number of terms to compare
        if num_term_1 < num_term_2:
            num_term_1 = num_term_2
            print ("num_term_1 should be larger than num_term_2. num_term_1 is set to", num_term_2)
        sample = kwargs["samples"] # number of samples of point clould which will be used to fit

        print("In, Fit, Diff")
        print(np.transpose([Zrn_Coeff, Zrn_Coeff_fitted, Zrn_Coeff-Zrn_Coeff_fitted]))
        
        self.RSS = [None]*num_term_1
        self.RSS[0] = sum((Zrn_Coeff[0:num_term_2]-Zrn_Coeff_fitted[0:num_term_2])**2)
        
        print("Residual Sum of Squares (1~66) :", self.RSS[0])

        # csv for each surface
        with open(self.savfile_final + '_Zer_fit_surf_' + str(self.surf) + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + "_sampling_" + str(sample) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(["Zernike Term", "In", "Fit", "Diff", "RSS(1~66)"])  
            writer.writerows(np.transpose([range(1, num_term_1+1), Zrn_Coeff, Zrn_Coeff_fitted, Zrn_Coeff-Zrn_Coeff_fitted, self.RSS]))
            # writer.writerow(["In", "Fit", "Diff"])  
            # writer.writerows(np.transpose([Zrn_Coeff, Zrn_Coeff_fitted, Zrn_Coeff-Zrn_Coeff_fitted]))
            # writer.writerow(["RSS", RSS])

    # Residual sum of squares of point cloud (resolution 10um)
    def PV_point_cloud(self, Zrn_Coeff, Zrn_Coeff_fitted, **kwargs):
        num_term_1 = kwargs["num_term_1"] # number of terms to fit
        num_term_2 = kwargs["num_term_2"] # number of terms to compare
        mav = kwargs["mav"]
        if num_term_1 < num_term_2:
            num_term_1 = num_term_2
            print ("num_term_1 should be larger than num_term_2. num_term_1 is set to", num_term_2)
        sample = kwargs["samples"] # number of samples of point clould which will be used to fit

        x = np.arange(-mav, mav+0.001, 0.01) # step ~ resolution
        y = np.arange(-mav, mav+0.001, 0.01) # step ~ resolution
        xy = [x,y]
        
        x = np.array([k[0] for k in list(product(*xy))]) # mm
        y = np.array([k[1] for k in list(product(*xy))]) # mm  
        
        rho = np.sqrt(x**2 + y**2)

        x = x[(rho < mav)]
        y = y[(rho < mav)]
        rho = rho[(rho < mav)]

        sag_in = 0
        for j in range(num_term_2):
            sag_in += Zrn_Coeff[j] * zernike_polynomials_xy(j, x/mav, y/mav) # normalize to mav

        sag_fit = 0
        for j in range(num_term_2):
            sag_fit += Zrn_Coeff_fitted[j] * zernike_polynomials_xy(j, x/mav, y/mav) # normalize to mav

        diff = sag_in-sag_fit

        self.RSS_pc, self.PV_val = [None]*num_term_1, [None]*num_term_1
        self.RSS_pc[0] = sum((sag_in-sag_fit)**2)
        self.PV_val[0] = diff.max() - diff.min()
        
        print("PV result :", self.PV_val[0], "mm")

        # csv for each surface
        with open(self.savfile_final + '_Zer_fit_surf_' + str(self.surf) + "_TF_" + "{:.2f}".format(self.TF) + "_RF_" + "{:.2f}".format(self.RF) + "_sampling_" + str(sample) + '.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            writer.writerow(["Zernike Term", "In", "Fit", "Diff", "RSS", "PV"])  
            writer.writerows(np.transpose([range(1, num_term_1+1), Zrn_Coeff, Zrn_Coeff_fitted, Zrn_Coeff-Zrn_Coeff_fitted, self.RSS_pc, self.PV_val]))
            # writer.writerow(["In", "Fit", "Diff"])  
            # writer.writerows(np.transpose([Zrn_Coeff, Zrn_Coeff_fitted, Zrn_Coeff-Zrn_Coeff_fitted]))
            # writer.writerow(["RSS", RSS])

    # Zernike fit from point cloud data for every TF/RF cases in list
    def zernike_fit_for_TF_RF_list(self, **kwargs):
        t0 = time.time()
        self.surf = kwargs["surf"]
        TF_list = kwargs["TF_list"]
        RF_list = kwargs["RF_list"]
        num_term_1 = kwargs["num_term_1"] # fitting 할 텀
        num_term_2 = kwargs["num_term_2"] # 비교할 텀

        Zrn_Coeff, mav = self.Reference_zernike_for_surf(surf = 11, num_term_1 = num_term_1, num_term_2 = num_term_2)

        try:
            os.mkdir("Results")
        except:
            pass
        xlsx_row_test = 0 
        workbook_test = xlsxwriter.Workbook('D:/Programming/MAAP_Results/Zernike_fit_TF_RF_list.xlsx')
        worksheet_test = workbook_test.add_worksheet()
        worksheet_test.set_column('B:G', 17)
        worksheet_test.write_row('B1', np.arange(0, 1.01, 0.2))
    
        for i in TF_list:
            self.TF = i
            xlsx_row_test += 2
            worksheet_test.set_row(xlsx_row_test-1, 90)
            # worksheet_test.write_row(xlsx_row_test-1, 0, [i])
            worksheet_test.merge_range(xlsx_row_test-1, 0, xlsx_row_test, 0, i)
            xlsx_col_test = 0 
            for j in RF_list:
                print("----------------------------------")
                print("TF", "{:.2f}".format(i), "RF", "{:.2f}".format(j))    
                print("----------------------------------")
                self.RF = j
                Zrn_Coeff_fitted = self.zernike_fit_from_zernike_pol_point_cloud(surf = self.surf, Zrn_Coeff = Zrn_Coeff, mav = mav, TF=i, RF=j, \
                                                                                num_term_1 = num_term_1, num_term_2 = num_term_2, samples=1000)
                # self.compare_zernike_coefficients(Zrn_Coeff, Zrn_Coeff_fitted, num_term_1 = num_term_1, num_term_2 = num_term_2, samples=1000)
                self.PV_point_cloud(Zrn_Coeff, Zrn_Coeff_fitted, num_term_1 = num_term_1, num_term_2 = num_term_2, mav=mav, samples=1000)
                
                xlsx_col_test += 1
                n = 6
                self.savefilename = "D:/Programming/MAAP_Results/MAAP_N2A_surf_11_zone_" + str(n)
                while True:
                    if isfile(self.savefilename + '_in_xy_total_TF_' + "{:.2f}".format(i) + "_RF_" + "{:.2f}".format(j) + ".png"):
                        break
                    else:
                        n -= 1
                        self.savefilename = "D:/Programming/MAAP_Results/MAAP_N2A_surf_11_zone_" + str(n)
                worksheet_test.insert_image(xlsx_row_test-1, xlsx_col_test, self.savefilename + '_in_xy_total_TF_' + "{:.2f}".format(i) + "_RF_" + "{:.2f}".format(j) + ".png", {'x_scale': 0.15, 'y_scale': 0.15})
                # worksheet_test.write(xlsx_row_test, xlsx_col_test, self.RSS[0])
                worksheet_test.write(xlsx_row_test, xlsx_col_test, self.PV_val[0])
                try:
                    app2.CV_stop()
                except:
                    pass
                
        workbook_test.close()
                
        # 수행시간
        t = (time.time() - t0)
        print("--- Total time : %d min %d sec ---" %((t/60), (t%60)))
