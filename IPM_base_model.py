"""
IPM Base Model: Uses labels and and generates a pest risk map

:param labels_filename: file with labels data from an Athena query (/queries/labels_query.txt)
:param coverage_filename: file with coverage data from an Athena query (/queries/coverage_query.txt)
:param row_tagid_map_filename: file with row tagid data to create mapping (/queries/row_map_query.txt)
:param post_dist_map_filename: file with post distance data to create mapping (/queries/post_map_query.txt)
:param select_customer = flag to switch between customers data: 'LJ' (for Le Jardin), 'NF' (for Nature Fresh),
                         'LK' (for Lakeside)
:param greenhouse_param = {'num_rows': total # of rows, 'start_row': start row number, 'num_posts': total # posts,
                          'start_post': start post number, 'customer': customer name,
                          'left_rows': slice(start row left of GH , end row left of GH, rows interval),
                          'right_rows': slice(start row right of GH , end row right of GH, rows interval),
                          'row_interval': delta rows on each side of the GH}

:param confg_param = {'label': pest/disease name,
                      'base_model_transmission': [beta, gamma],
                      'pressure_max': max pressure,
                      'base_model_dates': [start date, end date] (these are the dates for which to calculate the model;
                                          start and end should be at least one day apart),
                      'rows_range': [start row, end row] (this is for the rows to calculate in the model),
                       'height_range': [min high, max high] (currently not used in the model)}

:param train_flags = {'load_label_data': True or False (loads csv data. Must be set to true once for each customer.
                                         Once loaded the data is stored in /data_and_models),
               'prep_xarray_label': True or False (prepares xarray for the selected pest/disease, confg_param['label'].
                                    Must be set to true for every pest/disease to be modeled),

               'load_coverage_data': True or False (loads csv data. Must be set to true once for each customer.
                                         Once loaded the data is stored in /data_and_models)
               'prep_xarray_coverage': True or False (prepares xarray for coverage),
               'calc_base_model': True or False (calculates base model for the selected pest/disease data,
                                   confg_param['label']),
               'test_base_model': True or False (test base model by generating a sequence of plots of model outputs and
                                  measured labels: all days plot, 5 days plot, last day plot)
               }

:param plot_flags = {'plot_labels_report': True or False (lists all pests/diseases observed, dates, and generates plots
                                        of all of them including progression of selected the selected pest/disease,
                                        confg_param['label']),
              'plot_coverage': True or False (plots accumulated coverage),
              }

author: Adrian Fuxman
email: adrian.fuxman@ecoation.com

"""

# Import Libraries
import os
import numpy as np
import pandas as pd
import datetime as dt
from datetime import timedelta
from functools import reduce

import pickle
import time
import copy

from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, FuncFormatter, MaxNLocator)
from matplotlib import gridspec
import matplotlib.ticker as ticker
import xarray as xr
from xrviz.dashboard import Dashboard

from config import labels_filename, coverage_filename, row_tagid_map_filename, post_dist_map_filename, \
    greenhouse_param, confg_param, train_flags, plot_flags


# defines the name of the directory to be created
folder_plots = "/home/eduardo/Documents/Reports/"+str(greenhouse_param['customer'])+"/"+str(confg_param['base_model_dates'][1])


# Create target directory & all intermediate directories if don't exists
if not os.path.exists(folder_plots):
    os.makedirs(folder_plots)
    print("Directory " , folder_plots ,  " Created ")
else:
    print("Directory " , folder_plots ,  " already exists")


# Define grid and initialize

if greenhouse_param['customer']=='OR':
    row_list = list(range(greenhouse_param['start_row_left'],
                               greenhouse_param['start_row_left'] + greenhouse_param['num_rows_left']))
elif np.logical_and(greenhouse_param['customer']=='JM', greenhouse_param['phase']==2):
    row_list=greenhouse_param['left_rows']+greenhouse_param['right_rows']

else:
    row_list = list(list(range(greenhouse_param['left_rows'].start,
                              greenhouse_param['left_rows'].step+greenhouse_param['left_rows'].stop,greenhouse_param['left_rows'].step))+list(range(greenhouse_param['right_rows'].start,
                              greenhouse_param['right_rows'].step+greenhouse_param['right_rows'].stop, greenhouse_param['right_rows'].step)))


row = np.tile(np.array([row_list]).transpose(), (1, greenhouse_param['num_posts']))

post_list = list(range(greenhouse_param['start_post'], greenhouse_param['start_post'] + greenhouse_param['num_posts']))

if greenhouse_param['customer']=='OR':
    post = np.tile(np.array([post_list]), (greenhouse_param['num_rows_left'], 1))
else:
    post = np.tile(np.array([post_list]), (greenhouse_param['num_rows_left']+greenhouse_param['num_rows_right'], 1))


###### Defining the number of ticks on the y axis for the plots
if np.logical_and(greenhouse_param['customer']=='JM',greenhouse_param['phase']==2) :
    multiple_locator=int(len(greenhouse_param['left_rows']+greenhouse_param['right_rows'])*0.025)
    row_values = greenhouse_param['left_rows']

elif greenhouse_param['row_interval']==2:
    multiple_locator = int((greenhouse_param['num_rows_left']+ greenhouse_param['num_rows_right']) * 0.025/2)
    row_values=list(range(greenhouse_param['left_rows'].start, greenhouse_param['left_rows'].stop+greenhouse_param['left_rows'].step,greenhouse_param['left_rows'].step))
else:
    multiple_locator = int((greenhouse_param['num_rows_left'] + greenhouse_param['num_rows_right']) * 0.025)
    row_values=list(range(greenhouse_param['left_rows'].start, greenhouse_param['left_rows'].stop+greenhouse_param['left_rows'].step,greenhouse_param['left_rows'].step))

# row_values=list(range(greenhouse_param['start_row_left'],
#                           greenhouse_param['start_row_left'] + greenhouse_param['num_rows_left']))


list2=[]


list2.append(0)
for i,r_number in enumerate(row_values):
    if i==0:
        list2.append(r_number)
    elif i % multiple_locator==0:
        list2.append(r_number)



list_minor=[]
for i, r_number in enumerate(list2):
    if i == 0:
        list_minor.append(r_number)
    elif i % 2 == 0:
        list_minor.append(r_number)

# Creates a grid with specific number of rows and posts to be used by pressure and risk plots
def set_grid(ax):
    ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.yaxis.set_major_locator(5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(multiple_locator))

    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    return ax

def minor_grid(ax):
    ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.yaxis.set_major_locator(5)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    # ax.yaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(multiple_locator*2))

    ax.grid(which='major', color='#CCCCCC', linestyle='--')
    ax.grid(which='minor', color='#CCCCCC', linestyle=':')

    return ax

    # Get label data that corresponds to the tomato count data
    label = []
    label_int = []

    # Get information from tomato count to map label
    for k in range(0, len(metadata['distance'])):
                distance = metadata['distance'][k]
                #tagid = metadata['tagid'][k]
                rsid = int(metadata['rsid'][k])
                direction = metadata['direction'][k]

                #indx_id = label_data[
                #    np.logical_and(label_data['tag_id'] == tagid, label_data['row_session_id'] == rsid)].index
                indx_id = label_data[label_data['row_session_id'] == rsid].index
                indx_dist = label_data[np.logical_and(label_data['min_distance_interval'] < distance,
                                                      label_data['max_distance_interval'] > distance)].index
                indx_side = label_data[label_data['sensor_direction'] == direction].index
                indx_comb = indx_id.intersection(indx_dist.intersection(indx_side))

                if len(indx_comb) == 0:
                    label.append('un-labelled')
                else:
                    label.append(label_data['label'][indx_comb[0]])  # TO BE CHECKED AND FIXED IF NEEDED

                if label[k] == 'healthy' or label[k] == 'un-labelled':
                    label_int.append(0)
                else:
                    label_int.append(1)

    return label_int, label


def get_label_for_xarray(location, labels_data, confg_param, label_unique_days):
    # location: (row, post)

    # Initialize:
    label_int = -1 * np.ones(label_unique_days.shape)
    pressure = -1 * np.ones(label_unique_days.shape)

    # Get indices for search:
    indx_healthy = np.where(labels_data['label'].isin(['healthy']))

    indx_row = np.where(labels_data['row_number'].isin([location[0]]))[0]
    indx_post = np.where(labels_data['post_number'].isin([location[1]]))[0]
    indx_label = np.where(labels_data['label'].isin([confg_param['label']]))
    #indx_height = labels_data[np.logical_and(labels_data['height_cm'] > confg_param['height_range'][0],
    #                                              labels_data['height_cm'] < confg_param['height_range'][1])].index

    indx_list = [indx_row, indx_post, indx_label]
    indx_loc_label = list(reduce(np.intersect1d, (indx_list)))

    indx_list = [indx_row, indx_post, indx_healthy]
    indx_loc_healthy = list(reduce(np.intersect1d, (indx_list)))

    # Get label and pressure:
    if len(indx_loc_label) == 0 and len(indx_loc_healthy) == 0: # No labels at this location, check for nearby locations
        label_int = -1 * np.ones(label_unique_days.shape)
        pressure = -1 * np.ones(label_unique_days.shape)
    else:
        indx_loc = indx_loc_label + indx_loc_healthy

        for k in range(0, len(label_unique_days)):
            indx_loc_subset = labels_data['capture_date'][indx_loc] == label_unique_days[k]
            if np.any(indx_loc_subset == True):
                indx = []
                if len(indx_loc_subset) == 1:
                    indx = indx_loc_subset.keys()
                else:
                    for key in indx_loc_subset.keys():
                        if indx_loc_subset[key] == True:
                            indx.append(key)

                if all(i in indx_loc_healthy for i in indx):
                    label_int[k] = 0
                elif all(i in indx_loc_label for i in indx):
                    label_int[k] = 1

                pressure[k] = labels_data['pressure'][indx].mean()

    return label_int, pressure


def get_coverage(location, labels_data):
    # location: (row, post)

    coverage_int = []

    indx_row = np.where(labels_data['row_number'].isin([location[0]]))[0]
    indx_post = np.where(labels_data['post_number'].isin([location[1]]))[0]
    indx_list = [indx_row, indx_post]
    indx_loc = list(reduce(np.intersect1d, indx_list))

    # Get coverage
    if len(indx_loc) == 0: # No coverage found at this location
        coverage_int = 0
    else:
        coverage_int = 1

    return coverage_int


def get_coverage_for_xarray(location, coverage_data, coverage_unique_days):
    # location: (row, post)

    coverage_int = np.nan * np.ones(coverage_unique_days.shape)

    indx_row = np.where(coverage_data['row_number'].isin([location[0]]))[0]
    indx_post = np.where(coverage_data['post_number'].isin([location[1]]))[0]
    indx_list = [indx_row, indx_post]
    indx_loc = list(reduce(np.intersect1d, indx_list))

    # Get coverage
    if len(indx_loc) == 0:  # No coverage found at this location
        coverage_int = np.nan * np.zeros(coverage_unique_days.shape)
    else:
        for k in range(0, len(coverage_unique_days)):
            indx_loc_subset = coverage_data['capture_date'][indx_loc] == coverage_unique_days[k]
            if np.any(indx_loc_subset == True):
                coverage_int[k] = 1

    return coverage_int


def base_model(labels_data_xarray, confg_param, date_list):

    calc_label_int = False  # This flag is used to enable the calculation of label (0, 1) based risk model in addition
                                # pressure based risk model.

    # Initialize:
    intermediate_xarray = labels_data_xarray.copy(deep=True)


    # Add NaNs for dates in date_list that are not in labels_data_xarray
    dates_wno_labels_list = []
    for date in date_list:
        if pd.to_datetime(date).asm8 not in labels_data_xarray.coords['time'].values:
            dates_wno_labels_list.append(pd.to_datetime(date).asm8)

    nan_array = np.zeros((post.shape[0], post.shape[1], len(dates_wno_labels_list))) * np.nan
    label_int_nan_xarray = xr.DataArray(nan_array, coords=[row_list, post_list, dates_wno_labels_list],
                                    dims=('row', 'post', 'time'),
                                    name='label_int')
    pressure_nan_xarray = xr.DataArray(nan_array, coords=[row_list, post_list, dates_wno_labels_list],
                                   dims=('row', 'post', 'time'),
                                   name='pressure')

    nan_xarray = xr.Dataset({'label_int': (['row', 'post', 'time'], label_int_nan_xarray),
                                     'pressure': (['row', 'post', 'time'], pressure_nan_xarray)})
    nan_xarray.coords['row'] = row_list
    nan_xarray.coords['post'] = post_list
    nan_xarray.coords['time'] = dates_wno_labels_list

    intermediate_xarray = xr.concat([intermediate_xarray, nan_xarray], dim='time')
    intermediate_xarray.sortby('time').coords['time'].values # Sort by time to get values in ascending order



    # Start clock
    start_time = time.time()

    for k, date in enumerate(date_list):

        print('Calculating model for day ' + str(date))

        # Initialize

        interm_pressure_array = np.zeros((post.shape[0], post.shape[1])) * np.nan
        interm_pressure_min_array = np.zeros((post.shape[0], post.shape[1])) * np.nan
        interm_pressure_mean_array = np.zeros((post.shape[0], post.shape[1])) * np.nan

        interm_label_int_array = np.zeros((post.shape[0], post.shape[1])) * np.nan
        interm_label_int_min_array = np.zeros((post.shape[0], post.shape[1])) * np.nan
        interm_label_int_mean_array = np.zeros((post.shape[0], post.shape[1])) * np.nan

        # Update step:
        if k > 0:
            # Update pressure
            array_a = copy.deepcopy(intermediate_xarray['pressure'].sel(time=date).values)
            array_b = copy.deepcopy(intermediate_xarray['pressure'].sel(time=date).values)
            array_b[~np.isnan(array_b)] = 1

            array_c = (np.nan_to_num((array_b * 0), nan=1) *
                  intermediate_xarray['pressure'].sel(time=date_list[k - 1]).values)

            array_update = np.nansum(np.dstack((array_a, array_c)), 2)
            array_update[np.logical_and(np.isnan(array_a), np.isnan(array_c))] = np.nan

            intermediate_xarray['pressure'].loc[dict(row=row_list, post=post_list, time=date)] = array_update



            # Update label_int
            if calc_label_int:
                array_a = copy.deepcopy(intermediate_xarray['label_int'].sel(time=date).values)
                array_b = copy.deepcopy(intermediate_xarray['label_int'].sel(time=date).values)
                array_b[~np.isnan(array_b)] = 1

                array_c = (np.nan_to_num((array_b * 0), nan=1) *
                      intermediate_xarray['label_int'].sel(time=date_list[k - 1]).values)

                array_update = np.nansum(np.dstack((array_a, array_c)), 2)
                array_update[np.logical_and(np.isnan(array_a), np.isnan(array_c))] = np.nan

                intermediate_xarray['label_int'].loc[dict(row=row_list, post=post_list, time=date)] = array_update

            print('Updating step completed for day ' + str(date))

        # Propagation step:
        print('Propagating step for day ' + str(date))

        # For every selected row
        if greenhouse_param['customer']=='OR':
            scope=greenhouse_param['num_rows_left']
        else:
            scope=greenhouse_param['num_rows_left']+greenhouse_param['num_rows_right']
        for i in range(confg_param['rows_range'][0] - confg_param['rows_range'][0],
                           scope,1):

            #print('Propagating step for row ' + str(i) + ' for date ' + str(date))


            # For every post
            for j in range(0, post.shape[1]):

                # Every row on the left or right of the greenhouse
                nearby_post = list(range(post[i, j] - confg_param['range'], post[i, j] + confg_param['range'] + 1, 1))

                nearby_row = list(range(row[i, j] - greenhouse_param['row_interval'] * confg_param['range'],
                                        row[i, j] + greenhouse_param['row_interval'] * confg_param['range'] + 1,
                                        greenhouse_param['row_interval']))



                if greenhouse_param['customer']=='OR':
                    a_max = greenhouse_param['num_rows_left']
                    a_min = greenhouse_param['start_row_left']
                elif np.logical_and(greenhouse_param['customer']=='JM', greenhouse_param['phase']==2):
                    if np.logical_and(row[i,j] in greenhouse_param['left_rows'], row[i,j]<=147):
                        a_max=147
                        a_min= greenhouse_param['start_row_left']
                    elif np.logical_and(np.logical_and(row[i,j] in greenhouse_param['left_rows'], row[i,j]>=301), row[i,j]<=350):
                        a_max = 350
                        a_min = 301
                    elif np.logical_and(np.logical_and(row[i,j] in greenhouse_param['left_rows'], row[i,j]>=501), row[i,j]<=550):
                        a_max = 550
                        a_min = 501
                    elif np.logical_and(np.logical_and(row[i,j] in greenhouse_param['right_rows'], row[i,j]>=204), row[i,j]<=250):
                        a_max = 250
                        a_min = 204
                    elif np.logical_and(np.logical_and(row[i,j] in greenhouse_param['right_rows'], row[i,j]>=401), row[i,j]<=450):
                        a_max = 450
                        a_min = 401
                    elif np.logical_and(np.logical_and(row[i,j] in greenhouse_param['right_rows'], row[i,j]>=601), row[i,j]<=650):
                        a_max = 650
                        a_min = 601
                #
                elif greenhouse_param['customer']=='GN':
                    if row[i,j] % 2 ==0:
                        a_max= greenhouse_param['left_rows'].stop
                        a_min = greenhouse_param['left_rows'].start
                    else:
                        a_max = greenhouse_param['right_rows'].stop
                        a_min = greenhouse_param['right_rows'].start

                else:
                    if i < greenhouse_param['num_rows_left']:
                        a_max = greenhouse_param['start_row_left']+greenhouse_param['num_rows_left']-1
                        a_min = greenhouse_param['start_row_left']
                    elif i >= greenhouse_param['num_rows_left']:
                        a_max = greenhouse_param['start_row_right'] + greenhouse_param['num_rows_right'] - 1
                        a_min = greenhouse_param['start_row_right']

                nearby_post = np.unique(np.clip(nearby_post, post_list[0], post_list[-1]))

                nearby_row = np.unique(np.clip(nearby_row, a_min, a_max))



            # Spatial averaging of nearby cells
                option = 2  # Option 1: Un-weighted average, Option 2: Weighted average with transmission coefficients

                # Get nearby pressures
                pressures_nearby = intermediate_xarray.pressure.sel(row=nearby_row, post=nearby_post, time=date).values
                interm_pressure_min_array[i, j] = np.nanmin(pressures_nearby)
                interm_pressure_mean_array[i, j] = np.nanmean(pressures_nearby)

                # Get nearby labels
                if calc_label_int:
                    label_int_nearby = intermediate_xarray.label_int.sel(row=nearby_row, post=nearby_post,
                                                                        time=date).values
                    interm_label_int_min_array[i, j] = np.nanmin(label_int_nearby)
                    interm_label_int_mean_array[i, j] = np.nanmean(label_int_nearby)

                iterative_method = False  # This is an old method. It is an iterative approach, slow but easier to test
                                          # changes. It can be removed later.
                if iterative_method:
                    # Option 1) Un-weighted average
                    if option == 1:
                        pressure = intermediate_xarray.pressure.sel(row=nearby_row, post=nearby_post,
                                                                    time=date).mean(skipna=True).values
                    # Option 2) Weighted average with transmission coefficients
                    if option == 2:
                        pressure_curr = intermediate_xarray.pressure.sel(row=row[i, j], post=post[i, j],
                                                                         time=date).values

                        # if current value is not available:
                        if np.isnan(pressure_curr):
                            # An option to use mean of nearby cells as the propagating step. Drawback: values can be
                            # larger than nearby cells with actual values:
                            #pressure_curr = 0
                            #beta = 1

                            # Another option is to set the current to the minimum of nearby cells multiplied by a factor
                            pressure_curr = intermediate_xarray.pressure.sel(row=nearby_row, post=nearby_post,
                                                                                  time=date).min(skipna=True).values
                            beta = confg_param['base_model_transmission'][0]
                        else:
                            beta = confg_param['base_model_transmission'][0]

                        # Propagate pressure
                        pressure = pressure_curr + (1 - pressure_curr / confg_param['pressure_max']) * \
                                            (beta * intermediate_xarray.pressure.sel(row=nearby_row, post=nearby_post,
                                                                              time=date).mean(skipna=True).values)

                    # Store results per cell
                    if not np.isnan(pressure):
                        interm_pressure_array[i, j] = pressure

                # Calculate Label_int
                    if calc_label_int:
                        if option == 1:
                            # Option 1) Un-weighted average
                            label_int = intermediate_xarray.label_int.sel(row=nearby_row, post=nearby_post,
                                                                          time=date).mean(skipna=True).values

                        # Option 2) Weighted average with transmission coefficients
                        if option == 2:
                            label_int_curr = intermediate_xarray.label_int.sel(row=row[i, j], post=post[i, j],
                                                                                   time=date).values

                            # if current value is not available
                            if np.isnan(label_int_curr):
                                # An option to use mean of nearby cells as the propagating step. Drawback: values can
                                # be larger than nearby cells with actual values:
                                # label_int_curr = 0
                                # beta = 1

                                # Another option is to set the current to the minimum of nearby cells
                                label_int_curr = intermediate_xarray.label_int.sel(row=nearby_row, post=nearby_post,
                                                                                 time=date).min(skipna=True).values
                                beta = confg_param['base_model_transmission'][0]
                            else:
                                beta = confg_param['base_model_transmission'][0]

                            # Propagate pressure
                            label_int = label_int_curr + (1 - label_int_curr / 1) * \
                                            (beta * intermediate_xarray.label_int.sel(row=nearby_row, post=nearby_post,
                                                                                    time=date).mean(skipna=True).values)

                        # Store results per cell
                        if not np.isnan(label_int):
                            interm_label_int_array[i, j] = round(label_int)

        # Option 1) Un-weighted average
        if option == 1:
            interm_pressure_array = interm_pressure_mean_array

        # Option 2) Weighted average with transmission coefficients
        if option == 2:
            array_d = copy.deepcopy(intermediate_xarray.pressure.sel(time=date).values)
            array_e = copy.deepcopy(intermediate_xarray.pressure.sel(time=date).values)
            array_e[~np.isnan(array_e)] = 1

            array_f = np.nan_to_num((array_e * 0), nan=1) * interm_pressure_min_array

            pressure_calc_curr = np.nansum(np.dstack((array_d, array_f)), 2)
            pressure_calc_curr[np.logical_and(np.isnan(array_d), np.isnan(array_f))] = np.nan

            array_one = np.ones((post.shape[0], post.shape[1]))

            interm_pressure_array = pressure_calc_curr + (array_one - pressure_calc_curr /
                                    confg_param['pressure_max']) * (confg_param['base_model_transmission'][0] *
                                                                    interm_pressure_mean_array)

        if calc_label_int:
            # Option 1) Un-weighted average
            if option == 1:
                interm_label_int_array = interm_label_int_mean_array

            # Option 2) Weighted average with transmission coefficients
            if option == 2:
                array_d = copy.deepcopy(intermediate_xarray.label_int.sel(time=date).values)
                array_e = copy.deepcopy(intermediate_xarray.label_int.sel(time=date).values)
                array_e[~np.isnan(array_e)] = 1

                array_f = np.nan_to_num((array_e * 0), nan=1) * interm_label_int_min_array

                label_int_calc_curr = np.nansum(np.dstack((array_d, array_f)), 2)
                label_int_calc_curr[np.logical_and(np.isnan(array_d), np.isnan(array_f))] = np.nan

                array_one = np.ones((post.shape[0], post.shape[1]))

                interm_label_int_array = np.round(label_int_calc_curr + (array_one - label_int_calc_curr / 1) * (
                                                    confg_param['base_model_transmission'][0] *
                                                    interm_label_int_mean_array))

        # Normalize and store results per day
        intermediate_xarray['label_int'].loc[dict(row=row_list, post=post_list, time=date)] = interm_label_int_array
        intermediate_xarray['pressure'].loc[dict(row=row_list, post=post_list, time=date)] = interm_pressure_array

        print('Updating step completed for day ' + str(date))

    base_model_out = intermediate_xarray

    # Normalize and call it risk
    for date in date_list:
        base_model_out['pressure'].loc[dict(row=row_list, post=post_list, time=date)] = \
            base_model_out['pressure'].loc[dict(row=row_list, post=post_list, time=date)] / confg_param['pressure_max']\
            * 100
    base_model_out = base_model_out.rename_vars({'pressure': 'risk'})

    # Save model results:
    pickle.dump([base_model_out, date_list], open('./data_and_models/' + greenhouse_param['customer'] + '_' +
                                               confg_param['label']+"_" + str(greenhouse_param['compartment']) + '_base_model' + '.p', "wb"))

    # Save results to CSV:
    base_model_out.risk.sel(row=greenhouse_param['left_rows'], time=date_list[-1]).to_pandas().to_csv(
        './csv_out/' + greenhouse_param['customer'] + '_' + confg_param['label'] + '_risk_GH_left.csv')
    base_model_out.risk.sel(row=greenhouse_param['right_rows'], time=date_list[-1]).to_pandas().to_csv(
        './csv_out/' + greenhouse_param['customer'] + '_' + confg_param['label'] + '_risk_GH_right.csv')

    print("--- Model calculation took %s seconds ---" % (time.time() - start_time))
    return base_model_out


def evaluate_base_model(base_model_out, labels_data_xarray, date_list):




    intermediate_xarray = base_model_out

    if np.logical_and(greenhouse_param['customer']=='JM', greenhouse_param['phase']==2):
        risk_prediction_left_xarray = intermediate_xarray.risk.sel(row=greenhouse_param['left_rows'],time=date_list)
        risk_prediction_right_xarray = intermediate_xarray.risk.sel(row=greenhouse_param['right_rows'],time=date_list)
        risk_actual_left_xarray = labels_data_xarray.pressure.sel(row=greenhouse_param['left_rows'],time=date_list)
        risk_actual_right_xarray = labels_data_xarray.pressure.sel(row=greenhouse_param['right_rows'],time=date_list)
    else:
        ########Risk Arrays##############
        risk_prediction_left_xarray = intermediate_xarray.risk.sel(row=list(range(greenhouse_param['left_rows'].start, greenhouse_param['left_rows'].step + greenhouse_param['left_rows'].stop,
                                                                   greenhouse_param['left_rows'].step)), time=date_list)

        risk_prediction_right_xarray = intermediate_xarray.risk.sel(row=list(range(greenhouse_param['right_rows'].start, greenhouse_param['right_rows'].step + greenhouse_param['right_rows'].stop,
                                                                    greenhouse_param['right_rows'].step)), time=date_list)
        #######Pressure Arrays###########
        risk_actual_left_xarray = labels_data_xarray.pressure.sel(row=list(range(greenhouse_param['left_rows'].start, greenhouse_param['left_rows'].step + greenhouse_param['left_rows'].stop,
                                                                   greenhouse_param['left_rows'].step)), time=date_list)

        risk_actual_right_xarray = labels_data_xarray.pressure.sel(row=list(range(greenhouse_param['right_rows'].start, greenhouse_param['right_rows'].step + greenhouse_param['right_rows'].stop,
                                                                    greenhouse_param['right_rows'].step)), time=date_list)

 #### Renaming rows to enable editing of yaxis labels
    if np.logical_and(greenhouse_param['customer']=='JM', greenhouse_param['phase']==2):

        risk_prediction_left_xarray.coords['row']= range(len(greenhouse_param['left_rows']))
        risk_prediction_right_xarray.coords['row'] = range(len(greenhouse_param['right_rows']))
        risk_actual_left_xarray.coords['row']= range(len(greenhouse_param['left_rows']))
        risk_actual_right_xarray.coords['row'] = range(len(greenhouse_param['right_rows']))
    else:
        risk_prediction_left_xarray.coords['row'] = range(greenhouse_param['num_rows_left'])
        risk_prediction_right_xarray.coords['row'] = range(greenhouse_param['num_rows_left'])
        risk_actual_left_xarray.coords['row'] = range(greenhouse_param['num_rows_left'])
        risk_actual_right_xarray.coords['row'] = range(greenhouse_param['num_rows_left'])

    # Risk calculation step:

    # Risk calculation step:

    # interpolation_type = 'gaussian'
    # cmap = 'jet'
    # norm = None

    interpolation_type = 'gaussian'
    # cmap = colors.ListedColormap(["white", "pink","lightcoral","red", "darkred"])
    # bounds = [0, 20,40,60,80, 100]
    if confg_param['label'] == 'aphidius' or confg_param['label'] == 'macrolophus' or confg_param['label'] == 'persimilis':
        cmap = colors.ListedColormap(["mintcream", "springgreen", "seagreen"])
    else:
        cmap = colors.ListedColormap(["white", "pink", "darkred"])
    bounds = [0, 33.3, 66.6, 100]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    n_cols = 5
    if len(date_list) < n_cols:
        n_cols = len(date_list)

    cmap2 = colors.ListedColormap(['darkgreen', 'limegreen', 'yellow', 'orange'])
    bounds2 = [0, 1, 2, 3, 4]
    norm2 = colors.BoundaryNorm(bounds2, cmap2.N)

    bounds3 = np.linspace(-0.001, 100.001, 100)
    norm3 = colors.BoundaryNorm(bounds3, 100)

    #####################################

    widths_gs1 = [0.5, 0.5, 1.675, 2.0]

    fig1 = plt.figure()
    spec1 = gridspec.GridSpec(ncols=4, nrows=5, figure=fig1,width_ratios=widths_gs1, wspace=0.05, hspace=0.225)
    fig1_ax1 = fig1.add_subplot(spec1[0, 0])
    fig1_ax2 = fig1.add_subplot(spec1[0, 1])
    fig1_ax3 = fig1.add_subplot(spec1[1, 0])
    fig1_ax4 = fig1.add_subplot(spec1[1, 1])
    fig1_ax5 = fig1.add_subplot(spec1[2, 0])
    fig1_ax6 = fig1.add_subplot(spec1[2, 1])
    fig1_ax7 = fig1.add_subplot(spec1[3, 0])
    fig1_ax8 = fig1.add_subplot(spec1[3, 1])
    fig1_ax9 = fig1.add_subplot(spec1[4, 0])
    fig1_ax10 = fig1.add_subplot(spec1[4, 1])

    spec2=fig1.add_gridspec(ncols=4, nrows=5, width_ratios=widths_gs1, wspace=0.05, left=0.175)

    fig1_ax11 = fig1.add_subplot(spec2[:, -2])
    fig1_ax12 = fig1.add_subplot(spec2[:, -1])


    left_side = [fig1_ax1, fig1_ax3, fig1_ax5, fig1_ax7, fig1_ax9]
    right_side = [fig1_ax2, fig1_ax4, fig1_ax6, fig1_ax8, fig1_ax10]



    days = list(np.arange(-3, 0, 1))

    lower_plots = [fig1_ax9, fig1_ax10]
    corner_plots = [fig1_ax1, fig1_ax3, fig1_ax5, fig1_ax7, fig1_ax9]


    ##########################
    # Subplot Risk Maps

    # plt.suptitle("Test")
    for i, j, k in zip(days, left_side, right_side):
        date_to_plot = risk_prediction_left_xarray.coords['time'].values[i]
        # Prediction, left side of GH:
        risk_prediction_left_xarray.sel(time=date_to_plot).plot.imshow(ax=j, x='post', y='row',
                                                                       cmap=cmap, norm=norm3,
                                                                       interpolation=interpolation_type,
                                                                       add_colorbar=False)
        j.title.set_size(10)
        j.tick_params(labelsize=10)
        j.invert_xaxis()
        j.set_yticks(np.arange(len(list_minor)))
        j.set_yticklabels(list_minor)


        if j not in corner_plots:
            j.set_yticklabels([])
            j.set_ylabel('')

        if j not in lower_plots:
            j.set_xticklabels([])
            j.set_xlabel('')
        if greenhouse_param['customer'] == 'LJ':
            j.invert_yaxis()

        minor_grid(j)



        # Prediction, right side of GH:
        risk_prediction_right_xarray.sel(time=date_to_plot).plot.imshow(ax=k, x='post', y='row',
                                                                        cmap=cmap, norm=norm3,
                                                                        interpolation=interpolation_type,
                                                                        add_colorbar=False)
        k.title.set_size(10)
        k.tick_params(labelsize=10)
        k.set_ylabel('')
        # if greenhouse_param['customer']=='JM':
        #     j.set_xticks(greenhouse_param['right_rows'])
        if greenhouse_param['customer'] == 'LJ':
            k.invert_yaxis()

        if k not in corner_plots:
            k.set_yticklabels([])

        if k not in lower_plots:
            k.set_xticklabels([])
            k.set_xlabel('')

        minor_grid(k)

    #Include Last day in the GridSpec
    risk_prediction_left_xarray.sel(time=date_list[-1]).plot.imshow(ax=fig1_ax11, x='post', y='row', cmap=cmap, norm=norm3,
                                                                   interpolation=interpolation_type, add_colorbar=False)
    fig1_ax11.set_yticks(np.arange(len(list2)))
    fig1_ax11.set_yticklabels(list2)

    if greenhouse_param['customer'] == 'LJ':
        fig1_ax11.invert_yaxis()
    fig1_ax11.invert_xaxis()

    fig1_ax11.tick_params(labelsize=12)

    fig1_ax11 = set_grid(fig1_ax11)

    risk_prediction_right_xarray.sel(time=date_list[-1]).plot.imshow(ax=fig1_ax12, x='post', y='row', cmap=cmap, norm=norm3,
                                                                    interpolation=interpolation_type,
                                                                   add_colorbar=True)

    fig1_ax12.tick_params(labelsize=12)

    if greenhouse_param['customer'] == 'LJ':
        fig1_ax12.invert_yaxis()
    fig1_ax12.set_yticklabels([])
    fig1_ax12 = set_grid(fig1_ax12)
    fig1_ax12.set_ylabel('')

    fig1.get_axes()[0].annotate('left side', (0.15, 0.9),
                               xycoords='figure fraction', ha='left',
                               fontsize=12
                               )
    fig1.get_axes()[0].annotate('right side', (0.23, 0.9),
                               xycoords='figure fraction', ha='left',
                               fontsize=12
                               )
    plt.savefig(folder_plots + "/" + 'Risk_Map_' + str(confg_param['label']) + 'v2.png')
#########################################################
    # Subplot layout and specifics
    f, ((ax1, ax2, ax3, ax4, ax5, ax6), (ax7, ax8, ax9, ax10, ax11, ax12)) = plt.subplots(2, 6, figsize=(21, 10))

    left_side = [ax1, ax3, ax5, ax7, ax9, ax11]
    right_side = [ax2, ax4, ax6, ax8, ax10, ax12]



    days = list(np.arange(-3, 0, 1))

    lower_plots = [ax7, ax8, ax9, ax10, ax11, ax12]
    upper_plots = [ax1, ax2, ax3, ax4, ax5, ax6]
    corner_plots = [ax1, ax7]

    ##########################
    # Subplot Risk Maps

    # plt.suptitle("Test")
    for i, j, k in zip(days, left_side, right_side):
        date_to_plot = risk_prediction_left_xarray.coords['time'].values[i]
        # Prediction, left side of GH:
        risk_prediction_left_xarray.sel(time=date_to_plot).plot.imshow(ax=j, x='post', y='row',
                                                                       cmap=cmap, norm=norm3,
                                                                       interpolation=interpolation_type,
                                                                       add_colorbar=False)
        j.title.set_size(14)
        j.tick_params(labelsize=10)
        j.set_yticks(np.arange(len(list2)))
        j.set_yticklabels(list2)
        j.invert_xaxis()
        j.title.set_position([1.15, 1.0])

        if j not in corner_plots:
            j.set_yticklabels([])
            j.set_ylabel('')

        if j in upper_plots:
            j.set_xticklabels([])
            j.set_xlabel('')
        if greenhouse_param['customer'] == 'LJ':
            j.invert_yaxis()

        set_grid(j)

        # Prediction, right side of GH:
        risk_prediction_right_xarray.sel(time=date_to_plot).plot.imshow(ax=k, x='post', y='row',
                                                                        cmap=cmap, norm=norm3,
                                                                        interpolation=interpolation_type,
                                                                        add_colorbar=False)

        k.tick_params(labelsize=10)

        k.set_ylabel('')
        k.set_title('')
        if greenhouse_param['customer'] == 'LJ':
            k.invert_yaxis()

        if k not in corner_plots:
            k.set_yticklabels([])

        if k in upper_plots:
            k.set_xticklabels([])
            k.set_xlabel('')
        k.set_yticklabels([])
        set_grid(k)

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.savefig(folder_plots + "/" + 'Risk_Map_' + str(confg_param['label']) + '.png')

    ####################################################################################
    g, ((ax13, ax14, ax15, ax16, ax17, ax18), (ax19, ax20, ax21, ax22, ax23, ax24)) = plt.subplots(2, 6,
                                                                                                   figsize=(21, 10))
    left_side = [ax13, ax15, ax17, ax19, ax21, ax23]
    right_side = [ax14, ax16, ax18, ax20, ax22, ax24]

    days = list(np.arange(-3, 0, 1))

    lower_plots = [ax19, ax20, ax21, ax22, ax23, ax24]
    upper_plots = [ax13, ax14, ax15, ax16, ax17, ax18]
    corner_plots = [ax13, ax19]

    #######################
    # Subplots Pressure Map
    for i, j, k in zip(days, left_side, right_side):
        date_to_plot = risk_prediction_left_xarray.coords['time'].values[i]
        # # Actual, Left side of GH:
        risk_actual_left_xarray.sel(time=date_to_plot).plot.imshow(ax=j, x='post', y='row',
                                                                   cmap=cmap2, norm=norm2, add_colorbar=False)
        j.title.set_size(14)
        j.tick_params(labelsize=10)
        j.set_yticks(np.arange(len(list2)))
        j.set_yticklabels(list2)
        j.invert_xaxis()
        j.title.set_position([1.15, 1.0])
        if j not in corner_plots:
            j.set_yticklabels([])
            j.set_ylabel('')

        if j in upper_plots:
            j.set_xticklabels([])
            j.set_xlabel('')
        if greenhouse_param['customer'] == 'LJ':
            j.invert_yaxis()

        set_grid(j)

        # # Actual, right side of GH:
        Pressure_right_plots = risk_actual_right_xarray.sel(time=date_to_plot).plot.imshow(ax=k, x='post', y='row',
                                                                                           cmap=cmap2, norm=norm2,
                                                                                           add_colorbar=False)
        k.set_title('')
        k.tick_params(labelsize=10)
        k.set_ylabel('')

        if greenhouse_param['customer'] == 'LJ':
            k.invert_yaxis()

        if k not in corner_plots:
            k.set_yticklabels([])

        if k in upper_plots:
            k.set_xticklabels([])
            k.set_xlabel('')
        k.set_yticklabels([])
        set_grid(k)

        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        plt.savefig(folder_plots + "/" + 'Pressure_Map_' + str(confg_param['label']) + '.png')

    # Prediction, left and right for specific date
    date_to_plot = risk_prediction_left_xarray.coords['time'].values[-1]
    f, (ax1, ax2) = plt.subplots(1, 2)
    risk_prediction_left_xarray.sel(time=date_to_plot).plot.imshow(ax=ax1, x='post', y='row', cmap=cmap, norm=norm3,
                                                                   interpolation=interpolation_type, add_colorbar=False)
    if greenhouse_param['customer'] == 'LJ':
        plt.gca().invert_yaxis()
    ax1.invert_xaxis()
    ax1 = set_grid(ax1)
    risk_prediction_right_xarray.sel(time=date_to_plot).plot.imshow(ax=ax2, x='post', y='row', cmap=cmap, norm=norm3,
                                                                    interpolation=interpolation_type,
                                                                    add_colorbar=False)
    # ax2.invert_yaxis()
    ax2 = set_grid(ax2)
    ax2.set_ylabel('')
    if greenhouse_param['customer'] == 'LJ':
        plt.gca().invert_yaxis()
    # plt.savefig('LD_' + str(confg_param['label']) + '.png')
    plt.savefig(folder_plots + '/LD_' + str(confg_param['label']) + '.png')

    if train_flags['interactive_test_base_model']:
        # pass the data to Dashboard
        dash = Dashboard(risk_prediction_right_xarray)
        dash.show()

def main():

    # Load labels and coverage data
    if train_flags['load_label_data']:
        # Get labels data:
        labels_data = pd.read_csv(labels_filename, parse_dates=['capture_local_datetime'])
        # Add age time of label
        labels_data['capture_date'] = labels_data['capture_local_datetime'].dt.date


        labels_data = labels_data[labels_data['capture_date'] >= dt.date(2019, 12, 15)]
        labels_data.drop_duplicates(inplace=True)

        if greenhouse_param['customer'] == 'OR':
            labels_data.loc[labels_data['label'] == 'parasitized']=labels_data["category"].replace({"unknown": "Beneficials"}, inplace=True)
            labels_data.loc[labels_data['label'] == 'parasitized'] =labels_data["label"].replace({"parasitized": "aphidius"}, inplace=True)

         # To generate a single PD Map for Origin:
        labels_data_total = labels_data


        # Selecting coverage data based on compartment for Origin
        if greenhouse_param['customer'] == 'OR' and greenhouse_param['compartment'] == 2:
            labels_data = labels_data.loc[labels_data['bay'] == 2]
        elif greenhouse_param['customer'] == 'OR' and greenhouse_param['compartment'] == 5:
            labels_data = labels_data.loc[labels_data['bay'] == 5]

        # Selecting coverage data based on compartment for Origin
        if greenhouse_param['customer'] == 'LJ' and greenhouse_param['start_row_left'] == 900:
            labels_data = labels_data.loc[labels_data['row_number'] < 1080]
        elif greenhouse_param['customer'] == 'LJ' and greenhouse_param['start_row_left'] == 1080:
            labels_data = labels_data.loc[labels_data['row_number'] >= 1080]

        # Selecting coverage data based on compartment for Origin
        if greenhouse_param['customer'] == 'NF' and greenhouse_param['start_row_left'] == 4001:
            labels_data = labels_data[(labels_data['row_number'] > 4000)&(labels_data['row_number'] <= 5150)]
        elif greenhouse_param['customer'] == 'NF' and greenhouse_param['start_row_left'] == 5001:
            labels_data = labels_data[(labels_data['row_number'] > 5000)&(labels_data['row_number'] <= 5207)]
        elif greenhouse_param['customer'] == 'NF' and greenhouse_param['start_row_left'] == 6001:
            labels_data = labels_data[(labels_data['row_number'] > 6000) & (labels_data['row_number'] <= 6207)]



        # Filtering the data based on starting and end dates
        labels_data = labels_data.loc[(labels_data['capture_date'] >= confg_param['base_model_dates'][0].date())]
        labels_data = labels_data.loc[(labels_data['capture_date'] <= confg_param['base_model_dates'][1].date())]
        labels_data.reset_index(inplace=True)
        # Get map between row number and tagid for the wave sensor data:
        row_tagid_map = pd.read_csv(row_tagid_map_filename)
        # Get map between distance and post number for the wave sensor data:
        post_dist = pd.read_csv(post_dist_map_filename)
        post_dist = post_dist.sort_values(by=['post_number', 'distance'])
        post_dist_map = pd.DataFrame()
        post_dist_df = pd.DataFrame()

        for i in range(1, post.shape[1]+1):
                post_dist_df['post_number'] = [int(i)]
                post_dist_df['distance_start'] = round(post_dist[post_dist['post_number'] == i].iloc[0][1], -1)
                post_dist_df['distance_end'] = round(post_dist[post_dist['post_number'] == i].iloc[-1][1], -1) - 1

                post_dist_map = pd.concat([post_dist_map, post_dist_df], ignore_index=True)

        # Save loaded data:
        pickle.dump([labels_data, post_dist_map, row_tagid_map], open('./data_and_models/' +
                                                        greenhouse_param['customer']+"_" + str(greenhouse_param['compartment'])+ '_label_data.p', "wb"))
    else:
        # Read data
        [labels_data, post_dist_map, row_tagid_map] = pickle.load(open('./data_and_models/' +
                                                        greenhouse_param['customer']+"_" + str(greenhouse_param['compartment']) + '_label_data.p', "rb"))

    # Prepare labels and pressures in an xarray
    if train_flags['prep_xarray_label']:
        # Prepare label data as xarray


        #Plot every Day in the range, albeit no measurements
        label_unique_days = np.arange(labels_data['capture_date'].min(), labels_data['capture_date'].max()+ timedelta(days=1))

        #Plot only days with measurements
        # label_unique_days = np.sort(labels_data[labels_data['row_number'].isin(list(range(greenhouse_param['start_row_right'],
        #                               greenhouse_param['start_row_right'] + greenhouse_param['num_rows_right'])+range(greenhouse_param['start_row_left'],
        #                               greenhouse_param['start_row_left'] + greenhouse_param['num_rows_left'])))]['capture_date'].unique())


        n_unique_days = len(label_unique_days)
        label_arr = np.zeros((post.shape[0], post.shape[1], n_unique_days)) * np.nan
        pressure_arr = np.zeros((post.shape[0], post.shape[1], n_unique_days)) * np.nan

        for i in range(0, post.shape[0]):  # For every selected row

            print('Preparing Label Data for Row: ' + str(i))

            for j in range(0, post.shape[1]):  # For every post
                label_int, pressure = get_label_for_xarray(location=[row[i, j], post[i, j]], labels_data=labels_data,
                                                             confg_param=confg_param,
                                                             label_unique_days=label_unique_days)
                label_arr[i, j, 0:n_unique_days] = label_int
                pressure_arr[i, j, 0:n_unique_days] = pressure

        label_arr = np.where(label_arr == -1, np.nan, label_arr)
        pressure_arr = np.where(pressure_arr == -1, np.nan, pressure_arr)

        days_list = np.array([np.datetime64(date) for date in label_unique_days])

        label_int_xarray = xr.DataArray(label_arr, coords=[row_list, post_list, days_list],
                                        dims=('row', 'post', 'time'),
                                        name=confg_param['label'] + ' label_int')
        pressure_xarray = xr.DataArray(pressure_arr, coords=[row_list, post_list, days_list],
                                       dims=('row', 'post', 'time'),
                                       name=confg_param['label'] + ' pressure')

        labels_data_xarray = xr.Dataset({'label_int': (['row', 'post', 'time'], label_int_xarray),
                                         'pressure': (['row', 'post', 'time'], pressure_xarray)})
        labels_data_xarray.coords['row'] = row_list
        labels_data_xarray.coords['post'] = post_list
        labels_data_xarray.coords['time'] = days_list

        if greenhouse_param['customer'] == 'OR':
            for days in labels_data_xarray.coords['time']:

                for value in labels_data_xarray.coords['row']:

                    a=labels_data_xarray['pressure'].sel(row=value, time=days).sum()
                    if a >0:
                        b=labels_data_xarray['pressure'].sel({'row':value, 'time':days}).fillna(0)
                        labels_data_xarray['pressure'].loc[dict(time=days,row=value)]=b



        # Save labels\pressure data:
        pickle.dump([labels_data_xarray, label_unique_days], open('./data_and_models/' + greenhouse_param['customer'] +
                                                                  '_' + confg_param['label'] +"_" + str(greenhouse_param['compartment'])+ '_labels' '.p', "wb"))
    else:
        try:
            #Read data
            labels_data_xarray, label_unique_days = pickle.load(open('./data_and_models/' +
                                    greenhouse_param['customer'] + '_' + confg_param['label'] +"_" + str(greenhouse_param['compartment'])+ '_labels' + '.p', "rb"))
        except:
            print('No xarray label file found. Continuing without label data')

    if train_flags['load_coverage_data']:
        # Get coverage data:
        coverage_data = pd.read_csv(coverage_filename, parse_dates=['capture_local_datetime'])
        coverage_data['capture_date'] = coverage_data['capture_local_datetime'].dt.date
        coverage_data['capture_local_datetime'] = pd.to_datetime(coverage_data['capture_local_datetime'], errors='coerce')
        coverage_data.dropna(inplace=True)
        coverage_data = coverage_data.loc[coverage_data['capture_date'] >= confg_param['base_model_dates'][0].date()]
        coverage_data = coverage_data.loc[coverage_data['capture_date'] <= confg_param['base_model_dates'][1].date()]

        coverage_data=coverage_data[coverage_data['row_session_id'].apply(lambda x: str(x).isdigit())]

        coverage_data=coverage_data[coverage_data['bay'].apply(lambda x: str(x).isdigit())]
        coverage_data=coverage_data[coverage_data['row_number'].apply(lambda x: str(x).isdigit())]
        coverage_data.row_number = coverage_data.row_number.astype(int)
        coverage_data=coverage_data[coverage_data['tag_id'].apply(lambda x: str(x).isdigit())]


        # Selecting coverage data based on compartment for Origin
        if greenhouse_param['customer'] == 'OR' and greenhouse_param['compartment'] == 2:
            coverage_data = coverage_data.loc[coverage_data['bay'] == 2]
        elif greenhouse_param['customer'] == 'OR' and greenhouse_param['compartment'] == 5:
            coverage_data = coverage_data.loc[coverage_data['bay'] == 5]

        # Selecting coverage data based on phase for Le Jardin (Serre 6 rows 900-1079, Serre 7 rows 1080-1179)
        if greenhouse_param['customer'] == 'LJ' and greenhouse_param['start_row_left'] == 900:
            coverage_data = coverage_data.loc[coverage_data['row_number'] < 1080]
        elif greenhouse_param['customer'] == 'LJ' and greenhouse_param['start_row_left'] == 1080:
            coverage_data = coverage_data.loc[coverage_data['row_number'] >= 1080]
        coverage_data.reset_index(inplace=True)

        # Selecting coverage data based on compartment for Origin
        if greenhouse_param['customer'] == 'NF' and greenhouse_param['start_row_left'] == 4001:
            coverage_data = coverage_data.loc[(coverage_data['row_number'] > 4000) & (coverage_data['row_number'] <= 5150)]
        elif greenhouse_param['customer'] == 'NF' and greenhouse_param['start_row_left'] == 5001:
            coverage_data = coverage_data.loc[(coverage_data['row_number'] > 5000) & (coverage_data['row_number'] <= 5207)]
        elif greenhouse_param['customer'] == 'NF' and greenhouse_param['start_row_left'] == 6001:
            coverage_data = coverage_data.loc[(coverage_data['row_number'] > 6000) & (coverage_data['row_number'] <= 6207)]
        coverage_data.reset_index(inplace=True)
        # Save loaded data:
        pickle.dump(coverage_data, open('./data_and_models/' + greenhouse_param['customer'] +"_" + str(greenhouse_param['compartment'])+ '_coverage_data.p',
                                        "wb"))
    else:
        # Read data
        coverage_data = pickle.load(open('./data_and_models/' + greenhouse_param['customer'] +"_" + str(greenhouse_param['compartment'])+ '_coverage_data.p',
                                         "rb"))

    # Prepare labels and pressures in an xarray
    if train_flags['prep_xarray_coverage']:
        # Prepare coverage data as xarray
        # coverage_data['capture_date'] = coverage_data['capture_local_datetime'].dt.date
        #
        coverage_unique_days = np.arange(coverage_data['capture_date'].min(),
                                      coverage_data['capture_date'].max() + timedelta(days=1))

        # coverage_unique_days = np.sort(coverage_data[coverage_data['row_number'].isin(list(range(greenhouse_param['start_row_left'],
        #                               greenhouse_param['start_row_left'] + greenhouse_param['num_rows_left'])))]['capture_date'].unique())
        n_unique_days = len(coverage_unique_days)
        coverage_arr = np.zeros((post.shape[0], post.shape[1], n_unique_days)) * np.nan

        for i in range(0, post.shape[0]):  # For every selected row
            print('Preparing Coverage Data for Row: ' + str(i))

            for j in range(0, post.shape[1]):  # For every post
                coverage = get_coverage_for_xarray(location=[row[i, j], post[i, j]], coverage_data=coverage_data,
                                                             coverage_unique_days=coverage_unique_days)
                coverage_arr[i, j, 0:n_unique_days] = coverage

        days_list = np.array([np.datetime64(date) for date in coverage_unique_days])

        coverage_data_xarray = xr.DataArray(coverage_arr, coords=[row_list, post_list, days_list],
                                        dims=('row', 'post', 'time'),
                                        name='coverage')

        # Save labels\pressure data:
        pickle.dump([coverage_data_xarray, coverage_unique_days], open('./data_and_models/' +
                                                greenhouse_param['customer']+"_" + str(greenhouse_param['compartment']) + '_coverage.p', "wb"))
    else:
        try:
            #Read data
            coverage_data_xarray, coverage_unique_days = pickle.load(open('./data_and_models/' +
                                    greenhouse_param['customer']+"_" + str(greenhouse_param['compartment']) + '_coverage' + '.p', "rb"))
        except:
            print('No xarray coverage file found. Continuing without label data')

    # Calculate base model
    if train_flags['calc_base_model']:

        # Chose an option to calculate model
        calc_interval_option = 2  # 1: Uses only dates that have labels, 2: Uses every day within the selected dates

        # Get subset of the selected days that are available in the data
        date_list = []
        dates_selected = pd.date_range(start=confg_param['base_model_dates'][0].date(),
                                  end=confg_param['base_model_dates'][1].date(),
                                  periods=(confg_param['base_model_dates'][1].date() - confg_param['base_model_dates'][
                                      0].date()).days + 1).to_pydatetime().tolist()

        if calc_interval_option == 1:
            # Use only dates that have labels
            for date in dates_selected:
                if dt.datetime.date(date) in label_unique_days:
                    date_list.append(dt.datetime.date(date))

        if calc_interval_option == 2:
            # Use every day within selected dates range
            for date in dates_selected:
                date_list.append(dt.datetime.date(date))

        base_model_out = base_model(labels_data_xarray, confg_param, date_list=date_list)
    else:
        try:
            # Read data
            [base_model_out, date_list] = pickle.load(open('./data_and_models/' + greenhouse_param['customer'] + '_' +
                                               confg_param['label'] +"_" + str(greenhouse_param['compartment'])+ '_base_model' + '.p', "rb"))
        except:
            print('No base model file found. Continuing without base model')

    # Evaluate base model
    if train_flags['test_base_model'] and 'base_model_out' in locals() and 'labels_data_xarray' in locals():

        # Show results only for dates with labels

        # Get subset of the selected days that are available in the data
        date_list = []
        dates_selected = pd.date_range(start=confg_param['base_model_dates'][0].date(),
                                  end=confg_param['base_model_dates'][1].date(),
                                  periods=(confg_param['base_model_dates'][1].date() - confg_param['base_model_dates'][
                                      0].date()).days + 1).to_pydatetime().tolist()

        dates_w_label_list = []
        for date in dates_selected:
            if dt.datetime.date(date) in label_unique_days:
                dates_w_label_list.append(dt.datetime.date(date))

        evaluate_base_model(base_model_out=base_model_out, labels_data_xarray=labels_data_xarray,
                            date_list=dates_w_label_list)

    # # Plot results:
    cmap1 = colors.ListedColormap(['white', 'blue'])
    bounds1 = [0, 0.5, 1.1]
    norm1 = colors.BoundaryNorm(bounds1, cmap1.N)

    cmap5 = colors.ListedColormap(['green', 'yellow', 'orange', 'red'])
    bounds5 = [0, 1, 2, 3, 4]
    norm5 = colors.BoundaryNorm(bounds5, cmap5.N)
    #
    # Labels and pressure time progression:
    if plot_flags['plot_labels_report'] and 'labels_data_xarray' in locals():
        #Plot P&D Ocurrence over time


        labels_data=labels_data_total


        unique_labels = labels_data['label'][labels_data['row_number'].isin(list(list(range(greenhouse_param['start_row_right'],
                              greenhouse_param['start_row_right'] + greenhouse_param['num_rows_right']))+list(range(greenhouse_param['start_row_left'],
                              greenhouse_param['start_row_left'] + greenhouse_param['num_rows_left']))))].sort_values(ascending=False).unique()

        print('Labels: ', unique_labels)
        print('Days: ', [str(date) for date in label_unique_days])

        # Plot all labels as a function of time:
        if np.logical_and(greenhouse_param['customer']=='JM', greenhouse_param['phase']==2):
            reduced_label_unique_days = np.sort(
                labels_data[labels_data['row_number'].isin(greenhouse_param['left_rows']+greenhouse_param['right_rows'])][
                    'capture_date'].unique())
        else:
            reduced_label_unique_days = np.sort(labels_data[labels_data['row_number'].isin(list(list(range(greenhouse_param['start_row_right'],
                                  greenhouse_param['start_row_right'] + greenhouse_param['num_rows_right']))+list(range(greenhouse_param['start_row_left'],
                                  greenhouse_param['start_row_left'] + greenhouse_param['num_rows_left']))))]['capture_date'].unique())
        reduced_labels_data = pd.DataFrame(np.zeros((len(reduced_label_unique_days), len(unique_labels))) * np.nan, columns=unique_labels)
        reduced_labels_data = reduced_labels_data.set_index(reduced_label_unique_days)
        for k, label in enumerate(unique_labels):
            #dummy_dates = labels_data[labels_data['label'] == label].groupby(by='capture_date').groups.keys()
            # dummy_dates = labels_data[np.logical_and(labels_data['label'] == label,
            #                            labels_data['row_number'].isin(list(list(range(greenhouse_param['left_rows'].start,
            #                              greenhouse_param['left_rows'].stop))+list(range(greenhouse_param['right_rows'].start,
            #                              greenhouse_param['right_rows'].stop)))))].groupby(
            #                              by='capture_date').groups.keys()

            if np.logical_and(greenhouse_param['customer']=='JM', greenhouse_param['phase']==2):
                dummy_dates = labels_data[np.logical_and(labels_data['label'] == label,
                                                         labels_data['row_number'].isin(
                                                             greenhouse_param['left_rows']+greenhouse_param['right_rows']))].groupby(
                    by='capture_date').groups.keys()
            else:
                dummy_dates = labels_data[np.logical_and(labels_data['label'] == label,
                                                         labels_data['row_number'].isin(
                                                             list(list(range(greenhouse_param['left_rows'].start,
                                                                             greenhouse_param[
                                                                                 'left_rows'].stop)) + list(
                                                                 range(greenhouse_param['right_rows'].start,
                                                                       greenhouse_param[
                                                                           'right_rows'].stop)))))].groupby(
                    by='capture_date').groups.keys()

            reduced_labels_data.loc[dummy_dates, label] = k+1

        # colors_dict = dict(colors.TABLEAU_COLORS, **colors.CSS4_COLORS)
        # std_colors = list(colors_dict.keys())

        #Creates a dictionary to use specific colors for each pest and disease (Update once new PD are found)

        PD_dict = {'alternaria':u'mediumvioletred',
                   'aphelinus':u'dodgerblue',
                   'aphids':u'coral',
                   'aphidius':u'orangered',
                   'bacterial-canker':u'saddlebrown',
                   'blossom-end-rot':u'orange',
                   'botrytis':u'peru',
                   'caterpillars': u'steelblue',
                   'edema':u'crimson',
                   'encarsia':u'rosybrown ',
                   'fusarium-wilt':u'olive',
                   'healthy': u'green',
                   'macrolophus': u'darkviolet',
                   'pepino-mosaic-virus': u'indigo',
                   'persimilis': u'darkslateblue',
                   'pythium': u'silver',
                   'pottay-virus': u'goldenrod',
                   'powdery-mildew':u'navy',
                   'russet-mite':u'salmon',
                   'spider-mite': u'darkcyan',
                   'thrips': u'grey',
                   'unhealthy': u'turquoise',
                   'unknown-disease': u'darkviolet',
                   'unknown-pest': u'plum',
                   'unknown-virus':u'sienna',
                   'virus1':u'aquamarine',
                   'whitefly': u'orchid',

                   }

        # List comprehension for colors
        PD_colors = [PD_dict[x] for x in unique_labels]
        fig, ax = plt.subplots()


        reduced_labels_data.plot(ax=ax, kind='line', marker='s', markersize=7, grid=True, linewidth=18, color=PD_colors, figsize=(18,4))

        for line, name in zip(ax.lines, reduced_labels_data.columns):
            y = line.get_ydata().min()
            ax.annotate(name, xy=(1, y), xytext=(6, 0), color=line.get_color(),
                        xycoords=ax.get_yaxis_transform(), textcoords="offset points",
                        size=16, va="center")
        ax.tick_params(labelsize=12)


        plt.gca().yaxis.set_major_formatter(plt.NullFormatter())
        ax.get_legend().remove()
        plt.ylabel('Issue')
        plt.xlabel('Date')
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(folder_plots + '/PD_'+str(greenhouse_param['customer'])+'_'+str(greenhouse_param['phase'])+'.png')
        tailored_plot = False  # Manual flag to enable plotting of grouped pest/disease. THe code below needs to be
                                # as needed for each customer

        if tailored_plot:

            groupa_labels_data = reduced_labels_data[['whitefly','spider-mite','thrips','caterpillars', 'unhealthy',
                                                 'unknown-disease','unknown-virus','virus1','pottay-virus',
                                                 'pepino-mosaic-virus', 'aphidius']]
            std_colors[1] = 'blue'
            std_colors[4] = 'tab:orange'
            std_colors[5] = 'tab:orange'
            std_colors[6] = 'tab:red'
            std_colors[7] = 'tab:red'
            std_colors[8] = 'tab:red'
            std_colors[9] = 'tab:red'

            groupa_labels_data.plot(kind='line', marker='s', markersize=7, grid=True, linewidth=8, colors=std_colors)
            plt.gca().yaxis.set_major_formatter(plt.NullFormatter())
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
            plt.ylabel('Issue')
            plt.xlabel('Date')


            plt.xticks(np.arange(0, 10, 1))
            plt.yticks(np.arange(0, 5, 0.5))
            groupb_labels_data = reduced_labels_data[['powdery-mildew', 'blossom-end-rot', 'pythium', 'blight']]

            groupb_labels_data.plot(kind='line', marker='s', markersize=7, grid=True, linewidth=8, colors=std_colors)
            plt.gca().yaxis.set_major_formatter(plt.NullFormatter())
            plt.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.)
            plt.ylabel('Issue')
            plt.xlabel('Date')

    if plot_flags['plot_coverage'] and 'coverage_data_xarray' in locals():
        if np.logical_and(greenhouse_param['customer']=='JM', greenhouse_param['phase']==2):
            coverage_left = coverage_data_xarray.sel(row=greenhouse_param['left_rows'], time=coverage_unique_days)
            coverage_left.coords['row'] = range(greenhouse_param['num_rows_left'])

            coverage_right = coverage_data_xarray.sel(row=greenhouse_param['right_rows'], time=coverage_unique_days)
            coverage_right.coords['row'] = range(greenhouse_param['num_rows_right'])
        else:
            coverage_left = coverage_data_xarray.sel(row=list(range(greenhouse_param['left_rows'].start,
                       greenhouse_param['left_rows'].step + greenhouse_param['left_rows'].stop,
                       greenhouse_param['left_rows'].step)), time=coverage_unique_days)
            coverage_left.coords['row'] = range(greenhouse_param['num_rows_left'])

            coverage_right = coverage_data_xarray.sel(row=list(range(greenhouse_param['right_rows'].start,
                       greenhouse_param['right_rows'].step + greenhouse_param['right_rows'].stop,
                       greenhouse_param['right_rows'].step)), time=coverage_unique_days)
            coverage_right.coords['row'] = range(greenhouse_param['num_rows_right'])

        h, ((ax25, ax26, ax27, ax28, ax29, ax30), (ax31, ax32, ax33, ax34, ax35, ax36)) = plt.subplots(2, 6,
                                                                                                       figsize=(21, 10))


        left_side = [ax25, ax27, ax29, ax31, ax33, ax35]
        right_side = [ax26, ax28, ax30, ax32, ax34, ax36]

        days = list(np.arange(-3, 0, 1))

        lower_plots = [ax31, ax32, ax33, ax34, ax35, ax36]
        upper_plots = [ax25, ax26, ax27, ax28, ax29, ax30]
        corner_plots = [ax25, ax31]

        #######################
        # Subplots Coverage Map
        for i, j, k in zip(days, left_side, right_side):
            date_to_plot = coverage_data_xarray.coords['time'].values[i]
            # # Actual, Left side of GH:
            coverage_left.sel(time=date_to_plot).plot.imshow(ax=j, x='post',
                                                                                                       y='row',
                                                                                                       cmap=cmap1,
                                                                                                       norm=norm1,
                                                                                                       add_colorbar=False)
            j.title.set_size(14)
            j.tick_params(labelsize=10)
            j.invert_xaxis()
            j.title.set_position([1.15, 1.0])
              # set the ticks to be a
            j.set_yticks(np.arange(len(list2)))
            j.set_yticklabels(list2)  # change the ticks' names to x

            if j not in corner_plots:
                j.set_yticklabels([])
                j.set_ylabel('')

            if j in upper_plots:
                j.set_xticklabels([])
                j.set_xlabel('')

            if greenhouse_param['customer'] == 'LJ':
                j.invert_yaxis()

            set_grid(j)

            # # Actual, right side of GH:
            coverage_right.sel( time=date_to_plot).plot.imshow(ax=k, x='post',
                                                                                                        y='row',
                                                                                                        cmap=cmap1,
                                                                                                        norm=norm1,
                                                                                                        add_colorbar=False)
            # k.yaxis.set_ticks(range(greenhouse_param['num_rows_right']))
            # if greenhouse_param['customer']=='JM':
            #     k.yaxis.set_ticklabels(greenhouse_param['right_rows'])
            # else:
            #     k.yaxis.set_ticklabels(range(greenhouse_param['right_rows'].start,greenhouse_param['right_rows'].stop+1,greenhouse_param['right_rows'].step))
            k.tick_params(labelsize=10)
            k.set_ylabel('')
            k.set_title('')

            if greenhouse_param['customer'] == 'LJ':
                k.invert_yaxis()

            if k not in corner_plots:
                k.set_yticklabels([])

            if k in upper_plots:
                k.set_xticklabels([])
                k.set_xlabel('')
            k.set_yticklabels([])
            set_grid(k)

            plt.subplots_adjust(wspace=0.1, hspace=0.1)
            plt.savefig(folder_plots + "/" + 'Coverage_Map_' + str(confg_param['label']) + '.png')

        # Plot coverage latest week:
        fig, axs2 = plt.subplots(ncols=1)
        date_to_plot = pd.to_datetime(str(coverage_data_xarray.coords['time'].max().values)).date()
        f = coverage_data_xarray.sel(time=days_list[-7:]).sum(dim='time', skipna=True) / coverage_data_xarray.sel(time=days_list[-7:]).sum(dim='time', skipna=True)
        f.plot.imshow(ax=axs2, cmap=cmap1, norm=norm1, vmin=0, vmax=1.1, add_colorbar=False)
        axs2.set_title('Coverage on the week ending on ' + str(date_to_plot))

        axs2.grid()
        plt.gca().invert_xaxis()
        if greenhouse_param['customer'] == 'LJ':
            plt.gca().invert_yaxis()
        plt.savefig(folder_plots + "/" + 'Coverage_LastWeek_' + str(confg_param['label']) + '.png')

        # Plot coverage penultimate week:
        fig, axs2 = plt.subplots(ncols=1)
        date_to_plot = pd.to_datetime(str(coverage_data_xarray.coords['time'].max().values)).date()- dt.timedelta(days=7)
        f = coverage_data_xarray.sel(time=days_list[-14:-7]).sum(dim='time', skipna=True) / coverage_data_xarray.sel(time=days_list[-14:-7]).sum(dim='time', skipna=True)
        f.plot.imshow(ax=axs2, cmap=cmap1, norm=norm1, vmin=0, vmax=1.1, add_colorbar=False)
        axs2.set_title('Coverage on the week ending on ' + str(date_to_plot))

        axs2.grid()
        plt.gca().invert_xaxis()
        if greenhouse_param['customer'] == 'LJ':
            plt.gca().invert_yaxis()
        plt.savefig(folder_plots + "/" + 'Coverage_PenultimateWeek_' + str(confg_param['label']) + '.png')

        # Plot coverage antepenultimate week:
        fig, axs2 = plt.subplots(ncols=1)
        date_to_plot = pd.to_datetime(str(coverage_data_xarray.coords['time'].max().values)).date()- dt.timedelta(days=14)
        f = coverage_data_xarray.sel(time=days_list[-21:-14]).sum(dim='time', skipna=True) / coverage_data_xarray.sel(time=days_list[-21:-14]).sum(dim='time', skipna=True)
        f.plot.imshow(ax=axs2, cmap=cmap1, norm=norm1, vmin=0, vmax=1.1, add_colorbar=False)
        axs2.set_title('Coverage on the week ending on ' + str(date_to_plot))

        axs2.grid()
        plt.gca().invert_xaxis()
        if greenhouse_param['customer'] == 'LJ':
            plt.gca().invert_yaxis()
        plt.savefig(folder_plots + "/" + 'Coverage_AntePenultimateWeek_' + str(confg_param['label']) + '.png')

    labels_table = labels_data_xarray.to_dataframe().reset_index()

    date_list = list(labels_table['time'].unique())
    date_list.sort()
    labels_table = labels_table[labels_table['time'].isin(date_list[-2:])]
    labels_table['pressure'] = labels_table['pressure'].round(decimals=0)
    issues_table=labels_table.groupby(['time','row', 'pressure'])['post'].agg(list)
    issues_table=issues_table.reset_index()
    issues_table['IPM'] = str(confg_param['label'])
    issues_table.sort_values(['time','pressure'], ascending=False)
    def color(val):
        if val == 0:
            color = 'darkgreen'
        elif val == 1:
            color = 'limegreen'
        elif val == 2:
            color = 'yellow'
        elif val == 3:
            color = 'orange'
        return 'background-color: %s' % color

    issues_table.style.applymap(color, subset=['time'])
    cols = issues_table.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    issues_table=issues_table[cols]
    issues_table.to_csv(folder_plots + "/"+str(confg_param['label']) + 'issues_table.csv')
    plt.show()
if __name__ == '__main__':
    main()
