from skyfield.api import load, wgs84
import numpy as np
from skyfield.positionlib import Geocentric
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import calendar



eph = load('de421.bsp')
earth = eph['earth'] # vector from solar system barycenter to geocenter
sun = eph['sun'] # vector from solar system barycenter to sun
geocentric_sun = sun - earth # vector from geocenter to sun

ts = load.timescale()

def cross_product_matrix(a):
    """ computes the cross product matrix
    see https://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
    code from https://stackoverflow.com/questions/66707295/numpy-cross-product-matrix-function
    """
    return np.cross(a, np.identity(3) * -1)

def rotation_matrix_around_axis(axis_vector, rotation_degrees):
    """ creates a rotation matrix that rotates around a given axis
    see https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    """
    
    rotation_radians = rotation_degrees / 180 * np.pi
    return (
        np.cos(rotation_radians) * np.identity(3)
        + np.sin(rotation_radians) * cross_product_matrix(axis_vector)
        + (1 - np.cos(rotation_radians)) * np.outer(axis_vector, axis_vector)
        )
        
data=[]


year1 = 2024
year2 = 2024
month1 = 5
month2 = 5
day1 = 1
#day2 = 31
#month = 3
for year in range(year1,year2+1):
    for month in range(month1,month2+1):
        _, days_in_month = calendar.monthrange(year, month)

        for day in range(1,days_in_month+1):
            print(f"Calculating solar ephemeris for {year}-{month:02}-{day:02}")


            SSP = []
            times = []
            for hour in range(0, 24):
    
                for minute in range(0, 60, 10):
                #print(hour)
                #print(minute)
                    t = ts.utc(year, month, day, hour, minute)

                    sun_subpoint = wgs84.subpoint(geocentric_sun.at(t)) # subpoint method requires a geocentric position



                    terminator_angle_from_sun = 90.833 # 90 degrees + sun's semidiameter + refraction at horizon

                    sun_vec = geocentric_sun.at(t).position.au # numpy array of sun's position vector
                    normal_vec = np.cross(sun_vec, np.array([1,0,0]))# vector normal to sun position and x-axis
                    first_terminator_vec = rotation_matrix_around_axis(normal_vec, terminator_angle_from_sun) @ sun_vec # arbitrary first position on terminator

                    terminator_latitudes = []
                    terminator_longitudes = []

                    num_points_on_terminator = 100
                    for angle in np.linspace(0, 360, num_points_on_terminator):
                        terminator_vector = rotation_matrix_around_axis(sun_vec, angle) @ first_terminator_vec
                        terminator_position = Geocentric(terminator_vector, t=t)
                        geographic_position = wgs84.subpoint(terminator_position)
                        terminator_latitudes.append(geographic_position.latitude.degrees)
                        terminator_longitudes.append(geographic_position.longitude.degrees)

                    terminator_latitudes = np.array(terminator_latitudes)
                    terminator_longitudes = np.array(terminator_longitudes)
        
                    #print('subpoint latitude: ', sun_subpoint.latitude.degrees)
                    #print('subpoint longitude: ', sun_subpoint.longitude.degrees)
                    time_str=str(hour)+" "+str(minute)
                    time = str(datetime.strptime(time_str, '%H %M').time())
                    #print(time,sun_subpoint.latitude.degrees,sun_subpoint.longitude.degrees)
                    SSP1 = (time,sun_subpoint.latitude.degrees,sun_subpoint.longitude.degrees)
                    SSP.append(SSP1)
                    times.append(time)
                    STER1 = np.column_stack((terminator_longitudes, terminator_latitudes))
                    #decimal_time = str("%13.4f" % (hour + minute/60))
        

            # for i in range(len(STER1)):
                # print(STER1[i])
            # # # fig, ax = plt.subplots()
            # # # plt.plot(STER1[:,0], STER1[:,1], 'o')
            # # # ax.set_xlim(-180, 180)
            # # # ax.set_xticks(np.arange(-180, 181, 30))
            # # # ax.set_ylim(-90, 90)
            # # # ax.set_yticks(np.arange(-90, 91, 20))
            # # # plt.xlabel('Terminator Longitudes')
            # # # plt.ylabel('Terminator Latitudes')
            # # # plt.show()        
            # sys.exit()


      
                    parent_dir = str(year)
                    month1='{:02d}'.format(month)
                    day1='{:02d}'.format(day)
                    month_dir = os.path.join(parent_dir, str(month1))
                
                    if not os.path.exists(month_dir):
                        os.makedirs(month_dir)
                    
                    #import datetime
                    year2 = str(year)
                    month2 = str(month1)
                    day2 = str(day1)
                    date_str=year2+'-'+month2+'-'+day2
                    
                    
                    
                    try:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                        day_of_year = date_obj.strftime("%j")
                    except ValueError:
                        print("Invalid date provided.")
                    

                    #date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
                    #day_of_year = date_obj.strftime("%j")
                    #print(day_of_year)
                 

                    
                    day_dir = os.path.join(month_dir, str(day_of_year))

                
                    if not os.path.exists(day_dir):
                        os.makedirs(day_dir)
                
                
                    filester = os.path.join(day_dir, "{}_{}_{}_{:02d}{:02d}.STER".format(year, month1, day_of_year, hour, minute))

                    with open(filester, 'w') as f:
                        for i in range(len(STER1)):
                            data=STER1[i]
                            line = "{:.5f} {:.5f}".format(data[0], data[1])
                            f.write(line)
                            f.write('\n')
                        if f:
                            f.close()
        


            filessp = os.path.join(day_dir, str(year)+'_'+str(month1)+'_'+str(day_of_year)+'.SSP')

            with open(filessp, 'w') as f:
                for i in range(len(SSP)):
                    data=SSP[i]
                    line = "{} {:.5f} {:.5f}".format(data[0], data[1], data[2])
                    f.write(line)
                    f.write('\n')
                if f:
                    f.close()
        
        
        
                    # terminator_array = np.column_stack((terminator_latitudes, terminator_longitudes))
                    # print(terminator_array)

                    # filename = str(year)+'_'+str(month)+'_'+str(day)+'.SUNCD'
                    # print(filename)

        
                    # print(type(terminator_longitudes), type(terminator_latitudes))
        
        
                    # import matplotlib.pyplot as plt

                    # fig, ax = plt.subplots()
                    # ax.scatter(terminator_longitudes, terminator_latitudes)
                    # ax.scatter(sun_subpoint.longitude.degrees, sun_subpoint.latitude.degrees)
                    # ax.grid(True)
                    # ax.set_xlim(-180, 180)
                    # ax.set_xticks(np.arange(-180, 181, 30))
                    # ax.set_ylim(-90, 90)
                    # ax.set_yticks(np.arange(-90, 91, 20))
                    # dt = t.utc_datetime()
                    # dt_str = dt.strftime("%Y-%m-%d %H:%M:%S")
                    # ax.set_title("Terminator Line and Sun Subpoint Location for "+dt_str)
                    # plt.show()
                    # sys.exit()




























