This analysis uses observed turbidity/SSC data from USGS and SFEI sensors
to construct a spatially extrapolated, time-varying light attenuation field for 
the SFEI DELWAQ bay model. 

#### Processing scripts and data steps ####

Scripts - folder of processing scripts, ordered as Step#. Each script described in comments at top of script

Data_SSC_Raw - Folders of raw observational data

Data_SSC_Procssed - processed SSC time-series from Data_SSC_Raw. Generated using Step0_Process_RawData_v2.py

Data_Constructed_SSC_and_Kd - filled long-term time-series of SSC and Kd for each observational site. Generated with Step1_Build_Modeled_SSCandKd_v1.py

Data_Kd_Shifted - CSV and NetCDF files of long-term Kd time-series, and shifted versions of these time-series to better match observed Kd from USGS cruises. Generated with Step2_Bend_Kd_to_Observations_v1.py

Data_DELWAQ_InputFiles - DELWAQ-ready Kd netcdf files for fide grid and agg grid. Fine grid file generated with Step4_Create_DELWAQ_Input_Kd_vFineGrid.py. Agg grid with - Step5_Aggregate_FineGrid_to_AggGrid.py.
				Date ranges spec'ed in file naming convention. These scritps draw on a long-term time-series (currently wy 2010-2018) and can be written 
				for any period in this range. Current files are just short-term examples (-DCR 11/20/20). 

############# Other ###############

Data_Cruise - handling of USGS Peterson cruise data

Data_Forcing - organization of forcing data sets that are used to fill the SSC/Kd time-series in Step1_Build_Modeled_SSCandKd_v1.py

Match_Cruise_to_HFsite_for_ssc2kd_and_gamtrends.xlsx - Excel file for matching sensor sites to USGS Cruise sites for
								1) Building the "trend" forcing time-series for modeling/filling purposes
								2) Building site-specific SSC to Kd conversions. 

Match_Cruise_to_HFsite_for_Kd_Bending.xlsx - Excel file matching sensor sites to USGS Cruise sites for shfiting/bending the Kd time-series to match cruise observations

Station_Polygon_Coordinates.csv - coordinates of centroids of sensor-specific polygons used for grid extrapolation. 






