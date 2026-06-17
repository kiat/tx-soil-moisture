import os
from read_data import file_to_indexed_df
from soil_or_met import SoilOrMet
from dup_cleaner import dup_cleaner
from time_cleaner import time_cleaner
from treat_wrong_data import find_and_replace_wrong_data

class data_ingest:
    """
    A object that handles ingesting (prewash + cleaning) the raw TxSON data from a given folder.

    Attributes:
        input_data_folder (str): The path to the folder containing the input data (must be the raw data).
        expected_met_header (str): The expected header for the met data files, used for identifying met files.
        expected_soil_header (str): The expected header for the soil data files, used for identifying soil files.
        prewash_the_data (bool): Whether to prewash the data. Handle duplicates, missing timestamps, and other oddities
        clean_the_data (bool): Whether to clean the data. Handles nonsensical values.
        prewash_folder (str): The path to the folder containing the prewashed data.
        clean_folder (str): The path to the folder containing the cleaned data.
        download (bool): Whether to download the data.

    methods:
        open_data()
        prewash_data(met_dict, soil_dict)
        clean_data(met_dict, soil_dict)
        get_data_dict()
    """

    expected_soil_header = "Date,Ppt,SWC_5,SWC_10,SWC_20,SWC_50,T_5,T_10,T_20,T_50,Flag" 
    expected_met_header = 'TIMESTAMP,RECORD,Rain_mm_Tot,AirTC_Avg,RH,WS_ms_S_WVT,WindDir_D1_WVT,SlrW_Avg,ETos,Rso'

    def __init__(self, input_data_folder , expected_met_header=expected_met_header, expected_soil_header=expected_soil_header, prewash_the_data=False, clean_the_data=False, prewash_folder=None, clean_folder=None, download=False):
        self.data_folder = input_data_folder
        self.expected_met_header = expected_met_header
        self.expected_soil_header = expected_soil_header
        self.prewash = prewash_the_data
        self.clean = clean_the_data
        self.prewash_folder = prewash_folder
        self.clean_folder = clean_folder
        self.download = download

    ####### check methods ########

    def check_for_folders(self):

        # first check if the data folder exists, if not raise an error
        if not os.path.exists(self.data_folder):
            raise ValueError("Data folder does not exist")
        
        # if download is true, check if their respective folder is provided, else make the folder in the working directory
        working_dir = os.getcwd()
        if self.download:

            if self.prewash:
                if not self.prewash_folder:
                    self.prewash_folder = os.path.join(working_dir, "prewashed_data")
                os.makedirs(self.prewash_folder, exist_ok=True)

            if self.clean:
                if not self.clean_folder:
                    self.clean_folder = os.path.join(working_dir, "cleaned_data")
                os.makedirs(self.clean_folder, exist_ok=True)

    ####### data methods ########
 
    def open_df(self):
        """ Open the data into two dictionaries of df, one for met data and one for soil data, handles only raw data."""

        ### NOTE: HARDCODED (06/11/2026) MIN FEATURES FOR SOIL AND MET ###
        soil_or_met = SoilOrMet(current_met_header = self.expected_met_header, current_soil_header = self.expected_soil_header, min_soil_features = 9, min_met_features = 10) # this config of min features allows all raw files to be identified

        soildata_dict = {}
        metdata_dict = {}
        # loop through the files in the data folder, determine if they are soil or met for read_data script, and read them into a dictionary of dataframes
        for file in os.listdir(self.data_folder):
            file_path = os.path.join(self.data_folder, file)
            station_name = file.split(".")[0] # assumes file name is in format "station.dat"
            if os.path.isfile(file_path):
                is_soil_or_met = soil_or_met.determine_data_file(file_path)
                df = file_to_indexed_df(file_path, is_soil_or_met)
                if df is not None:
                    if is_soil_or_met == "soil":
                        soildata_dict[station_name] = df
                    elif is_soil_or_met == "met":
                        metdata_dict[station_name] = df
        return metdata_dict, soildata_dict  # keys are station names, values are dataframes with datetime index and columns of features.


    def prewash_data(self, met_dict, soil_dict):
        
        for station, df in met_dict.items():
            df = dup_cleaner(df)
            df = time_cleaner(df) # just fills in missing timestamp
            df = find_and_replace_wrong_data(df)
            met_dict[station] = df

            if self.download:
                prewash_file_path = os.path.join(self.prewash_folder, f"{station}.csv")
                df.to_csv(prewash_file_path)

        for station, df in soil_dict.items():
            df = dup_cleaner(df)
            df = time_cleaner(df) # just fills in missing timestamp
            df = find_and_replace_wrong_data(df)
            soil_dict[station] = df

            if self.download:
                prewash_file_path = os.path.join(self.prewash_folder, f"{station}.csv")
                df.to_csv(prewash_file_path)

        return met_dict, soil_dict
    
    def clean_data(self, met_dict, soil_dict):
        # NOTE: this method will handle imputing the data.
        raise NotImplementedError("clean_data method not implemented yet")

    def get_data_dict(self):

        self.check_for_folders()

        met_dict, soil_dict = self.open_df()

        if self.prewash:
            met_dict, soil_dict = self.prewash_data(met_dict, soil_dict)
        if self.clean:
            met_dict, soil_dict = self.clean_data(met_dict, soil_dict)
        return met_dict, soil_dict