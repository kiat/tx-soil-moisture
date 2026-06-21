import os
from read_data import file_to_indexed_df
from soil_or_met import SoilOrMet
from dup_cleaner import dup_cleaner
from time_cleaner import time_cleaner
from treat_wrong_data import find_and_replace_wrong_data
from treat_subhourly_data import treat_subhourly_data

class data_ingest:
    """
    A object that handles ingesting (prewash + cleaning) the raw TxSON data from a given folder.

    Attributes:
        input_data_folder (str): The path to the folder containing the input data (must be the raw data).
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

    def __init__(self, input_data_folder, prewash_the_data=False, clean_the_data=False, prewash_folder=None, clean_folder=None, download=False):
        self.data_folder = input_data_folder
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

        soil_or_met = SoilOrMet() 

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


    def prewash_data(self, df, station_name, download):
        df = dup_cleaner(df)
        df = treat_subhourly_data(df) # condense the data to hourly
        df = time_cleaner(df) # just fills in missing timestamp
        df = find_and_replace_wrong_data(df)
        
        if download:
            prewash_file_path = os.path.join(self.prewash_folder, f"{station_name}.csv")
            df.to_csv(prewash_file_path)  # write the prewashed result, NOT the raw df
        
        return df
    
    def clean_data(self, df, station_name, download):
        # NOTE: this method will handle imputing the data.
        raise NotImplementedError("clean_data method not implemented yet")

    def get_data_dict(self):

        self.check_for_folders()

        met_dict, soil_dict = self.open_df()

        for station, df in met_dict.items():
            if self.prewash:
                met_dict[station] = self.prewash_data(df, station, self.download)

            if self.clean:
                met_dict[station] = self.clean_data(df, station, self.download)

        for station, df in soil_dict.items():
            if self.prewash:
                soil_dict[station] = self.prewash_data(df, station, self.download)

            if self.clean:
                soil_dict[station] = self.clean_data(df, station, self.download)

        return met_dict, soil_dict

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest a folder of raw TxSON .dat files (read + optional prewash/clean)."
    )

    parser.add_argument("input_folder", help="folder of raw .dat files to ingest")
    parser.add_argument("--prewash", action="store_true",
                        help="prewash each station: dedup, condense sub-hourly, fill hourly gaps, NA out-of-range values")
    parser.add_argument("--clean", action="store_true",
                        help="clean each station, i.e. impute values (not yet implemented)")
    parser.add_argument("--download", action="store_true",
                        help="write the processed CSVs to disk")
    parser.add_argument("--prewash-folder", default=None,
                        help="where to write prewashed CSVs (default: 'prewashed_data' in the working directory)")
    parser.add_argument("--clean-folder", default=None,
                        help="where to write cleaned CSVs (default: 'cleaned_data' in the working directory)")

    args = parser.parse_args()

    ingest = data_ingest(
        input_data_folder=args.input_folder,
        prewash_the_data=args.prewash,
        clean_the_data=args.clean,
        prewash_folder=args.prewash_folder,
        clean_folder=args.clean_folder,
        download=args.download,
    )

    met_dict, soil_dict = ingest.get_data_dict()

    print(f"\nMet  stations loaded : {len(met_dict)}  — {sorted(met_dict)}")
    print(f"Soil stations loaded : {len(soil_dict)} — {sorted(soil_dict)}")

    if args.download and args.prewash:
        print(f"\nprewashed CSVs written to {ingest.prewash_folder}")
    if args.download and args.clean:
        print(f"cleaned CSVs written to {ingest.clean_folder}")


if __name__ == "__main__":
    main()