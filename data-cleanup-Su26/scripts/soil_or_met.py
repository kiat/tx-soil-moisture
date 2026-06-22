from datetime import datetime
import os

expected_soil_header = "Date,Ppt,SWC_5,SWC_10,SWC_20,SWC_50,T_5,T_10,T_20,T_50,Flag"
expected_met_header = 'TIMESTAMP,RECORD,Rain_mm_Tot,AirTC_Avg,RH,WS_ms_S_WVT,WindDir_D1_WVT,SlrW_Avg,ETos,Rso'

class SoilOrMet:
    def __init__(self, current_met_header = expected_met_header, current_soil_header = expected_soil_header, min_soil_features = 9, min_met_features = 10):
        """
        A object that determines if a given file contains soil or met data

        Attributes:
        current_met_header: list of expected met features
        current_soil_header: list of expected soil features
        min_soil_features: minimum number of expected soil features that must be present in the file to be classified as soil
        min_met_features: minimum number of expected met features that must be present in the file to

        Methods:
        get_n_lines(file_name, n)
        is_valid_date(date_str, format_str)
        find_timestamp(n_lines)
        determine_data_file(file_name)
        """
        self.current_met_header = current_met_header
        self.current_soil_header = current_soil_header
        self.min_soil_features = min_soil_features
        self.min_met_features = min_met_features

    ### DETERMINING IF FILE IS MET, SOIL, OR UNKNOWN ###

    def get_n_lines(self, file_path, n):
        lines = []
        with open(file_path, 'r') as in_file:
            for line in in_file:
                lines.append(line)
                if len(lines) >= n:
                    break
        return lines

    def is_valid_date(self, date_str, format_str):
        """ Given a string and a date format:
        1. Check if the string is a valid date in the given format
        2. RETURN True if it is a valid date, False otherwise 
        """
        try:
            return bool(datetime.strptime(date_str, format_str))
        except ValueError:
            return False

    def find_timestamp(self, n_lines):
        """ Given a certain number of lines:
        1. check the lines for a timestamp
        2. RETURN True if there is a timestamp, False otherwise
        """
        for line in n_lines:
            entries = line.split(",")
            for entry in entries:
                if entry:
                    entry = entry.strip('"\n') # drop parentheses and any \n
                    if self.is_valid_date(entry, "%Y-%m-%d %H:%M:%S"):
                        return True
        return False

    def determine_data_file(self, file_path):
        """ Given a file:
        1. read the first 10 lines of the file
        2. check for the presence of a timestamp in the first 10 lines
        3. count the number of features in the user supplied current met and soil headers
        4. check for the presence of the features in the first 10 lines of the file
        5. RETURN the data type (soil, met, or unknown) based on the number of specific features found and the presence of a timestamp
        """

        # input validation
        if not isinstance(file_path, str):
            raise TypeError("file_path must be a string.")
        if file_path.strip() == "":
            raise ValueError("file_path must be a non-empty string.")

        # 1. make sure the file contains measurement data by checking for timestamp within the first 10 rows

        # get the first 10 lines
        lines = self.get_n_lines(file_path, 10)

        # get the lines with commas:
        rows = []
        for line in lines:
            if "," in line:
                rows.append(line)

        # check for YYYY-MM-DD hh:mm:ss in the rows
        if not self.find_timestamp(rows):
            return 'unknown'

        # 2. check for the features.

        # find the number of soil and met features in the files
        for row in rows:
            entries = row.split(",")

            soil_features = []
            met_features = []
            for entry in entries:
                if entry:
                    entry = entry.strip('"\n') # drop "" and any \n
                    entry = entry.strip("'\n") # drop '' and any \n
                    entry = entry.strip() # drop any leading or trailing whitespace

                    if entry in self.current_soil_header:
                        soil_features.append(entry)
  
                    if entry in self.current_met_header:
                        met_features.append(entry)

            # determine type
            if len(soil_features) >= self.min_soil_features:
                return 'soil'
            elif len(met_features) >= self.min_met_features:
                return 'met'

        return 'unknown'