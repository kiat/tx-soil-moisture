import pandas as pd
from datetime import datetime

class SoilOrMet:
    def __init__(self, current_met_header, current_soil_header, min_soil_features = 11, min_met_features = 10):
        self.current_met_header = current_met_header
        self.current_soil_header = current_soil_header
        self.min_soil_features = min_soil_features
        self.min_met_features = min_met_features

    ### DETERMINING IF FILE IS MET, SOIL, OR UNKNOWN ###

    def get_n_lines(self,file_name, n):
        """ Given a text file:
        1. Read the first n lines
        2. RETURN the lines as a list
        """
        in_file = open(file_name, 'r')

        line = in_file.readline()
        lines = []
        while line:
            if len(lines) < n:
                lines.append(line)
            line = in_file.readline()

        in_file.close()

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
        2. RETURN True if there is a timestamp (YYYY-MM-DD hh:mm:ss), False otherwise
        """
        for line in n_lines:
            entries = line.split(",")
            for entry in entries:
                if entry:
                    entry = entry.strip('"\n') # drop parentheses and any \n
                    if self.is_valid_date(entry, "%Y-%m-%d %H:%M:%S"):
                        return True
        return False

    def determine_data_file(self, file_name):
        """ Given a file:
        1. read the first 10 lines of the file
        2. check for the presence of a timestamp in the first 10 lines
        3. count the number of features in the user supplied current met and soil headers
        4. check for the presence of the features in the first 10 lines of the file
        5. RETURN the data type (soil, met, or unknown) based on the number of specific features found and the presence of a timestamp
        """

        # 1. make sure the file contains measurement data by checking for timestamp within the first 10 rows

        # get the first 10 lines
        lines = self.get_n_lines(file_name, 10)

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
                    entry = entry.strip('"\n') # drop parentheses and any \n
                    if entry in self.current_soil_header:
                        soil_features.append(entry)
  
                    elif entry in self.current_met_header:
                        met_features.append(entry)

            # determine data type
            if len(soil_features) >= self.min_soil_features:
                return 'soil'
            elif len(met_features) >= self.min_met_features:
                return 'met'

        return 'unknown'