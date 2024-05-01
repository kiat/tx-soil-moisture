#!/bin/bash

GREP_OPTIONS=''

cookiejar=$(mktemp cookies.XXXXXXXXXX)
netrc=$(mktemp netrc.XXXXXXXXXX)
chmod 0600 "$cookiejar" "$netrc"
function finish {
  rm -rf "$cookiejar" "$netrc"
}

trap finish EXIT
WGETRC="$wgetrc"

prompt_credentials() {
    echo "Enter your Earthdata Login or other provider supplied credentials"
    read -p "Username (lijmr): " username
    username=${username:-lijmr}
    read -s -p "Password: " password
    echo "machine urs.earthdata.nasa.gov login $username password $password" >> $netrc
    echo
}

exit_with_error() {
    echo
    echo "Unable to Retrieve Data"
    echo
    echo $1
    echo
    echo "https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2021_292.dap.nc4"
    echo
    exit 1
}

prompt_credentials
  detect_app_approval() {
    approved=`curl -s -b "$cookiejar" -c "$cookiejar" -L --max-redirs 5 --netrc-file "$netrc" https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2021_292.dap.nc4 -w '\n%{http_code}' | tail  -1`
    if [ "$approved" -ne "200" ] && [ "$approved" -ne "301" ] && [ "$approved" -ne "302" ]; then
        # User didn't approve the app. Direct users to approve the app in URS
        exit_with_error "Please ensure that you have authorized the remote application by visiting the link below "
    fi
}

setup_auth_curl() {
    # Firstly, check if it require URS authentication
    status=$(curl -s -z "$(date)" -w '\n%{http_code}' https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2021_292.dap.nc4 | tail -1)
    if [[ "$status" -ne "200" && "$status" -ne "304" ]]; then
        # URS authentication is required. Now further check if the application/remote service is approved.
        detect_app_approval
    fi
}

setup_auth_wget() {
    # The safest way to auth via curl is netrc. Note: there's no checking or feedback
    # if login is unsuccessful
    touch ~/.netrc
    chmod 0600 ~/.netrc
    credentials=$(grep 'machine urs.earthdata.nasa.gov' ~/.netrc)
    if [ -z "$credentials" ]; then
        cat "$netrc" >> ~/.netrc
    fi
}

fetch_urls() {
  if command -v curl >/dev/null 2>&1; then
      setup_auth_curl
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        curl -f -b "$cookiejar" -c "$cookiejar" -L --netrc-file "$netrc" -g -o $stripped_query_params -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  elif command -v wget >/dev/null 2>&1; then
      # We can't use wget to poke provider server to get info whether or not URS was integrated without download at least one of the files.
      echo
      echo "WARNING: Can't find curl, use wget instead."
      echo "WARNING: Script may not correctly identify Earthdata Login integrations."
      echo
      setup_auth_wget
      while read -r line; do
        # Get everything after the last '/'
        filename="${line##*/}"

        # Strip everything after '?'
        stripped_query_params="${filename%%\?*}"

        wget --load-cookies "$cookiejar" --save-cookies "$cookiejar" --output-document $stripped_query_params --keep-session-cookies -- $line && echo || exit_with_error "Command failed with error. Please retrieve the data manually."
      done;
  else
      exit_with_error "Error: Could not find a command-line downloader.  Please install curl or wget"
  fi
}

fetch_urls <<'EDSCEOF'
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_345.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_346.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_347.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_348.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_349.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_350.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_351.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_352.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_353.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_354.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_355.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_356.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_357.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_358.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_359.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_360.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_361.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_362.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_363.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_364.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2022_365.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_001.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_002.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_003.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_004.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_005.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_006.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_007.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_008.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_009.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_010.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_011.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_012.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_013.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_014.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_015.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_016.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_017.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_018.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_019.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_020.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_021.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_022.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_023.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_024.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_025.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_026.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_027.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_028.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_029.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_030.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_031.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_032.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_033.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_034.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_035.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_036.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_037.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_038.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_039.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_040.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_041.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_042.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_043.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_044.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_045.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_046.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_047.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_048.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_049.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_050.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_051.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_052.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_053.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_054.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_055.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_056.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_057.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_058.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_059.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_060.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_061.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_062.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_063.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_064.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_065.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_066.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_067.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_068.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_069.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_070.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_071.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_072.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_073.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_074.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_075.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_076.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_077.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_078.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_079.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_080.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_081.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_082.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_083.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_084.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_085.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_086.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_087.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_088.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_089.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_090.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_091.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_092.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_093.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_094.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_095.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_096.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_097.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_098.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_099.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_100.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_101.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_102.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_103.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_104.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_105.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_106.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_107.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_108.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_109.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_110.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_111.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_112.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_113.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_114.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_115.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_116.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_117.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_118.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_119.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_120.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_121.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_122.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_123.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_124.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_125.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_126.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_127.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_128.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_129.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_130.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_131.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_132.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_133.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_134.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_135.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_136.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_137.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_138.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_139.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_140.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_141.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_142.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_143.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_144.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_145.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_146.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_147.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_148.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_149.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_150.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_151.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_152.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_153.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_154.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_155.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_156.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_157.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_158.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_159.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_160.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_161.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_162.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_163.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_164.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_165.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_166.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_167.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_168.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_169.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_170.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_171.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_172.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_173.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_174.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_175.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_176.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_177.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_178.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_179.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_180.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_181.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_182.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_183.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_184.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_185.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_186.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_187.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_188.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_189.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_190.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_191.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_192.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_193.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_194.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_195.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_196.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_197.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_198.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_199.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_200.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_201.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_202.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_203.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_204.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_205.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_206.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_207.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_208.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_209.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_210.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_211.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_212.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_213.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_214.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_215.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_216.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_217.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_218.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_219.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_220.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_221.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_222.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_223.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_224.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_225.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_226.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_227.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_228.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_229.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_230.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_231.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_232.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_233.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_234.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_235.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_236.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_237.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_238.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_239.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_240.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_241.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_242.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_243.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_244.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_245.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_246.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_247.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_248.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_249.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_250.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_251.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_252.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_253.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_254.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_255.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_256.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_257.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_258.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_259.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_260.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_261.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_262.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_263.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_264.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_265.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_266.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_267.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_268.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_269.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_270.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_271.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_272.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_273.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_274.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_275.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_276.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_277.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_278.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_279.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_280.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_281.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_282.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_283.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_284.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_285.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_286.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_287.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_288.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_289.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_290.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_291.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_292.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_293.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_294.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_295.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_296.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_297.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_298.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_299.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_300.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_301.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_302.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_303.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_304.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_305.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_306.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_307.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_308.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_309.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_310.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_311.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_312.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_313.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_314.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_315.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_316.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_317.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_318.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_319.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_320.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_321.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_322.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_323.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_324.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_325.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_326.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_327.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_328.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_329.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_330.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_331.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_332.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_333.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_334.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_335.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_336.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_337.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_338.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_339.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_340.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_341.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_342.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_343.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_344.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_345.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_346.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_347.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_348.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_349.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_350.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_351.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_352.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_353.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_354.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_355.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_356.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_357.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_358.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_359.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_360.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_361.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_362.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_363.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_364.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2023_365.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_001.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_002.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_003.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_004.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_005.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_006.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_007.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_008.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_009.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_010.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_011.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_012.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_013.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_014.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_015.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_016.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_017.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_018.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_019.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_020.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_021.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_022.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_023.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_024.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_025.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_026.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_027.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_028.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_029.dap.nc4
https://opendap.earthdata.nasa.gov/collections/C2205122332-POCLOUD/granules/ucar_cu_cygnss_sm_v1_2024_030.dap.nc4
EDSCEOF