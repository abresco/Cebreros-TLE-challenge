# CEBREROS-RFI PROJECT ESA

# 2026 
# AGUSTI BRESCO, MIGUEL ESCUDERO



# import requests
# import json
# from datetime import datetime



# iso_date_start = "2025-03-19T00:00:00Z"
# iso_date_end = "2025-03-20T00:00:00Z"
# # 2. Convertir a objeto datetime
# dt_start = datetime.fromisoformat(iso_date_start.replace('Z', '+00:00'))
# dt_end = datetime.fromisoformat(iso_date_end.replace('Z', '+00:00'))


# jd_start = (dt_start.timestamp() / 86400) + 2440587.5
# jd_end = (dt_end.timestamp() / 86400) + 2440587.5

# url = 'https://satchecker.cps.iau.org/tools/get-tle-data/'
# params = {'id': '61449',
#           'id_type': 'catalog',
#           'start_date_jd': jd_start,
#           'end_date_jd': jd_end
#         }

# r = requests.get(url, params=params)
# print(json.dumps(r.json(), indent=4))

import requests
import re
from datetime import datetime

# Dates to define
iso_date_start = "2025-03-19T23:10:00Z"
iso_date_end = "2025-03-19T23:20:00Z"
iso_date_target = "2025-03-19T23:16:00Z"

# Coordinates of Cebreros site 
site_lon = -4.7247  
site_lat = 40.4528
site_alt = 0.794

# Target ID. 499 is Mars. Hera is 61449
target_id = '61449'

params = {
    'format': 'text',
    'COMMAND': target_id,
    'MAKE_EPHEM': 'YES',
    'EPHEM_TYPE': 'OBSERVER',
    'CENTER': 'c@399',  # Cebreros site code in Horizons
    # 'CENTER': f'{site_lon},{site_lat},{site_alt}',  # This format is not working currently
    'START_TIME': '2025-Mar-19',
    'STOP_TIME': '2025-Mar-19',
    'STEP_SIZE': '1m',
    'QUANTITIES': '23'  # 23=azimuth, 24=elevation
}

response = requests.get('https://ssd.jpl.nasa.gov/api/horizons.api', params=params)
print("Status:", response.status_code)
print("Response preview:\n", response.text[:2000]) 


lines = response.text.split('\n')
in_data = False
az_el_data = []

for line in lines:
    if '$$SOE' in line:
        in_data = True
        continue
    if '$$EOE' in line:
        break
    if in_data and line.strip() and not line.startswith(' Date'): 
        match = re.search(r'([-\d.]+)\s+([-\d.]+)', line, re.X)
        if match:
            azi, elv = match.groups()
            az_el_data.append({'azi': float(azi), 'elv': float(elv), 'line': line.strip()})

# Resultados para la fecha target aproximada (últimos minutos)
if az_el_data:
    print("\nAzimuth and Elevation for every minute from Cebreros:")
    for data in az_el_data[-5:]:  # Last 5 points (around 23:16)
        print(f"Azi: {data['azi']:.2f}°, Elv: {data['elv']:.2f}°")
    print(f"\nAzi/Elv a ~{iso_date_target}: {az_el_data[-1]['azi']:.2f}°, {az_el_data[-1]['elv']:.2f}°")
else:
    print("Some error occurred.")
