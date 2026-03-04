# CEBREROS-RFI PROJECT ESA

# 2026 
# AGUSTI BRESCO 

# CEBREROS-RFI PROJECT ESA

# 2026 
# AGUSTI BRESCO 

import requests
import json
from datetime import datetime



iso_date_start = "2025-03-19T00:00:00Z"
iso_date_end = "2025-03-20T00:00:00Z"
# 2. Convertir a objeto datetime
dt_start = datetime.fromisoformat(iso_date_start.replace('Z', '+00:00'))
dt_end = datetime.fromisoformat(iso_date_end.replace('Z', '+00:00'))


jd_start = (dt_start.timestamp() / 86400) + 2440587.5
jd_end = (dt_end.timestamp() / 86400) + 2440587.5

url = 'https://satchecker.cps.iau.org/tools/get-tle-data/'
params = {'id': '61449',
          'id_type': 'catalog',
          'start_date_jd': jd_start,
          'end_date_jd': jd_end
        }

r = requests.get(url, params=params)
print(json.dumps(r.json(), indent=4))