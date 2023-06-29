import pandas as pd
import os

# Load Data
ds_result = pd.read_csv(os.path.join(os.getcwd(), "data", "modify", "test_result_m.csv"))

# Check and Set if Survived == Result then Compare True
ds_result['Compare'] = ds_result['Survived'] == ds_result['Result']

print(ds_result['Compare'].count())
print(ds_result.query('Compare == True')['Compare'].count())

Tot_cnt = ds_result['Compare'].count()
OK_cnt = ds_result.query('Compare == True')['Compare'].count()

print(OK_cnt / Tot_cnt * 100)


