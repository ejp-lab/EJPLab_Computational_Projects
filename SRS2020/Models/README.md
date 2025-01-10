# Sklearn Pickled Models
Model trained for SRS2020 and the scaler are avalible here as pickled objects. 
These objects can be accesed in python through the code below
```python
import pickle
gbt_model = pickle.load(open('Relaxed_Beta16_Min.sav, 'rb'))
minmax_scaler = pickle.load(open('scaler.sav','rb'))
```
