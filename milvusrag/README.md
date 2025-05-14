
On ryzon7:
- cd ~/AI-ML
- ```virtualenv testenv``` # srinitest
- ```source testenv/bin/activate``` # srinitest/bin/activate
- ```jupyter notebook --no-browser --NotebookApp.token='' --NotebookApp.password=''```

python version3.9.19
- cd /home/srini/AI-ML/notebooks
- ```jupyter nbconvert --to script MilvusRag.ipynb;streamlit run MilvusRag.py```


** Milvus102.ipynb - This file contains code for loading the XLS file data into Milvus after cleaning the data.
