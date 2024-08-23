
On ryzon7:
cd ~/AI-ML
virtualenv testenv # srinitest
source testenv/bin/activiate # srinitest/bin/activate
jupyter notebook --no-browser --NotebookApp.token='' --NotebookApp.password=''

python version3.9.19
cd /home/srini/AI-ML/notebooks
jupyter nbconvert --to script MilvusRag.ipynb;streamlit run MilvusRag.py
