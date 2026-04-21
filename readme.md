# retail market basket analysis

association rule mining on the uci online retail ii dataset using the apriori algorithm, with an interactive streamlit dashboard.

## dataset

download the dataset from the link below:
https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci

rename the downloaded file to `online_retail.csv` and place it inside the `data/` folder before running anything.

## setup & running 

open the project folder in vs code, then open a terminal (`ctrl + backtick`) and run each command below one at a time.

**step 1 — create virtual environment**
```bash
python -m venv venv
```

**step 2 — activate virtual environment**
```bash
venv\Scripts\activate
```

**step 3 — install dependencies**
```bash
pip install pandas numpy matplotlib seaborn mlxtend openpyxl streamlit networkx pyvis plotly
```

**step 4 — preprocess data and build basket matrix**
```bash
python src/preprocess.py
```

wait for: `saved basket matrix to data/basket_matrix.csv` before moving on. this step takes 5-10 minutes.

**step 5 — run exploratory data analysis**
```bash
python src/eda.py
```

plots will save to `notebooks/eda_plots/` and open on screen.

**step 6 — generate association rules**
```bash
python src/assrule.py
```

wait for: `saved rules to data/association_rules.csv`. this takes 2-4 minutes.

**step 7 — launch the dashboard**
```bash
streamlit run app/streamlit_app.py
```

the browser opens automatically at `http://localhost:8501`.

## notes

- make sure your terminal shows `(venv)` at the start before running any python command. if not, re-run step 2.
- the `data/` folder is intentionally empty in the repo. you must add `online_retail.csv` yourself after cloning..

## project structure

```
datamining/
├── app/
│   └── streamlit_app.py       # streamlit dashboard
├── data/                      # place online_retail.csv here (git ignored)
├── notebooks/
│   └── eda_plots/             # generated after running eda.py (git ignored)
├── src/
│   ├── preprocess.py          # data cleaning and basket matrix generation
│   ├── eda.py                 # exploratory data analysis and plots
│   └── assrule.py             # apriori algorithm and rule generation
├── .gitignore
├── requirements.txt
└── readme.md
```