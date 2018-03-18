import numpy as np
import pandas as pd
import statsmodels.api as sm
import utils
import copy

filename = 'cars.csv'
df = pd.read_csv(filename)
print(len(df))
print(df.columns)

# scatter plot -- nothing to see
#utils.scatter_plot_dataframe(df)

# correlation matrix and plot
#print(df.corr())
#utils.correlation_plot(df)

dep_var = 'mpg'
indep_vars = copy.deepcopy(df.columns).drop(['mpg', 'name'])

X = df[indep_vars]
print(np.shape(X))
X = sm.add_constant(X)

y = df[dep_var]

model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
print(results)


mod = sm.OLS(y, X)
res = mod.fit_regularized(alpha=1.0, L1_wt=1.0)
print(res.params)
print(res.summary())