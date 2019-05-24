import pandas as pd
import numpy as np
from pathlib import Path
import requests

def fetch_and_cache(data_url, file, data_dir="data", force = False):
	"""
	(Credit: John DeNero)
	Download and cache a url and return the file object.

	Dependent: 	from pathlib import Path
				import requests
    
    data_url: the web address to download
    file: the file in which to save the results.
    data_dir: (default="data") the location to save the data
    force: if true the file is always re-downloaded 
    
    return: The pathlib.Path to the file.
	"""
	data_dir = Path(data_dir)
	data_dir.mkdir(exist_ok=True)
	file_path = data_dir/Path(file)
	if force and file_path.exists():
		file_path.unlink()
	if force or not file_path.exists():
		print('Downloading...', end = ' ')
		resp = requests.get(data_url)
		with file_path.open('wb') as f:
			f.write(resp.content)
		print('Done!')
	else:
		import time
		created = time.ctime(file_path.stat().st_ctime)
		print("Using cached version downloaded at", created)
	return file_path

def remove_outliers(data, variable, lower=-np.inf, upper=np.inf):
    """
    Input:
      data (data frame): the table to be filtered
      variable (string): the column with numerical outliers
      lower (numeric): observations with values lower than this will be removed
      upper (numeric): observations with values higher than this will be removed
    
    Output:
      a winsorized data frame with outliers removed
      
    Note: This function should not change mutate the contents of data.
    """  
    return data.loc[((data[variable] > lower) & (data[variable] < upper))]


def sample(series , n, replacement = False):
	""" Sample from series n times. 
	Dependent: import numpy as np
				import pandas as pd

	series: Series to be sampled from.
	n: Number of times to draw from series.
	replacement: (default False) Boolean, sample with replacement.

	return: pd.Series of samples drawn from series
	"""
	assert isinstance(series, pd.Series)
	return np.random.choice(series, n, replace = replacement).tolist()


def multi_sample(df, on, ns , return_row = False):
    """ Conduct multistage sample (SRS of Clusters, then SRS within clusters) on DataFrame df.
	Dependent: import numpy as np
                import pandas as pd
                def sample(series , n, replacement = False):
                    return np.random.choice(series, n, replace = replacement).tolist()

    df: DataFrame to sample from
    on: Row name to groupby
    return_row: Row(s) to be included in final result (row name or list of row names).
                if false, all row names will be included.
    ns: tuple of form (first_stage_n, second_stage_n) where n is number of samples drawn
    
    return: pd.DataFrame of samples drawn from df
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(ns, tuple)
    if not return_row:
        return_row = list(df.columns.values)
    
    n_1, n_2 = ns[0], ns[1]
    first_stage = df.loc[bus[on].isin(sample(bus[on].unique() , n_1))]
    
    multi_sample_result = pd.DataFrame()
    
    for name, group in first_stage.groupby(on):
        multi_sample_result[name] = sample(group[return_row], n = n_2)
        
    return multi_sample_result

def count_null(series):
	""" Count null entries in a series
	Dependent: import pandas as pd

	series: series to be counted.

	return: integer number of null values.
	"""
	assert isinstance(series, pd.Series)
	return len(series.loc[series.isna()])

def train_test_split(data, percent_train = .8):
	""" Randomly split data into train and test groups 
	with percent_train % of data in training group.

	Dependent: import numpy as np
			   import pandas as pd

	Inputs:
			data: pandas dataframe of data. 
			percent_train: A percentage of data to put in training group

	Returns: train, test - pandas DataFrames. 
	"""
	data = data.copy()

	data_len = len(data)
	if percent_train > 1:
		percent_train = percent_train/100

	assert (0 <= percent_train & percent_train <= 1), "percent_train must be convertable to percentage."

	shuffled_indices = np.random.permutation(data_len)
	split = int(data_len * train_float)

	train_indices = shuffled_indices[:split]
	test_indices = shuffled_indices[split:]

	train = data.iloc[train_indices]
	test = data.iloc[test_indices]

	return train, test

