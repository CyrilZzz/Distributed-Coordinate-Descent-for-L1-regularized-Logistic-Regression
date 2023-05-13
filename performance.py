from model import DistributedLassoLogReg
import pandas as pd
import time

data = pd.read_csv('dataset.csv')

# We try different numbers of partitions and measure the training time
for i in range(1, 17, 3):
    model = DistributedLassoLogReg(i)
    model.preprocessing(data, 'TenYearCHD')
    start_time = time.time()
    model.fit('train_data.csv')
    end_time = time.time()
    print(f"Elapsed time ({i}): {end_time-start_time} seconds")
    print(model.beta)
    model.predict('test_data.csv')
