#for savitzky golay
import numpy as np

range_start_value = int(input("please input the starting value of the range you want for the SG smoothing: "))
range_end_value = int(input("please input the end value of the range you want for the SG smoothing: "))
step_size = int(input("please input the step size for SG smoothing: "))

interval = np.arange(range_start_value, range_end_value, step_size)