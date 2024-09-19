# labtest1.py
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt
from scipy.signal import savgol_filter

np.random.seed(0)  
heart_rate_data = np.random.randint(60, 100, size=1440)
low_pass_filter=sp.signal(-np.pi, np.pi)

def compute_hourly_averages(data):
    hourly_averages = np.mean(data.reshape(-1, 60), axis=1)
    
    return hourly_averages


plt.figure(figsize=(15, 8))


plt.plot(heart_rate_data, label='Original Noisy Data', alpha=0.5)




hours = np.arange(24)

plt.xlabel('Time (minutes)')
plt.ylabel('Heart Rate (bpm)')
plt.title('Heart Rate Monitoring')
plt.legend()
plt.grid(True)
plt.show()


def identify_elevated_periods(data, threshold=100, duration=20):
    elevated_periods = []
    count = 0
    start = None

    for i, rate in enumerate(data):
        if rate > threshold:
            if count == 0:
                start = i
            count += 1
        else:
            if count >= duration:
                elevated_periods.append((start, i))
            count = 0

    if count >= duration:
        elevated_periods.append((start, len(data)))

    return elevated_periods

elevated_periods = identify_elevated_periods(heart_rate_data)


plt.figure(figsize=(15, 8))
plt.plot(heart_rate_data, label='Heart Rate Data', alpha=1.0)

for start, end in elevated_periods:
    plt.axvspan(start, end, color='red', alpha=0.3, label='Elevated Period' if start == elevated_periods[0][0] else "")

plt.xlabel('Time (minutes)')
plt.ylabel('Heart Rate (bpm)')
plt.title('Heart Rate Monitoring with Elevated Periods')
plt.legend()
plt.grid(True)
plt.show()




