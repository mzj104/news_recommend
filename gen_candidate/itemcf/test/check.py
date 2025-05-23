import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

data = sio.loadmat("trend_12.mat")
data1 = np.squeeze(data["curve"])
print(data1[100])

data = sio.loadmat("trend_null.mat")
data2 = np.squeeze(data["curve"])

data = sio.loadmat("trend_2.mat")
data3 = np.squeeze(data["curve"])

data = sio.loadmat("trend_6.mat")
data4 = np.squeeze(data["curve"])

data = sio.loadmat("trend_12_05.mat")
data5 = np.squeeze(data["curve"])

data = sio.loadmat("trend_12_03.mat")
data6 = np.squeeze(data["curve"])

plt.plot(data1)
plt.plot(data2)
plt.plot(data3)
plt.plot(data4)
plt.plot(data5)
plt.plot(data6)
plt.legend(["12","null","2","6","12_0.5","12_0.3"])
plt.show()