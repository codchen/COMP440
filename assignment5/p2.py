import numpy as np

ro1 = [0.8684, 0.1078, 0.0238, 0]
ro2 = [0.3922, 0.3856, 0, 0.2222]
ro3 = [0.4762, 0, 0.1164, 0.4074]
ro4 = [0, 0.2778, 0.0926, 0.6296]
Q = np.asarray([ro1,ro2,ro3,ro4])
Ns = [1000,5000,10000]
for N in Ns:
	start = np.asarray([1,0,0,0])
	for i in range(N):
		start = start.T.dot(Q)
	print(start)