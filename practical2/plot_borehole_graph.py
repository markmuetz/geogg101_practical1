import datetime as dt
import dateutil.parser
import numpy as np
import pylab as plt

sim_borehole = np.loadtxt('cal_borehole_sim_obs.txt', delimiter='\t', dtype=object)
obs_borehole = np.loadtxt('Borehole_Records.txt', delimiter='\t', dtype=object)

dts  = [dateutil.parser.parse(sim_borehole[i, 0]) for i in range(len(sim_borehole))]
dts2 = [dateutil.parser.parse(obs_borehole[i, 0]) for i in range(len(obs_borehole))]
dts2 = np.array(dts2)

m = obs_borehole[:, 1].astype(float) != -999

end_date = dt.datetime(1976, 1, 1)
m2 = dts2[m] < end_date

plt.plot(dts2[m][m2], obs_borehole[:, 1][m][m2].astype(float), 'k+')
plt.plot(dts, sim_borehole[:, 1].astype(float))
plt.show()
