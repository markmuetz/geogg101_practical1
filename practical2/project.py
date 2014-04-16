import argparse
import datetime as dt
import dateutil.parser
import numpy as np
import pylab as plt

DATA_DIR = 'data_files'

def cal():
    sim_borehole = np.loadtxt('%s/cal_val/cal_borehole_sim.txt'%(DATA_DIR), delimiter='\t', dtype=object)
    obs_borehole = np.loadtxt('%s/obs/Borehole_Records.txt'%(DATA_DIR), delimiter='\t', dtype=object)

    dts1  = [dateutil.parser.parse(sim_borehole[i, 0]) for i in range(len(sim_borehole))]
    dts2 = [dateutil.parser.parse(obs_borehole[i, 0]) for i in range(len(obs_borehole))]
    dts2 = np.array(dts2)

    for i in range(4):
	m = obs_borehole[:, i + 1].astype(float) != -999
	end_date = dt.datetime(1976, 1, 1)
	m2 = dts2[m] < end_date

	plt.title('cal')
	plt.plot(dts2[m][m2], obs_borehole[:, i + 1][m][m2].astype(float), 'k+')
	plt.plot(dts1, sim_borehole[:, i + 1].astype(float))
    #plt.show()

    return dts1, dts2, sim_borehole, obs_borehole

def val():
    start_date = dt.datetime(1970, 1, 1)
    end_date = dt.datetime(1981, 1, 1)

    sim_borehole = np.loadtxt('%s/cal_val/cal_val_borehole_sim.txt'%(DATA_DIR), delimiter='\t', dtype=object)
    obs_borehole = np.loadtxt('%s/obs/Borehole_Records.txt'%(DATA_DIR), delimiter='\t', dtype=object)

    dts1  = np.array([dateutil.parser.parse(sim_borehole[i, 0]) for i in range(len(sim_borehole))])
    dts2 = np.array([dateutil.parser.parse(obs_borehole[i, 0]) for i in range(len(obs_borehole))])

    f1 = plt.figure(1)
    f1.subplots_adjust(hspace=0)
    for i in range(4):
	ax = plt.subplot(4, 1, i + 1)
	if i != 3:
	    plt.setp(ax.get_xticklabels(), visible=False)
	if i == 0:
	    plt.text(dt.datetime(1977, 1, 1), 68, 'Validation period')
	    plt.text(dt.datetime(1972, 1, 1), 68, 'Calibration period')

	m = obs_borehole[:, i + 1].astype(float) != -999
	m1 = (dts1 > start_date) & (dts1 < end_date)
	m2 = (dts2[m] > start_date) & (dts2[m] < end_date)

	plt.plot(dts1[m1], sim_borehole[:, i + 1][m1].astype(float))
	plt.plot(dts2[m][m2], obs_borehole[:, i + 1][m][m2].astype(float), 'k+')

	plt.axvline(x=dt.datetime(1971, 1, 1), color='k')
	plt.axvline(x=dt.datetime(1976, 1, 1), color='k')
    #plt.show()

    return dts1, dts2, sim_borehole, obs_borehole

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cal', action='store_true')
    parser.add_argument('-v', '--val', action='store_true')
    return parser.parse_args()

def main(args):
    res = {}
    if args.cal:
	res['cal'] = cal()
    if args.val:
	res['val'] = val()
    plt.show()
    return res

if __name__ == "__main__":
    args = create_args()
    main(args)
