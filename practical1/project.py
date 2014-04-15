import re
import argparse
import glob

import numpy as np
import pylab as plt
from matplotlib.ticker import FuncFormatter

OBS_FILENAME = 'data_files/calibration-files/fowey_observed_data.txt'
MODEL_CAL_DIR = 'raw_results/calibration'
DEMO_N = '0.025'

def load_all_cal_data():
    '''Loads in all calibrated data and stores manning n with each dataset'''
    cal_data_files = glob.glob('%s/model*.txt'%MODEL_CAL_DIR)
    all_cal_data = []
    for cal_data_file in sorted(cal_data_files):
	n = cal_data_file.split('/')[-1].split('_')[1][1:]
	cal_data = load_cal_data(cal_data_file)
	all_cal_data.append((n, cal_data))
    return all_cal_data

def load_cal_data(filename):
    '''Loads in individual calibrated dataset'''
    f = open(filename)
    lines = f.readlines()
    split_lines = [re.split('\t|,', l.strip().replace(' ', '')) for l in lines]
    cal_data = np.array(split_lines).astype('float')
    return cal_data

def calibrate_data(args):
    '''Calibrate modelled data against observed data'''
    print 'calibrating data'

    obs = np.loadtxt(OBS_FILENAME, delimiter=',')
    obs_vel_mag     = obs[:, 1]
    obs_water_level = obs[:, 3]
    all_cal_data = load_all_cal_data()

    # Build some data structures to hold results.
    tests = [rmse, nse]
    variables = [('Velocity magnitude', 1), ('Water level', 3)]

    ns = []
    vals = {}
    for var_name, var_col in variables:
	vals[var_name] = {}
	for test in tests:
	    vals[var_name][test.func_name] = []

    # Perform the tests.
    for n, cal_data in all_cal_data:
	print 'n', n

	if n == DEMO_N:
	    print 'plotting data for %s'%n
	    # TODO:
	    #fig = plt.figure()
	    #ax1 = fig.add_subplot(121)
	    #ax2 = fig.add_subplot(122)
	    #ax1.plot(


	ns.append(float(n))
	for var_name, var_col in variables:
	    var_cal_data = cal_data[:, var_col]
	    var_obs = obs[:, var_col]
	    for test in tests:
		# Give the model time to settle down before comparing.
		# First 48 timesteps skipped.
		test_value = test(var_obs[48:], var_cal_data[48:])
		print '  %s:'%var_name, test.func_name, test_value
		vals[var_name][test.func_name].append(test_value)
    
    def fmt(x, pos):
        return '%1.2f'%x

    graph_formats = {
            'Velocity magnitude': {'rmse': (np.arange( 0.04, 0.081, 0.01), 0.035, 0.085),
                        'nse':  (np.arange( 0.86, 0.961, 0.02), 0.85, 0.97)},
            'Water level': {'rmse': (np.arange( 0.03, 0.041, 0.002), 0.029, 0.041),
                            'nse':  (np.arange( 0.995, 0.9971, 0.001), 0.9945, 0.9975)}}


    formatter = FuncFormatter(fmt)
    if args.plot:
	# Plot the results.
        f = plt.figure(1)
        f.subplots_adjust(hspace=0.3)
        for i, test in enumerate(tests):
            for j, (var_name, var_col) in enumerate(variables):
		test_name = test.func_name
                graph_fmt = graph_formats[var_name][test_name]
                ax = plt.subplot('22%i'%(i + 2*j + 1))
                ax.xaxis.set_major_formatter(formatter)
                plt.yticks(graph_fmt[0])
                if i == 1:
                    ax.yaxis.tick_right()
                    ax.yaxis.set_label_position('right')

                plt.title(var_name)
                if j == 1:
                    plt.xlabel('Bed friction coefficient (n)')
                else:
                    #plt.setp(ax.get_xticklabels(), visible=False)
                    pass

		plt.ylabel(test_name.upper())
		plt.plot(ns, vals[var_name][test_name], label=var_name)
                plt.plot(ns[3], vals[var_name][test_name][3], 'bo')
                #plt.legend()
                plt.ylim([graph_fmt[1], graph_fmt[2]])
        plt.show()

    return obs, all_cal_data

def load_dredging_data():
    raw_data = np.loadtxt('data_files/dredging_effect_with_nice_headers.txt', delimiter=',')
    time = raw_data[:, 0]
    data = raw_data[:, 1::2]
    cols = np.loadtxt('data_files/dredging_effect_column_headers.txt', dtype=object, delimiter=',')
    return time, cols, data

def plot_dredge_data(time, cols, data):
    for offset in range(4):
	#print('%s\n%s'%(cols[offset], cols[offset + 8]))
        plt.title('%s velocity magnitude (m/s)'%(cols[offset].split(' - ')[0]))
	d1 = data[:, offset]
	d2 = data[:, offset + 8]
	plt.plot(d1, label='undredged')
	plt.plot(d2, label='dredged')
        plt.legend(loc='best')
	print(rmse(d1, d2))
	plt.show()

    for offset in [2, 3]:
	#print('%s\n%s'%(cols[offset], cols[offset + 8]))
        plt.title('%s delta velocity magnitude (m/s)'%(cols[offset].split(' - ')[0]))
	d1 = data[:, offset]
	d2 = data[:, offset + 8]
	plt.plot(d1 - d2)
	print(rmse(d1, d2))
	plt.show()

    for offset in range(4):
	#print('%s\n%s'%(cols[offset + 4], cols[offset + 12]))
        plt.title('%s surface water elevation'%(cols[offset].split(' - ')[0]))
	d1 = data[:, offset + 4]
	d2 = data[:, offset + 12]
	plt.plot(d1, label='undredged')
	plt.plot(d2, label='dredged')
        plt.legend(loc='best')
	print(rmse(d1, d2))
	plt.show()

def rmse(obs, mod):
    '''Calc RMSE, will not work if len(obs) != len(mod)'''
    return np.sqrt(1./len(obs) * ((obs - mod)**2).sum())

def nse(obs, mod):
    '''Calc Nash-Sutcliffe efficiency, will not work if len(obs) != len(mod)'''
    obs_mean = obs.mean()
    return 1 - ((obs - mod)**2).sum() / ((obs - obs_mean)**2).sum()

def model_scenario(args):
    print 'modelling scenario'
    time, cols, dredge_data = load_dredging_data()
    if args.plot:
	plot_dredge_data(time, cols, dredge_data)
    return {'time': time, 'cols': cols, 'dredge_data': dredge_data}

def run_project(args):
    '''Entry point'''
    res = {}
    if args.cal:
	res['cal'] = calibrate_data(args)
    if args.mod:
	res['mod'] = model_scenario(args)
    return res

def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cal', action='store_true')
    parser.add_argument('-m', '--mod', action='store_true')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    run_project(create_args())
