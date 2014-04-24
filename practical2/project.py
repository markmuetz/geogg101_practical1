#!/usr/bin/python
import argparse
import calendar
import datetime as dt
import dateutil.parser
import numpy as np
import pylab as plt
from glob import glob
from scipy import interpolate, stats

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

    return dts1, dts2, sim_borehole, obs_borehole

def load_val():
    sim_borehole = np.loadtxt('%s/cal_val/cal_val_borehole_sim.txt'%(DATA_DIR), delimiter='\t', dtype=object)
    obs_borehole = np.loadtxt('%s/obs/Borehole_Records.txt'%(DATA_DIR), delimiter='\t', dtype=object)

    sim_river = np.loadtxt('%s/cal_val/cal_val_river_flow_sim.txt'%(DATA_DIR), delimiter='\t', dtype=object)
    obs_river = np.loadtxt('%s/obs/Discharge_Records.txt'%(DATA_DIR), delimiter='\t', dtype=object)

    sim_borehole_times = np.array([dateutil.parser.parse(sim_borehole[i, 0]) for i in range(len(sim_borehole))])
    obs_borehole_times = np.array([dateutil.parser.parse(obs_borehole[i, 0]) for i in range(len(obs_borehole))])

    sim_river_times = np.array([dateutil.parser.parse(sim_river[i, 0]) for i in range(len(sim_river))])
    obs_river_times = np.array([dateutil.parser.parse(obs_river[i, 0]) for i in range(len(obs_river))])

    return {'sim_borehole_times': sim_borehole_times, 
            'obs_borehole_times': obs_borehole_times, 
            'sim_river_times': sim_river_times, 
            'obs_river_times': obs_river_times, 
            'sim_borehole': sim_borehole, 
            'sim_river': sim_river, 
            'obs_borehole': obs_borehole,
            'obs_river': obs_river
            }

def plot_val(args, res):
    sim_borehole_times = res['sim_borehole_times'] 
    obs_borehole_times = res['obs_borehole_times'] 
    sim_river_times = res['sim_river_times'] 
    obs_river_times = res['obs_river_times'] 
    sim_borehole = res['sim_borehole'] 
    sim_river = res['sim_river'] 
    obs_borehole = res['obs_borehole']
    obs_river = res['obs_river']

    borehole_names = ('Borehole 5', 'Borehole 35', 'Borehole 37', 'Borehole 65')
    river_names = ('Karup', 'Hagebro')

    start_date = dt.datetime(1970, 1, 1)
    end_date = dt.datetime(1981, 1, 1)

    graph_settings = (
            ((62, 66), np.arange(62, 67, 1)),
            ((41, 45), np.arange(41, 46, 1)),
            ((33, 37), np.arange(33, 38, 1)),
            ((25, 29), np.arange(25, 30, 1)))
    f1 = plt.figure('Borehole cal')
    f1.subplots_adjust(hspace=0.2)
    plt.ylabel('Water height (m)')
    for i in range(4):
	ax = plt.subplot(4, 1, i + 1)
        ax.set_ylabel('%s\nHeight (m)'%borehole_names[i])
	if i != 3:
	    plt.setp(ax.get_xticklabels(), visible=False)
	if i == 0:
	    plt.text(dt.datetime(1977, 1, 1), 66.5, 'Validation period')
	    plt.text(dt.datetime(1972, 1, 1), 66.5, 'Calibration period')

	m = obs_borehole[:, i + 1].astype(float) != -999
	m1 = (sim_borehole_times > start_date) & (sim_borehole_times < end_date)
	m2 = (obs_borehole_times[m] > start_date) & (obs_borehole_times[m] < end_date)

	plt.plot(sim_borehole_times[m1], sim_borehole[:, i + 1][m1].astype(float))
	plt.plot(obs_borehole_times[m][m2], obs_borehole[:, i + 1][m][m2].astype(float), 'k+')
        ax.set_yticks(graph_settings[i][1])
        plt.ylim(graph_settings[i][0])

	plt.axvline(x=dt.datetime(1971, 1, 1), color='k')
	plt.axvline(x=dt.datetime(1976, 1, 1), color='k')

        for j in range(2):
            f2 = plt.figure('borehole scatter')
            f2.subplots_adjust(hspace=0.2, wspace=0.4)
            ax = plt.subplot(2, 4, (j * 4 + 1) + i)

            if i == 0:
                if j == 0:
                    ax.set_ylabel('Calibration period')
                elif j == 1:
                    ax.set_ylabel('Validation period')

            if j == 0:
                print('cal')
                scatter_start_date = dt.datetime(1971, 1, 1)
                end_date = dt.datetime(1976, 1, 1)
                plt.title(borehole_names[i])
            else:
                print('val')
                scatter_start_date = dt.datetime(1976, 1, 1)
                end_date = dt.datetime(1981, 1, 1)

            m1 = (sim_borehole_times > scatter_start_date) & (sim_borehole_times < end_date)
            m2 = (obs_borehole_times[m] > scatter_start_date) & (obs_borehole_times[m] < end_date)

            intdate_to_sim = interpolate.interp1d([int(t.strftime('%s')) for t in sim_borehole_times[m1]], sim_borehole[:, i + 1][m1], bounds_error=False)
            x = obs_borehole[:, i + 1][m][m2].astype(float)
            y = intdate_to_sim([int(t.strftime('%s')) for t in obs_borehole_times[m][m2]])
            plt.plot(x, y, 'kx')
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

            line = [slope * graph_settings[i][0][0] + intercept, slope * graph_settings[i][0][1] + intercept]
            plt.plot(graph_settings[i][0], line, 'b-', label='R=%1.2f\nm=%1.2f\nc=%1.2f'%(r_value, slope, intercept))

            print('  %s: RMSE: %1.2f, NSE: %1.2f, R: %1.2f'%(borehole_names[i], rmse(x, y), nse(x, y), r_value))

            plt.xlim(graph_settings[i][0])
            plt.ylim(graph_settings[i][0])
            ax.set_xticks(graph_settings[i][1])
            ax.set_yticks(graph_settings[i][1])
            plt.legend(loc='lower right', frameon=False, prop={'size':10})
            #plt.text(graph_settings[i][0][0] + 0.5, graph_settings[i][0][1] - 1, 'R=%1.2f\nm=%1.2f\nc=%1.2f'%(r_value, slope, intercept), fontsize=10)

            plt.figure('Borehole cal')

    graph_settings = (
            ((3, 11), np.arange(3, 12, 1)),
            ((1.6, 3.2), np.arange(1.6, 3.3, 0.4)))
    f1 = plt.figure('River cal')
    f1.subplots_adjust(hspace=0.2)

    for i in range(2):
	ax = plt.subplot(2, 1, i + 1)
        ax.set_ylabel('%s\ndischarge (m/s)'%river_names[i])
	if i == 0:
	    plt.text(dt.datetime(1977, 1, 1), 17, 'Validation period')
	    plt.text(dt.datetime(1972, 1, 1), 17, 'Calibration period')
	    plt.setp(ax.get_xticklabels(), visible=False)

	m = obs_river[:, i + 1].astype(float) != -999
	m1 = (sim_river_times > start_date) & (sim_river_times < end_date)
	m2 = (obs_river_times[m] > start_date) & (obs_river_times[m] < end_date)

	plt.plot(sim_river_times[m1], sim_river[:, i + 1][m1].astype(float))
	plt.plot(obs_river_times[m][m2], obs_river[:, i + 1][m2].astype(float), 'k+')

	plt.axvline(x=dt.datetime(1971, 1, 1), color='k')
	plt.axvline(x=dt.datetime(1976, 1, 1), color='k')

        for j in range(2):

            f2 = plt.figure('river scatter')
            f2.subplots_adjust(hspace=0.2, wspace=0.2)
            ax = plt.subplot(2, 2, (j * 2 + 1) + i)

            if i == 0:
                if j == 0:
                    ax.set_ylabel('Calibration period')
                elif j == 1:
                    ax.set_ylabel('Validation period')

            if j == 0:
                print('cal')
                scatter_start_date = dt.datetime(1971, 1, 1)
                end_date = dt.datetime(1976, 1, 1)
                ax.set_title(river_names[i])
            else:
                print('val')
                scatter_start_date = dt.datetime(1976, 1, 1)
                end_date = dt.datetime(1981, 1, 1)

            m1 = (sim_river_times > scatter_start_date) & (sim_river_times < end_date)
            m2 = (obs_river_times[m] > scatter_start_date) & (obs_river_times[m] < end_date)

            intdate_to_sim = interpolate.interp1d([int(t.strftime('%s')) for t in sim_river_times[m1]], sim_river[:, i + 1][m1], bounds_error=False)
            x = obs_river[:, i + 1][m][m2].astype(float)
            y = intdate_to_sim([int(t.strftime('%s')) for t in obs_river_times[m][m2]])
            plt.plot(x, y, 'kx')
            slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)

            print('  %s: RMSE: %1.2f, NSE: %1.2f, R: %1.2f'%(river_names[i], rmse(x, y), nse(x, y), r_value))

            line = [slope * graph_settings[i][0][0] + intercept, slope * graph_settings[i][0][1] + intercept]
            plt.plot(graph_settings[i][0], line, 'b-', label='R=%1.2f\nm=%1.2f\nc=%1.2f'%(r_value, slope, intercept))
            #plt.plot(graph_settings[i][0], line, 'b-', label='R = %1.2f'%r_value)

            plt.xlim(graph_settings[i][0])
            plt.ylim(graph_settings[i][0])
            ax.set_xticks(graph_settings[i][1])
            ax.set_yticks(graph_settings[i][1])
            plt.legend(loc='lower right', frameon=False, prop={'size':10})

        plt.figure('River cal')

def rmse(obs, mod):
    '''Calc RMSE, will not work if len(obs) != len(mod)'''
    return np.sqrt(1./len(obs) * ((obs - mod)**2).sum())

def nse(obs, mod):
    '''Calc Nash-Sutcliffe efficiency, will not work if len(obs) != len(mod)'''
    obs_mean = obs.mean()
    return 1 - ((obs - mod)**2).sum() / ((obs - obs_mean)**2).sum()

def sim(args, ctrl, cc_only=True):
    if cc_only:
        sims = ('cc_low', 'cc_med', 'cc_hi')
    else:
        sims = ('cc_low', 'cc_med', 'cc_hi', 'cc_hi_precip_only', 'cc_hi_evap_only')

    f = plt.figure('Effects of CC')
    res = {}
    for sim_index, sim in enumerate(sims):
        res[sim] = {}
        print(sim)
        river_discharge_filename = glob('%s/sim/%s/*M11.txt'%(DATA_DIR, sim))[0]
        borehole_filename = glob('%s/sim/%s/*SZ.txt'%(DATA_DIR, sim))[0]

        river_discharge = np.loadtxt(river_discharge_filename, skiprows=3, delimiter='\t', dtype=object)
        borehole = np.loadtxt(borehole_filename, skiprows=3, delimiter='\t', dtype=object)

        res[sim]['sim_river'] = river_discharge
        res[sim]['sim_borehole'] = borehole

        if False:
            for i in range(len(river_discharge[:, 0])):
                if river_discharge[i, 0] != ctrl['sim_river'][i, 0]:
                    print('uh-oh %s'%str(river_discharge[i, 0]))

            plt.title('%s - ctrl'%(sim))

            ax = plt.subplot(6, 2, 2 + sim_index * 4)
            plt.plot(ctrl['sim_river_times'], river_discharge[:, 1].astype(float) - ctrl['sim_river'][:, 1].astype(float), 'b-', label='r1')

            ax = plt.subplot(6, 2, 4 + sim_index * 4)
            plt.plot(ctrl['sim_river_times'], river_discharge[:, 2].astype(float) - ctrl['sim_river'][:, 2].astype(float), 'g-', label='r2')

            plt.legend()

        names = ('Low', 'Medium', 'High') 
        cs = ('g', 'Orange', 'r')
        if args.regime:
            for r in range(2):
                ax = plt.subplot(2, 3, 1 + sim_index + 3 * r)
                if r == 0:
                    ax.set_title(names[sim_index])

                scen_reg = regime(args, ctrl['sim_river_times'], river_discharge)
                ctrl_reg = regime(args, ctrl['sim_river_times'], ctrl['sim_river'])
                plot_regime(args, scen_reg[r], loc='first', name=sim, ax=ax, r=r, c=cs[sim_index])
                plot_regime(args, ctrl_reg[r], loc='second', name='calibration', ax=ax, r=r)

#    if args.regime:
#        sim = 'comb'
#        hi_reg = regime(args, ctrl['sim_river_times'], res['cc_hi']['sim_river'])
#        hi_precip_scen_reg = regime(args, ctrl['sim_river_times'], res['cc_hi_precip_only']['sim_river'])
#        hi_evapor_scen_reg = regime(args, ctrl['sim_river_times'], res['cc_hi_evap_only']['sim_river'])
#        combined_reg = (hi_precip_scen_reg + hi_evapor_scen_reg) / 2
#        plot_regime(args, combined_reg, fgr_string=sim, loc='first', name='%s scenario'%sim)
#        plot_regime(args, hi_reg, fgr_string=sim, loc='second', name='%s scenario'%sim)

    return res

def regime(args, times, river_flows):
    # Monthly total for each year (10 years).
    monthly_total = np.zeros((2, 12, 10))
    for y in range(10):
        year = 1971 + y
        start_date = dt.datetime(year, 1, 1)
        end_date = dt.datetime(year, 12, 31)

        year_mask = (times >= start_date) & (times <= end_date)

        # Create empty month mask.
        month_mask = []
        for m in range(12):
            month_mask.append(np.zeros_like(times[year_mask]).astype(bool))

        # Fill it in for this year.
        for i, t in enumerate(times[year_mask]):
            for j in range(12):
                if t.month == j + 1:
                    month_mask[j][i] = True

        for m in range(12):
            for r in range(2):
                river_flow = river_flows[:, r + 1].astype(float)
                monthly_total[r][m, y] = river_flow[year_mask][month_mask[m]].mean()\
                                         * calendar.monthrange(year, m + 1)[1]

    return monthly_total

def plot_regime(args, monthly_total, loc='only', name='', ax=None, r=0, c=''):
    months = [calendar.month_name[i + 1][:1] for i in range(12)]

    graph_settings = ((120, 230), (55, 85))

    if True:
        if args.bar:
            if loc == 'only':
                plt.bar(range(12), monthly_total.mean(axis=1), width=1, color='b', 
                        yerr=monthly_total.std(axis=1), error_kw=dict(ecolor='k'))
            elif loc == 'first':
                plt.bar(np.arange(12) + 0.1, monthly_total.mean(axis=1), width=0.4, color='b', 
                        yerr=monthly_total.std(axis=1), error_kw=dict(ecolor='k'))
            elif loc == 'second':
                plt.bar(np.arange(12) + 0.5, monthly_total.mean(axis=1), width=0.4, color='g', 
                        yerr=monthly_total.std(axis=1), error_kw=dict(ecolor='k'))
        else:
            if loc == 'only':
                plt.plot(np.arange(12) + 0.5, monthly_total.mean(axis=1), 'b-')
            elif loc == 'first':
                plt.plot(np.arange(12) + 0.5, monthly_total.mean(axis=1), '-', color=c, label=name)
            elif loc == 'second':
                plt.plot(np.arange(12) + 0.5, monthly_total.mean(axis=1), 'k--', label=name)

        if name in ['simulated', 'observed']:
            plt.legend(loc='best')
        ax.set_xticks(np.arange(12) + 1./2)
        ax.set_xticklabels(months)
        plt.ylim(graph_settings[r])
        plt.xlim((0, 12))
    if name in ['cc_low', 'observed'] and r == 0:
        plt.ylabel('Karup average monthly\ndischarge (cumecs)')
    elif name in ['cc_low', 'observed']:
        plt.ylabel('Hagebro average monthly\ndischarge (cumecs)')

    if False:
        # Plot each year's monthly discharge.
        for y in range(10):
            plt.plot(np.arange(12) + 1./2, monthly_total[:, y], label='y:%i'%(y + 1971))
        #plt.legend()


def create_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cal', action='store_true')
    parser.add_argument('-v', '--val', action='store_true')
    parser.add_argument('-s', '--sim', action='store_true')
    parser.add_argument('-r', '--regime', action='store_true')
    parser.add_argument('-b', '--bar', action='store_true')
    return parser.parse_args()

def main(args):
    res = {}
    res['val'] = load_val()

    if args.cal:
	res['cal'] = cal()
    if args.val:
	plot_val(args, res['val'])
    if args.sim:
	res['sim'] = sim(args, res['val'])
    if args.regime:
	res['sim_reg'] = regime(args, res['val']['sim_river_times'], res['val']['sim_river'])
	res['obs_reg'] = regime(args, res['val']['obs_river_times'], res['val']['obs_river'])
        f = plt.figure('cal regime')
        for r in range(2):
            ax = plt.subplot(2, 1, r + 1)
            plot_regime(args, res['obs_reg'][r], loc='first', name='observed', ax=ax, c='b', r=r)
            plot_regime(args, res['sim_reg'][r], loc='second', name='simulated', ax=ax, c='k', r=r)
    plt.show()
    return res

if __name__ == "__main__":
    args = create_args()
    main(args)
