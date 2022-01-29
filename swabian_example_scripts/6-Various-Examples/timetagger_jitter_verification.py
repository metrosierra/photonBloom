"""In this example, we measure the jitter/resolution of the Time Tagger channels with the built-in test signals
and compare the results to the specifications of the Time Tagger, which is connected.
"""
import TimeTagger
import numpy as np
import matplotlib.pyplot as plt
import time

# These are the specified RMS jitters for the different editions and options of the Time Tagger
jitter_specs_rms = {'Time Tagger 20 Performance': 34, ' Time Tagger Ultra Performance': 9,
                    'Time Tagger Ultra Value': 42, 'Time Tagger 20 Value': 790,
                    'Standard': 9, 'HighResA': 7, 'HighResB': 5, 'HighResC': 4}
res_modes = {'Standard': TimeTagger.Resolution.Standard, 'HighResA': TimeTagger.Resolution.HighResA,
             'HighResB': TimeTagger.Resolution.HighResB, 'HighResC': TimeTagger.Resolution.HighResC}


def gaussian(x, mu, sigma, A):
    """
    Returns a scaled Gaussian function for visual comparison of the measured data
    """
    return A/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*(x-mu)**2/sigma**2)


def synchronized_correlation_measurement(tagger, channels, binwidth=1, bins=int(5e5), duration=int(30e12)):
    """
    For the jitter measurement, we use correlation measurements of the periodic built-in test signal.
    The function initializes multiple correlation measurements between channels.
    To have a simultaneous start, we make use of the Synchronized Measurement class.
    Returns indices, data, and measured channels
    """
    # Instantiate helper class to synchronize measurements
    sync_measurement = TimeTagger.SynchronizedMeasurements(tagger)
    correlation_measurements = []
    measured_channels = np.array([], dtype=object)
    # Here we use a single channel, which is correlated to all other channels.
    start_channel = channels[0]
    for stop_channel in channels[1:]:
        if stop_channel - start_channel >= 100:  # In case synchronizer is used, update the start channel to new device.
            start_channel = stop_channel
        else:
            # Initiate the measurements and register them for the synchronized measurement
            correlation_measurements.append(TimeTagger.Correlation(tagger, start_channel, stop_channel, binwidth, bins))
            sync_measurement.registerMeasurement(correlation_measurements[-1])
            # To keep track of the measured correlations
            measured_channels = np.append(measured_channels, '{:2d} - {:2d}'.format(start_channel, stop_channel))
    # Starting the measurement
    sync_measurement.startFor(duration)
    sync_measurement.waitUntilFinished()
    data = []
    indices = []
    for cor_measurement in correlation_measurements:
        # Retrieving the data from the Time Tagger
        indices.append(cor_measurement.getIndex())
        data.append(cor_measurement.getData())
    return indices, data, measured_channels


def warmup(tagger, duration=int(30e12)):
    """
    Function to warm-up the Time Tagger until temperatures are stable (Time Tagger Ultra)
    or wait for the specified duration (Time Tagger 20).
    """
    all_channels = tagger.getChannelList(TimeTagger.ChannelEdge.Rising)
    tagger.setTestSignal(all_channels, True)
    if tagger.getModel() == "Time Tagger 20":
        print("Please wait for {} seconds".format(duration/1e12))
        synchronized_correlation_measurement(tagger, all_channels, duration=duration)
        print("The Time Tagger is now warmed up.")
    else:
        print("For the Time Tagger Ultra, we can check for stable temperatures on the board.")
        warmed_up = False
        cnt = 0
        while not warmed_up:
            pcb_temperatures = []
            fpga_temperatures = []
            # Measure 10 times, then compare if temperature has been stable. If not, repeat.
            for i in range(10):
                synchronized_correlation_measurement(tagger, all_channels, duration=int(1e12))
                cnt = cnt + 1
                sensor_data = tagger.getSensorData()
                pcb = sensor_data[0]['FPGA board']['Board Temp #1']
                fpga = sensor_data[0]['FPGA board']['FPGA Temp']
                pcb_temperatures.append(pcb)
                fpga_temperatures.append(fpga)
                print("t: {:3d} s, Board temperature: {:2.1f} °C, FPGA temperature: {:2.1f} °C".format(cnt, pcb, fpga))
            if ((max(pcb_temperatures) - min(pcb_temperatures)) < 0.5
                    and (max(fpga_temperatures) - min(fpga_temperatures)) < 1.0):
                warmed_up = True
                print("The Time Tagger is warmed up.")
                print("")
            else:
                print("The Time Tagger is not yet warmed up.")


def get_jitter_and_delay(index, data):
    """
    Calculate RMS jitter and mean (giving the internal delay of the test signal to later shift the plots)
    Two channels contribute to the jitter. Yet, we are interested in the RMS jitter of ONE channel.
    Hence we adjust by a factor of sqrt(2).
    """
    mean = np.round(np.average(index, weights=data), decimals=1)
    std = np.round(np.sqrt(np.average((index-mean)**2, weights=data))/np.sqrt(2), decimals=1)
    return std, mean


def create_time_tagger_and_get_infos(serial=''):
    """
    Creates the Time Tagger object and retrieves the information about the device. By explicitly specifying a serial
    number a specific Time Tagger can be chosen, in case more than one device is connected and not already taken care
    of by a synchronizer.
    """
    tagger = TimeTagger.createTimeTagger(serial)
    model = tagger.getModel()
    if model not in ['Time Tagger 20', 'Time Tagger Ultra']:
        raise ValueError('Device currently not supported')
    license_info = tagger.getLicenseInfo().split()
    edition = license_info[license_info.index('Edition:')+1]
    if edition == 'High-Res':
        modes = res_modes.keys()
    else:
        modes = [model + ' ' + edition]  # Only single mode available  # Only single mode available
    print('Connected to a ' + model + ' ' + edition)
    print('Serial: {}'.format(tagger.getSerial()))
    return tagger, model, edition, modes


# Searching for available Time Taggers
serials = TimeTagger.scanTimeTagger()
print('Found {} connected device(s)'.format(len(serials)))
tagger, model, edition, modes = create_time_tagger_and_get_infos()
time.sleep(0.5)
print('To properly analyze the jitter, the TimeTagger has to be warmed up.')
warmup(tagger)

# Storing our data in lists to account for the different modes of the HighRes option.
measured_jitters = []
within_specs = []
measured_channels = []

# We loop over the different HighRes modes, if available. Or only use the single available mode.
for i, mode in enumerate(modes):
    if edition == 'High-Res':
        print('Setting the Time Tagger into {} mode'.format(mode))
        # We need to first free the Time Tagger to initiate it again with a different HighRes modes
        TimeTagger.freeTimeTagger(tagger)
        tagger = TimeTagger.createTimeTagger(resolution=res_modes[mode])
    print('Single channel RMS jitter is specified with {} ps'.format(jitter_specs_rms[mode]))
    if 'HighRes' in mode:
        channels_available = tagger.getChannelList(TimeTagger.ChannelEdge.HighResRising)
    else:
        channels_available = tagger.getChannelList(TimeTagger.ChannelEdge.Rising)
    print('The available channel numbers are {}'.format(channels_available))
    tagger.setTestSignal(channels_available, True)  # Reactivating the test signals
    print('Measuring for 30 seconds')
    # Retrieving the measured data
    indices, data, meas_chan = synchronized_correlation_measurement(tagger, channels_available, duration=int(30e12))
    measured_channels.append(meas_chan)
    measured_jitters.append(np.full(len(meas_chan), '', dtype=object))
    within_specs.append(np.full(len(meas_chan), '', dtype=object))
    print('Measurement complete.\nNow evaluating the data')
    time.sleep(1)

    fig, ax = plt.subplots()  # Create a plot to visualize the TWO channel jitter
    # Looping over the measurement data, evaluating the single channel RMS jitter from it and displaying the results
    # For the visual comparison a Gaussian with standard deviation = sqrt(2)*specified_RMS_jitter is used
    # since we look at the two-channel jitter
    for j, (ind, dat) in enumerate(zip(indices, data)):
        std, mean = get_jitter_and_delay(ind, dat)
        measured_jitters[i][j] = std
        within_specs[i][j] = std < jitter_specs_rms[mode]
        print('channel numbers ' + measured_channels[i][j] +
              ': measured single channel RMS jitter: {} ps, within specifications: {}'.format(std, within_specs[i][j]))
        if j == len(data)-1:  # make only label for last curve to not clutter the plot.
            label = 'measured jitter'
        else:
            label = None
        ax.plot(ind-mean, dat/1e3, label=label)
    ax.set_xlim((-jitter_specs_rms[mode]*5, jitter_specs_rms[mode]*5))
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('kCounts')
    ax.plot(ind, gaussian(ind, 0, jitter_specs_rms[mode]*np.sqrt(2), np.sum(data)/len(data)/1e3),
            color='k', ls='--', label='specified jitter')
    ax.set_title('Visual comparison of the measured two-channel jitters to specifications')
    ax.legend(loc=1)
    plt.show()
    print('\n')

print('Plotting a summary of the results')

# Summary of the measured RMS jitters for a single channel in a table
colLabels = ['single channel\nRMS jitter (ps)', 'within\nspecifications']
fig, axes = plt.subplots(1, len(modes), figsize=(len(modes)*7, len(measured_channels[0])/1.6+0.7))
axes = np.atleast_1d(axes)
for i in np.arange(len(modes)):
    content = np.hstack((measured_jitters[i].reshape(-1, 1), within_specs[i].reshape(-1, 1)))
    colors = np.full_like(content, 'white')
    colors[within_specs[i] == True, 1] = 'C2'
    colors[within_specs[i] == False, 1] = 'C3'
    the_table = axes[i].table(cellText=content, colLabels=colLabels, rowLabels=measured_channels[i],
                              loc='center', cellColours=colors, cellLoc='center')
    the_table.scale(0.78, 2.2)
    axes[i].axis('tight')
    axes[i].axis('off')
    if edition == 'High-Res':
        axes[i].set_title(list(res_modes.keys())[i], size=12, pad=7)
    else:
        axes[i].set_title(model + ' ' + edition, size=12, pad=7)
axes[0].text(-0.05, 0.5, 'channel combination', rotation=90, transform=axes[0].transAxes, va='center', size=12)
if edition == 'High-Res':
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
plt.show()

TimeTagger.freeTimeTagger(tagger)  # Freeing the Time Tagger again so that it can be used in another application
