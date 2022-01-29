import numpy as np
import threading
import sys
from time import sleep

from TimeTagger import TimeTagStream, ChannelEdge, CustomMeasurement, createTimeTagger


# There is an issue that if the CPU idles, the transfer rates drop depending on the PC because of power-saving.
# Having one CPU at max load fixes this issue (only required for some PCs together with the Time Tagger 20).
DUMMY_THREAD = False


def probe_transfer_rate(tagger):
    # The idea is apply a higher input data rate than the USB transfer rate.
    # The internal test signal frequency is increased and 4 channels are used for the measurement.
    [tagger.setTestSignal(i, False) for i in tagger.getChannelList()]
    channels = tagger.getChannelList(ChannelEdge.Rising)[:4]
    old_test_signal_divider = tagger.getTestSignalDivider()
    if tagger.getModel() == 'Time Tagger Ultra':
        # 25 MHz per channel for TTU, default divider: 63 = 800 kHz
        tagger.setTestSignalDivider(2)
    else:
        # ~2.5 MHz per channel for TT20, default divider: 74 ~ 830 kHz
        tagger.setTestSignalDivider(25)

    tagger.setTestSignal(channels, True)

    if DUMMY_THREAD:
        def dummy_load():
            while True:
                x = 0
        t = threading.Thread(target=dummy_load, daemon=True)
        t.start()

    time_integrate = 1
    buffer_size = int(2e8)
    # Data rate is measured via the direct memory time tag stream access.
    stream = TimeTagStream(tagger, buffer_size, channels)

    tagger.sync()
    tagger.clearOverflows()
    avgs = 10
    sleep(1)
    transfer_rates = np.zeros(avgs)
    for i in range(avgs):
        stream.getData()  # clears the data
        sleep(time_integrate)
        buffer = stream.getData()

        if buffer.size >= buffer_size:
            print("Error - buffer size exceeded.")
            return False

        transfer_rates[i] = buffer.size / time_integrate

        print("test run {:2d}: transfer rate {:.1f} MTags/s".format(i+1, transfer_rates[i]/1e6))

    overflows = tagger.getOverflows()

    if overflows == 0:
        print("WARNING - input test signal rate too low.")
    transfer_rates_sorted = np.sort(transfer_rates)

    # Take the median as the data rate.
    data = {'transfer_rate': transfer_rates_sorted[avgs//2],
            'transfer_rates': transfer_rates, 'overflows': overflows}

    tagger.setTestSignalDivider(old_test_signal_divider)
    return data


if __name__ == '__main__':
    if sys.maxsize <= 2**32:
        print("This example only works with 64 bit Python shells.")
    else:
        import TimeTagger
        print("Time Tagger Software Version {}".format(TimeTagger.getVersion()))
        tagger = createTimeTagger()
        print("Model:    {}".format(tagger.getModel()))
        print("Serial:   {}".format(tagger.getSerial()))
        print("Hardware: {}".format(tagger.getPcbVersion()))
        print("\nTest Maximum Time-Tag-Stream transfer rate with four active channels.")
        result = probe_transfer_rate(tagger)
        TimeTagger.freeTimeTagger(tagger)
