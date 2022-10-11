#!/usr/bin/env python3

import TimeTagger
from big_tagger import BaseTag

class TagClient(BaseTag):
    def __init__(self, ip_address = '192.168.0.2', **kwargs):

        self.target_ip = ip_address
        print("Search for Time Taggers on the network...")
        # Use the scanTimeTaggerServers() function to search for Time Tagger servers in the local network
        self.serverlist = TimeTagger.scanTimeTaggerServers()

        print("{} servers found.".format(len(self.serverlist)))
        print(self.serverlist)

        self.client = self.server_handshake()

        super().__init__(self.client, **kwargs)
        print('\nTimetagger object initialising...assuming it is reading from PhotonSpot nanowire single-photon detector...will prompt about detector gain settings in a bit\n')
        self.get_methods_naive()
        # print('Here are the available class methods to be used in interactive mode')
        self.welcome_message()

    def __enter__(self):
        return self

########################### initialisation/info grabbing methods #################################

    def server_handshake(self):
        try:
            self.target_server_info = TimeTagger.getTimeTaggerServerInfo('{}'.format(self.target_ip))
            # print('Information about Time Tagger server on {}:'.format(self.target_ip))
            # print(self.target_server_info)

        except RuntimeError:
            raise Exception('No Time Tagger server available on {} and the default port 41101.'.format(self.target_ip))

        self.client = TimeTagger.createTimeTaggerNetwork(self.target_ip)
        print('\nConnecting to the server on localhost.')
        print('Server handshake successful!!!!!!!!!!!\n')
        self.get_info()

        return self.client


if __name__ == '__main__':
    with TagClient() as tcee:
        print("\n\n################## With TagClient as tcee ###################\n\n")
        import code; code.interact(local=locals())
