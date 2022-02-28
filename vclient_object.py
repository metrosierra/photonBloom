#!/usr/bin/env python3

import TimeTagger
from big_tagger import BaseTag


class VirtualClient(BaseTag):

    def __init__(self):

        self.client = self.make_virtualtagger()
        super().__init__(self.client)

        print('\nVirtual Timetagger object initialising...assuming it is reading from PhotonSpot nanowire single-photon detector...will prompt about detector gain settings in a bit\n')
        methods = self.get_methods_naive()
        print('Here are the available class methods to be used in interactive mode')
        print(methods)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        TimeTagger.freeTimeTagger(self.client)
        print('Ciao!')

    def make_virtualtagger(self):

        self.client = TimeTagger.createTimeTaggerVirtual()
    
        return self.client

    def virtual_replay(self):
        pass

if __name__ == '__main__':
    with VirtualClient() as vtcee:
        print("\n\n################## With TagClient as vtcee ###################\n\n")
        import code; code.interact(local=locals())
