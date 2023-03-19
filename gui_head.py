#!/usr/bin/env python3

import PySimpleGUI as sg
from datetime import datetime

from datalogger_mark2 import Lotus
#for screen resolution detection
import ctypes
import platform

class Hydrangea():
        
    def __init__(self):
        # checking if the OS is windows and making sure the GUI resolution matches
        try:
            if int(platform.release()) >= 8:
                ctypes.windll.shcore.SetProcessDpiAwareness(True)
        except:
            print('This is not a windows system?')


        print('Hey there, I am Hydrangea, your friendly nanowiresinglephotonheraldedthermaltriangleresonatorg2 assistant')
        print('I wrap the datalogger class Lotus as a GUI to provide a more user-friendly interface')
        print('Please change how I look however you wish, I -will- be judging your choice of aesthetics')
        
        self.datalogger = Lotus()
        print('PhotonBloom datalogger class Lotus initialised')
        sg.theme('Light Brown 3')


    def __enter__(self):
        return self

    # some exiting housekeeping
    def __exit__(self, exc_type, exc_value, exc_traceback):
        print('__exit__ called')
        if exc_type:
            self.datalogger.__exit__()
            print(f'exc_type: {exc_type}')
            print(f'exc_value: {exc_value}')
            print(f'exc_traceback: {exc_traceback}')

            now = datetime.now()
            log = open('photonbloomgui_errorlog.txt', 'a+')
            separator = '\n{}\n'.format(''.join('#' for i in range(10)))
            info = 'Datetime = {}\n'.format(now.strftime("%d/%m/%Y, %H:%M:%S")) + f'exc_type: {exc_type}\n' + f'exc_value: {exc_value}\n' + f'exc_traceback: {exc_traceback}\n'
            log.write(separator + info)
            log.close()


    def make_window(self, theme):
        sg.theme(theme)
        menu_def = [['&Application', ['&Exit']],
                    ['&Help', ['&About']] ]
        right_click_menu_def = [[], ['Versions', 'Exit']]
        graph_right_click_menu_def = [[], ['Erase','Draw Line', 'Draw',['Circle', 'Rectangle', 'Image'], 'Exit']]



        basics_layout = [ [sg.Text('Counter CH'), 
                          sg.Checkbox('1', default=True, k='CB1'), sg.Checkbox('2', default=True, k='CB2'),
                          sg.Checkbox('3', default=True, k='CB3'), sg.Checkbox('4', default=True, k='CB4'),
                          sg.Checkbox('-1', default=False, k='CB-1'), sg.Checkbox('-2', default=False, k='CB-2'),
                          sg.Checkbox('-3', default=False, k='CB-3'), sg.Checkbox('-4', default=False, k='CB-4'),
                          sg.Text('Binwidth (ns)'), sg.Input(key = '-TAGCOUNTER_BINW-', size = (5, 1), default_text = '1e8'),
                          sg.Text('N_bins'), sg.Input(key = '-TAGCOUNTER_BINNO-', size = (5, 1), default_text = '100'), sg.Button('Start Tag Counter', bind_return_key = False)],

                        ####################################################
                          [sg.Text('XCorr CH'), sg.Input(key = '-TAGXCORR_CH1-', size = (4, 1), default_text = '3'),
                          sg.Text('XCorr CH'), sg.Input(key = '-TAGXCORR_CH2-', size = (4, 1), default_text = '4'),
                          sg.Text('Binwidth (ns)'), sg.Input(key = '-TAGXCORR_BINW-', size = (8, 1), default_text = '10'), sg.Text('N_bins'), sg.Input(key = '-TAGXCORR_BINNO-', size = (8, 1), default_text = '6000'), sg.Button('Start Tag XCorr', bind_return_key = False)]
                        ]

        advanced_layout = [ 
                          [sg.Text('TrigCorr TrigCH'), sg.Input(key = '-TAGTRIGCORR_TRIGCH-', size = (8, 1), default_text = '1'),
                          sg.Text('TrigCorr XCH'), sg.Input(key = '-TAGTRIGCORR_XCH1-', size = (8, 1), default_text = '3'),
                          sg.Text('TrigCorr XCH'), sg.Input(key = '-TAGTRIGCORR_XCH2-', size = (8, 1), default_text = '4')],

                          [sg.Text('Binwidth (ns)'), sg.Input(key = '-TAGTRIGCORR_BINW-', size = (8, 1), default_text = '30'), 
                           sg.Text('N_bins'), sg.Input(key = '-TAGTRIGCORR_BINNO-', size = (8, 1), default_text = '6000'), 
                           sg.Text('N_stacks'), sg.Input(key = '-TAGTRIGCORR_STACKS-', size = (8, 1), default_text = '40'), 
                           sg.Button('Start Tag XCorr', bind_return_key = False)],

                        [sg.Sizer(h_pixels = 100, v_pixels = 25)],
                        ####################################################
                          [sg.Text('SweepCorr TrigCH'), sg.Input(key = '-TAGSWEEPCORR_TRIGCH-', size = (8, 1), default_text = '2'),
                           sg.Text('SweepCorr XCH'), sg.Input(key = '-TAGSWEEPCORR_XCH1-', size = (8, 1), default_text = '3'),
                           sg.Text('SweepCorr XCH'), sg.Input(key = '-TAGSWEEPCORR_XCH2-', size = (8, 1), default_text = '4')],

                          [sg.Text('Binwidth (ns)'), sg.Input(key = '-TAGSWEEPCORR_BINW-', size = (8, 1), default_text = '2'), 
                           sg.Text('N_bins'), sg.Input(key = '-TAGSWEEPCORR_BINNO-', size = (8, 1), default_text = '6000'), 
                           sg.Text('N_steps'), sg.Input(key = '-TAGSWEEPCORR_STEPS-', size = (8, 1), default_text = '5'), 
                           sg.Button('Start Tag SweepCorr', bind_return_key = False)]
                        ]

##### main messages monitor window, very very important!!!!
        main_tab_layout = [
                          [sg.Text('Exuberant Monitor')],
                          [sg.Output(size=(75, 10))],
                          [sg.Frame('Basic Live Plotting', basics_layout, font = 'Any 12', title_color = 'blue')],
                          [sg.Frame('Advanced Live Plotting', advanced_layout, font = 'Any 12', title_color = 'blue')]
                        ]

        graphing_layout = [
                          [sg.Text("Anything you would use to graph will display here!")],
                          [sg.Graph((200,200), (0,0),(200,200),background_color="black", key='-GRAPH-', enable_events=True, right_click_menu=graph_right_click_menu_def)],
                          [sg.T('Click anywhere on graph to draw a circle')]
                        ]

        theme_layout = [[sg.Text("See how elements look under different themes by choosing a different theme here!")],
                        [sg.Listbox(values = sg.theme_list(), 
                        size =(20, 12), 
                        key ='-THEME LISTBOX-',
                        enable_events = True)],
                        [sg.Button("Set Theme")]
                       ]
        
        layout = [[sg.MenubarCustom(menu_def, key='-MENU-', font='Courier 15', tearoff=True)],
                    [sg.Text('PhotonBloom Live Plotting GUI alpha version', size=(38, 1), justification='center', font=("Helvetica", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]
        layout +=[[sg.TabGroup([[sg.Tab('Important Bits', main_tab_layout),
                                sg.Tab('Graphing', graphing_layout),
                                sg.Tab('Theming', theme_layout)]], key='-TAB GROUP-', expand_x=True, expand_y=True),
                ]]
        layout[-1].append(sg.Sizegrip())
        self.window = sg.Window('QMLab PhotonBloom 2023', layout, right_click_menu=right_click_menu_def, right_click_menu_tearoff=True, grab_anywhere=True, resizable=True, margins=(0,0), use_custom_titlebar=True, finalize=True, keep_on_top=True)
        self.window.set_min_size(self.window.size)
        return self.window



        
#################################################################################3
    def main(self):
        self.window = self.make_window(sg.theme())
        # This is an Event Loop 

        try:
            while True:
                event, values = self.window.read(timeout=10)

                # if event not in (sg.TIMEOUT_EVENT, sg.WIN_CLOSED):
                #     print('============ Event = ', event, ' ==============')
                #     print('-------- Values Dictionary (key=value) --------')
                #     for key in values:
                #         print(key, ' = ', values[key])


                if event in (None, 'Exit'):
                    print("[LOG] Clicked Exit!")
                    break

                elif event == 'Start Tag Counter':
                    print("[LOG] Clicked Start Tag Counter!")

                    channel_key_list = [ch for ch in values if values[ch] == True
                                        and 'CB' in ch]
    
                    channel_list = [int(cbtext[2:]) for cbtext in channel_key_list]
                    print(channel_list, 'channels selected!')
                    binwidth_ns = int(float(values['-TAGCOUNTER_BINW-']))
                    binno = int(values['-TAGCOUNTER_BINNO-'])
                    self.datalogger.tag_counter(startfor = -1, channels = channel_list, binwidth_ns = binwidth_ns, n = binno)
                    

                elif event == 'Start Tag XCorr':
                    print("[LOG] Clicked Start Tag XCorr!")

                    lead_ch = int(values['-TAGXCORR_CH1-'])
                    follow_ch = int(values['-TAGXCORR_CH2-'])
                    binwidth_ns = int(float(values['-TAGXCORR_BINW-']))
                    binno = int(values['-TAGXCORR_BINNO-'])
                    self.datalogger.tag_correlation(startfor = -1, channels = [lead_ch, follow_ch], binwidth_ns = binwidth_ns, n = binno)
                    

                elif event == 'Start Tag TrigCorr':
                    print("[LOG] Clicked Start Tag TrigCorr!")

                    trig_ch = int(values['-TAGTRIGCORR_TRIGCH-'])
                    lead_ch = int(values['-TAGTRIGCORR_XCH1-'])
                    follow_ch = int(values['-TAGTRIGCORR_XCH2-'])
                    binwidth_ns = int(float(values['-TAGTRIGCORR_BINW-']))
                    binno = int(values['-TAGTRIGCORR_BINNO-'])
                    stacks = int(values['-TAGTRIGCORR_STACKS-'])
                    self.datalogger.tag_triggered_correlation(startfor = -1, channels = [trig_ch, lead_ch, follow_ch], binwidth_ns = binwidth_ns, n = binno, stacks = stacks)

                elif event == 'Start Tag SweepCorr':
                    print("[LOG] Clicked Start Tag SweepCorr!")

                    trig_ch = int(values['-TAGSWEEPCORR_TRIGCH-'])
                    lead_ch = int(values['-TAGSWEEPCORR_XCH1-'])
                    follow_ch = int(values['-TAGSWEEPCORR_XCH2-'])
                    binwidth_ns = int(float(values['-TAGSWEEPCORR_BINW-']))
                    binno = int(values['-TAGSWEEPCORR_BINNO-'])
                    steps = int(values['-TAGSWEEPCORR_STEPS-'])
                    self.datalogger.tag_sweep_correlation(startfor = -1, channels = [trig_ch, lead_ch, follow_ch], binwidth_ns = binwidth_ns, n = binno, step_no = steps)



                ################### Auxiliary ######################
                elif event == 'About':
                    print("[LOG] Clicked About!")
                    sg.popup('PhotonBloom Live Plotting GUI alpha version',
                            '“And all I loved, I loved alone.”',
                            keep_on_top=True)
    
                elif event == "-GRAPH-":
                    graph = self.window['-GRAPH-']       # type: sg.Graph
                    graph.draw_circle(values['-GRAPH-'], fill_color='yellow', radius=20)
                    print("[LOG] Circle drawn at: " + str(values['-GRAPH-']))
    
                elif event == "Set Theme":
                    print("[LOG] Clicked Set Theme!")
                    theme_chosen = values['-THEME LISTBOX-'][0]
                    print("[LOG] User Chose Theme: " + str(theme_chosen))
                    self.window.close()
                    self.window = self.make_window(theme_chosen)

                elif event == 'Versions':
                    sg.popup_scrolled(__file__, sg.get_versions(), keep_on_top=True, non_blocking=True)
                ################### Auxiliary ######################

        except Exception as e:
            sg.popup_error_with_traceback(f'An error happened.  Here is the info:', e)



        self.window.close()
        exit(0)


#### actually running of function
if __name__ == '__main__':

    with Hydrangea() as hy:
        hy.main()
        print('Exiting Program')



