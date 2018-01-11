#=====================================================================================================
# This script is a OS process wrapper script that will execute the script "InvokeMultipleProcess.py"
# for n number of processes and with a grocery store range to cycle through
#
# Input Args Usage/Example:
# "C:\ProgramData\Anaconda3\python.exe D:\project\python\pycharm\multiValidation\InvokeMultipleProcesses.py 5 15-20"
# "5"=number of processes to run at a time.  "15-20"=the store range to process sequentially
#=====================================================================================================

from multiprocessing import Process
import numpy as np

import sys, os, time
from subprocess import Popen, list2cmdline
# works across all three OS
from sys import platform
import multiprocessing  # the module we will be using for multiprocessing


# function to execute commands.  Borrowed this from online.  Ref/Credit:
# http://code.activestate.com/recipes/577376-simple-way-to-execute-multiple-process-in-parallel/
def exec_commands(cmds):
    ''' Exec commands in parallel in multiple process
    (as much as we have CPU)
    '''
    if not cmds: return # empty list

    def done(p):
        return p.poll() is not None
    def success(p):
        return p.returncode == 0
    def fail():
        sys.exit(1)

    max_task = cpu_count()
    processes = []
    while True:
        while cmds and len(processes) < max_task:
            task = cmds.pop()
            print (list2cmdline(task))
            processes.append(Popen(task))

        for p in processes:
            if done(p):
                if success(p):
                    processes.remove(p)
                else:
                    fail()

        if not processes and not cmds:
            break
        else:
            time.sleep(0.05)

# if we want to specify a number to test with.  0 is all
def runSubProcess(rangeNumber):

    # Print which range of restaurants we are processing
    print('Range Number: ' + str(rangeNumber))

    # If we are Testing.  0 is no testing.
    testNum = 0

    # Win 10
    if platform == 'win32':
        execString = 'C:\ProgramData\Anaconda3\python.exe  '
        scriptString = 'D:\project\python\pycharm\multiValidation\GenTimeSeriesOptionsAndResults.py '

    # Mac
    elif platform == 'darwin':
        execString = 'python  '
        scriptString = '//Project/data/kg_corpgroc/GenTimeSeriesOptionsAndResults.py '

    # Amazon AMI - really Ubuntu
    elif platform == 'linux':
        execString = 'python3  '
        scriptString = '//data/GenTimeSeriesOptionsAndResults.py '

    # Set the range number of the restaraunts we are processing
    paramString = str(rangeNumber) + ' False ' + str(testNum)

    print('Full script: ' + execString + scriptString + paramString)

    os.system(execString + scriptString + paramString)


# This was a default of how I wanted the stores being split up
#  depending no how it turned out after running, the over-ride parameters controlled the running later on
def fn_determine_store_range_per_system():

    # Win 10 6x12 @ 4.0 ghz
    if platform == 'win32':
        ret_storeStartNum = 1  # Blocks - AWS starting at 1
        ret_storeRange = np.arange(1, 20, step=1)

    # Mac - 4x8 @ 2.0 ghz
    elif platform == 'darwin':
        ret_storeStartNum = 20  # Blocks - AWS starting at 20
        ret_storeRange = np.arange(20, 35, step=1)

    # Amazon AMI 8 or 16 @ ?.?ghz
    elif platform == 'linux':
        ret_storeStartNum = 35  # Blocks - AWS starting at 35
        ret_storeRange = np.arange(35, 54, step=1)

    return ret_storeStartNum, ret_storeRange

# Main
if __name__ == "__main__":  # Allows for the safe importing of the main module
    print("Platform: " + platform)
    print("There are %d CPUs on this machine: " % multiprocessing.cpu_count())

    # Determine # of processes
    number_processes = multiprocessing.cpu_count()
    # Override
    if len(sys.argv) >= 2:
        number_processes = int(sys.argv[1])
        print('Override - Number of CPU:  ' + str(number_processes))

    # determine range - as an initial default - moved to an over-ride below later based on
    #  how fast each computer could process the stores
    storeStartNum, storeRange = fn_determine_store_range_per_system()

    # If passed in - override
    if len(sys.argv) >= 3:
        ''' input of n-n or 10-20 '''
        arg_range = sys.argv[2]
        #arg_InputRange = '10-20'
        argRangeResults = list(map(int, arg_range.split('-')))

        storeStartNum = argRangeResults[0]
        storeRange = np.arange(argRangeResults[0], argRangeResults[1], step=1)

        print('Override - Store Start: ' + str(storeStartNum))
        print('Override - Range: ' + str(storeRange))

    # (Future Enhancement) Add an over-ride parameter for aws AMI (not fully implemented)
    # will have to re-work this item for the multi-processing into array
    arg_directoryOveride = ''
    if len(sys.argv) >= 4:
        ''' Input over-ride for directory '''
        arg_directoryOveride = sys.argv[3]

    # Actual implementataion of  pooling the individual processes
    pool = multiprocessing.Pool(number_processes)
    results = pool.map_async(runSubProcess, storeRange)
    pool.close()
    pool.join()