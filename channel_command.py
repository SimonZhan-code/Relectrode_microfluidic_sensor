import serial
from binascii import unhexlify
import time
from MUX import *

# Command dictionary for current 4 channels
first = {"fast_f": "fe0801115b016eef", "fast_b": "fe0801115b026fef",
         "start_stop": "fe0801115b0370ef", "pause_continue": "fe0801115b0471ef"}
second = {"fast_f": "fe0801125b016fef", "fast_b": "fe0801125b0270ef",
          "start_stop": "fe0801125b0371ef", "pause_continue": "fe0801125b0472ef"}
third = {"fast_f": "fe0801135b0170ef", "fast_b": "fe0801135b0271ef",
         "start_stop": "fe0801135b0372ef", "pause_continue": "fe0801135b0473ef"}
fourth = {"fast_f": "fe0801145b0171ef", "fast_b": "fe0801145b0272ef",
          "start_stop": "fe0801145b0373ef", "pause_continue": "fe0801145b0474ef"}

# Initialize a valve object

mux = MUX()


# Moving module function
# which requires channel input and corresponding command input

def make_move(channel, command, my_serial):
    curr = unhexlify(channel[command])
    my_serial.write(curr)


# function to control channel to fill in spiral
# time used: 10.7s
def spiral_filling(my_serial):
    mux.set_valves(2, 3)
    time.sleep(0.05)
    make_move(third, "start_stop", my_serial)
    time.sleep(2)
    make_move(second, "start_stop", my_serial)
    time.sleep(4)
    make_move(second, "start_stop", my_serial)
    time.sleep(1.5)
    make_move(third, "start_stop", my_serial)
    time.sleep(0.05)
    mux.set_valves(3)
    time.sleep(3)
    mux.set_all_valves_off()
    time.sleep(0.05)
    empty(my_serial, third)
    time.sleep(0.05)
    refilling(my_serial, second)



# function to fill pixels based on spiral shape
# time used: 4.95s
def pixel_filling_spiral(my_serial):
    mux.set_valves(2)
    make_move(second, "start_stop", my_serial)
    time.sleep(1)
    make_move(second, "start_stop", my_serial)
    time.sleep(0.1)
    mux.set_valves(2, 3)
    time.sleep(0.1)
    make_move(third, "start_stop", my_serial)
    time.sleep(0.2)
    make_move(third, "start_stop", my_serial)
    time.sleep(0.1)
    mux.set_valves(2)
    time.sleep(0.1)
    make_move(second, "start_stop", my_serial)
    time.sleep(1)
    mux.set_valves(2, 3)
    time.sleep(0.05)
    make_move(third, "start_stop", my_serial)
    time.sleep(2)
    make_move(third, "start_stop", my_serial)
    time.sleep(0.1)
    make_move(second, "start_stop", my_serial)
    time.sleep(0.1)
    mux.set_all_valves_off()


# function to leave the pixel alone
# time used: 2.4s
def pixel_alone(my_serial):
    mux.set_valves(1, 3)
    time.sleep(0.1)
    make_move(third, "start_stop", my_serial)
    time.sleep(0.1)
    make_move(first, "start_stop", my_serial)
    time.sleep(2)
    make_move(first, "start_stop", my_serial)
    time.sleep(0.1)
    make_move(third, "start_stop", my_serial)
    time.sleep(0.1)
    mux.set_all_valves_off()


# function to clean everything from the channel
# time 30s
def cleaning(my_serial):
    mux.set_valves(3)
    time.sleep(0.05)
    make_move(third, "start_stop", my_serial)
    time.sleep(2)
    mux.set_valves(1, 3)
    time.sleep(0.05)
    # make_move(third, "start_stop", my_serial)
    # time.sleep(0.5)
    make_move(first, "start_stop", my_serial)
    time.sleep(5)
    make_move(first, "start_stop", my_serial)
    time.sleep(0.05)
    mux.set_valves(3)
    time.sleep(2)
    make_move(third, "start_stop", my_serial)
    time.sleep(0.05)
    mux.set_valves(1)
    time.sleep(0.05)
    make_move(first, "start_stop", my_serial)
    time.sleep(1.1)
    make_move(first, "start_stop", my_serial)
    mux.set_valves(3)
    time.sleep(0.05)
    make_move(fourth, "start_stop", my_serial)
    time.sleep(5)
    mux.set_valves(1, 3)
    time.sleep(0.05)
    make_move(first, "start_stop", my_serial)
    time.sleep(5)
    make_move(first, "start_stop", my_serial)
    time.sleep(0.05)
    make_move(fourth, "start_stop", my_serial)
    time.sleep(7)
    mux.set_valves(3)
    time.sleep(2)
    mux.set_all_valves_off()
    time.sleep(0.05)
    empty(my_serial, fourth)
    time.sleep(0.05)
    empty(my_serial, third)
    time.sleep(0.05)
    refilling(my_serial, first)




    # mux.set_valves(3)
    # suction_cleaning(my_serial)
    # time.sleep(0.1)
    # mux.set_valves(1)
    # time.sleep(0.1)
    # make_move(first, "start_stop", my_serial)
    # time.sleep(1)
    # make_move(first, "start_stop", my_serial)
    # time.sleep(0.1)
    # mux.set_valves(1, 3)
    # time.sleep(0.1)
    # make_move(fourth, "start_stop", my_serial)
    # time.sleep(2)
    # make_move(fourth, "start_stop", my_serial)
    # time.sleep(0.1)
    # mux.set_valves(1)
    # time.sleep(0.1)
    # make_move(first, "start_stop", my_serial)
    # time.sleep(1)
    # make_move(first, "start_stop", my_serial)
    # time.sleep(0.1)
    # mux.set_valves(1, 3)
    # time.sleep(0.1)
    # make_move(fourth, "start_stop", my_serial)
    # time.sleep(2)
    # make_move(first, "start_stop", my_serial)
    # time.sleep(1.5)

    # make_move(fourth, "start_stop", my_serial)
    # suction_cleaning(my_serial)
    # time.sleep(2)
    # make_move(first, "start_stop", my_serial)
    # time.sleep(2.2)
    # mux.set_valves(3)
    # make_move(third, "start_stop", my_serial)
    # time.sleep(3)
    # make_move(third, "start_stop", my_serial)
    # time.sleep(2)
    # mux.set_all_valves_off()



# Commands to ultimately clean everything from the chip

def suction_cleaning(my_serial):
    mux.set_valves(3)
    time.sleep(0.1)
    make_move(fourth, "start_stop", my_serial)
    time.sleep(0.1)
    make_move(third, "start_stop", my_serial)
    time.sleep(4.5)
    make_move(fourth, "start_stop", my_serial)
    time.sleep(0.1)
    make_move(third, "start_stop", my_serial)


# Refill liquid according to channel

def refilling(my_serial, channel):
    make_move(channel, "fast_b", my_serial)


# Empty everything in corresponding channel

def empty(my_serial, channel):
    make_move(channel, "fast_f", my_serial)
