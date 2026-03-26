
"""

Simple safety wrapper for Tello movements.


"""

import time


# SETTINGS

MIN_BATTERY = 20     # % battery required
COOLDOWN = 2         # seconds between movements

last_move_time = 0


# BASIC SAFETY CHECKS
def is_safe(tello):
    """
    Check battery before moving
    """
    battery = tello.get_battery()
    print(f"Battery: {battery}%")

    if battery < MIN_BATTERY:
        print("⚠️ Battery too low → landing")
        tello.land()
        return False

    return True


def allow_move():
    """
    Prevents too many commands too fast
    """
    global last_move_time

    current_time = time.time()

    if current_time - last_move_time > COOLDOWN:
        last_move_time = current_time
        return True

    return False



# For SAFE MOVEMENT FUNCTIONS

def safe_move_up(tello):
    if is_safe(tello) and allow_move():
        print("Moving UP")
        tello.move_up(20)


def safe_move_down(tello):
    if is_safe(tello) and allow_move():
        print("Moving DOWN")
        tello.move_down(20)


def safe_move_forward(tello):
    if is_safe(tello) and allow_move():
        print("Moving FORWARD")
        tello.move_forward(30)


def safe_hover():
    print("Hovering (no movement)")



# EMERGENCY STOP
def emergency_land(tello):
    print(" EMERGENCY LAND")
    tello.land()