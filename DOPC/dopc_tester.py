#   Testing tool for the "Delivery Order Price Calculator"-server
#   Code by Adrian Grun (egrun@gmx.de)  January 2025
from requests import get
from random import randint
from random import uniform as randfloat

#   Reference data for generating test-input
_venues = ["Berlin", "Helsinki", "Stockholm", "Tokyo"]
_venue_geodata = [[52.5003197, 13.4536149], [60.17012143, 24.92813512], [59.3466978, 18.0314984], [35.6459122, 139.7115264]]
_RED = "\033[1;31m"
_GREEN = "\033[0;32m"
_YELLOW = "\033[1;33m"
_GREY = "\033[0;37m"

def generate_test(illegal):
    if illegal:
        venue = "Moon"
        cart_value = randint(-10, 10)
        x = randfloat(-120.0, 120.0)
        y = randfloat(-210.0, 210.0)
    else:
        v = randint(0, len(_venues)-1)
        venue = _venues[v]
        lat, lon = _venue_geodata[v][0], _venue_geodata[v][1]
        cart_value = randint(1, 2000)
        x = randfloat(lat-0.02, lat+0.02)
        y = randfloat(lon-0.02, lon+0.02)
    print(f"\n\t{_YELLOW}Generated test-input:{_GREY}\n", end="")
    print(f"\tVenue:\t\t\t{venue}\n\tCart Value:\t\t{cart_value}\n\tLatitude:\t\t{x}\n\tLongitude:\t\t{y}")

    request = "http://localhost:8000/api/v1/delivery-order-price?venue_slug=home-assignment-venue-" + venue
    request += "&cart_value=" + str(cart_value)
    request += "&user_lat=" + str(x)
    request += "&user_lon=" + str(y)
    try:
        get(request)
        print(f"{_GREEN}\tRequest successfully sent!{_GREY}")
    except:
        print(f"{_RED}\tCould not connect to localhost. Server down?{_GREY}")

if __name__ == "__main__":
    print(f"{_GREY}\tThis program can send randomly generated delivery requests to the local server on port 8000.")
    print(f"\tPlease make sure the server ({_YELLOW}dopc_server.py{_GREY}) is running before attempting to use this tool.")
    while 1:
        action = input(f"\n\t{_YELLOW}Please choose an action:\n\t1) {_GREY}Run a random test."
                       f"{_YELLOW} 2) {_GREY}Run an illegal test. "
                       f"{_YELLOW} 3) {_GREY}Exit the program. ")
        action = action.strip()
        if len(action) != 1 or (action != "1" and action != "2" and action != "3"):
            print(f"{_RED}\tInvalid input. Please repeat.{_GREY}")
        else:
            if action == "1":
                generate_test(False)
            elif action == "2":
                generate_test(True)
            else:
                break