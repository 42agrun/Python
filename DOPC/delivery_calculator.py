#   Delivery Order Price Calculator
#   Code by Adrian Grun (egrun@gmx.de)  January 2025
from requests import get
import math

#   There are two crucial global variables in this project:
#   '_errors' will save any encountered issues for documentation,
#   meanwhile '_locations' will contain all supported delivery venues.
_errors = []
_locations = ["berlin", "helsinki", "stockholm", "tokyo"]

def get_integer(num, ven):
    if num == -1:
        return
    while num < len(ven) and ven[num].isdigit() is False:
        num += 1
    value = ""
    while num < len(ven) and ven[num].isdigit() is True:
        value += ven[num]
        num += 1
    return int(value)

def get_range_data(data):
    #   How many data-sets are there?
    packets = data.split("{")
    #   First stored set within array is empty/irrelevant
    packets.pop(0)
    range_data = [[]]
    for x in packets:
        i = x.find("\"min\"")
        j = x.find("\"max\"")
        k = x.find("\"a\"")
        l = x.find("\"b\"")
        if i == -1 or j == -1 or k == -1 or l == -1:
            _errors.append("Error! Incomplete distance range-data! (min/max/a/b)<br />")
            return
        i, j, k, l = (get_integer(i, x), get_integer(j, x),
                      get_integer(k, x), get_integer(l, x))
        range_data.append((i, j, k, l))
    range_data.pop(0)
    return range_data

def check_for_ranges(num, venue):
    if num == -1:
        return
    while num < len(venue) and venue[num] != "[":
        num += 1
    specs = ""
    while num < len(venue) and venue[num] != "]":
        specs += venue[num]
        num += 1
    return get_range_data(specs)

def get_delivery_specs(dynamic_venue):
    i = dynamic_venue.find("order_minimum_no_surcharge")
    surcharge = get_integer(i, dynamic_venue)
    i = dynamic_venue.find("base_price")
    baseprice = get_integer(i, dynamic_venue)
    i = dynamic_venue.find("distance_ranges")
    ranges = check_for_ranges(i, dynamic_venue)
    if surcharge is None or baseprice is None or ranges is None:
        _errors.append("Error! Cannot find required venue delivery specs!<br />")
        return
    return surcharge, baseprice, ranges

def parse_coordinates(static_venue):
    i = len(static_venue)-1
    findme, saveme = "coordinates", ""
    #   Searching the keyword ("coordinates") at the end of the page
    while static_venue[i] and i > -1:
        if static_venue[i].isspace() or not static_venue[i].isalpha():
            if findme[::-1] == saveme:
                break
            saveme = ""
        else:
            saveme += static_venue[i]
        i -= 1
    #   In case the page isn't properly formatted
    if i == -1:
        _errors.append("Error! Cannot find any static venue-coordinates!<br />")
        return
    #   Skipping forward to the coordinates themselves
    while i < len(static_venue) and static_venue[i] != "[":
        i += 1
    i += 1
    coordinates = ""
    #   Saving the coordinates
    while i < len(static_venue) and static_venue[i] != "]":
        coordinates += static_venue[i]
        i += 1
    return coordinates

#   Requesting static and dynamic venue data for further processing
def get_venue_data(venue):
    url = "https://consumer-api.development.dev.woltapi.com/home-assignment-api/v1/venues/home-assignment-venue-"
    url += venue.lower()
    static = url + "/static"
    dynamic = url + "/dynamic"
    try:
        s_response = get(f"{static}")
        d_response = get(f"{dynamic}")
    except:
        _errors.append("Unable to connect to the internet!<br />")
        return None, None
    if str(s_response).find("200") == -1 or str(d_response).find("200") == -1:
        _errors.append("Unable to acquire venue-data! API possibly down?<br />")
        return None, None
    coordinates = parse_coordinates(s_response.text)
    if coordinates is None:
        return None, None
    coordinates = coordinates.split(',')
    delivery_specs = get_delivery_specs(d_response.text)
    return coordinates, delivery_specs

#   Checking a global list ('_locations') containing the supported venues for delivery
#   If there are already errors present this function will be omitted, otherwise redundant
def check_data(data):
    if len(_errors):
        return
    okay = False
    for x in _locations:
        if data[0].lower() == x:
            okay = True
    if not okay:
        _errors.append(f"Invalid venue entered! '{data[0]}' is not supported!<br />")
    if data[1] < 0:
        _errors.append("Only positive cart values are accepted!<br />")
    elif data[1] == 0:
        _errors.append("Cart is empty! (Value: 0)<br />")
    if (data[2] < -90 or data[2] > 90) or (data[3] < -180 or data[3] > 180):
        _errors.append("Invalid coordinates! Latitude range: -90/90 & Longitude range: -180/180<br />")

#   Formula: base_price + a + b * distance / 10
#   Checking if the delivery distance is acceptable
def calculate_fees(distance, ranges, price):
    n, i = 0, 0
    for x in ranges:
        if x[1] == 0:
            n = x[0]
            break
        # Consider allowed minimum and maximum distance
        elif distance >= x[0] and distance < x[1]:
            n = x[1]
            break
        i += 1
    if distance > n:
        _errors.append(f"Distance ({distance} meters) is too great for delivery!<br />")
        return
    fee = price + ranges[i][2] + round(ranges[i][3] * distance / 10)
    return fee

#   Formula for calculating the distance between two pairs of coordinates on Earth
#   The result will be in km, needs to be adjusted for meters via multiplication by 1000
def calculate_distance(x1, x2, y1, y2):
    distance = math.acos((math.sin(math.radians(x1)) * math.sin(math.radians(x2))) + (math.cos(math.radians(x1)) * math.cos(math.radians(x2))) * (math.cos(math.radians(y2) - math.radians(y1)))) * 6371
    return round(distance * 1000)

def create_json_file(total, surcharge, cart, fee, distance):
    content = "{\n\t\"total_price\": " + str(int(total))
    content += ",\n\t\"small_order_surcharge\": " + str(int(surcharge))
    content += ",\n\t\"cart value\": " + str(int(cart))
    content += ",\n\t\"delivery\": {\n\t\t\"fee\": " + str(int(fee))
    content += ",\n\t\t\"distance\": " + str(int(distance)) + "\n\t}\n}"
    return content

def start_dopc(argv):
    _errors.clear()
    data = []
    if len(argv) != 4:
        return "illegal"
    data.append(str(argv[0]))           # venue
    try:
        data.append(int(argv[1]))       # cart value
    except:
        _errors.append("Illegal character(s) used within cart-value!<br />")
    try:
        data.append(float(argv[2]))     # user-latitude
        data.append(float(argv[3]))     # user-longitude
    except:
        _errors.append("Illegal character(s) used within coordinates!<br />")
    check_data(data)
    if len(_errors):
        return _errors
    #   User-input from the URL-string has been accepted so far
    venue_location, venue_specs = get_venue_data(data[0])
    #   Venue LONGitude and LATitude are stored in location[0] and [1]
    #   Swapped order compared to argv/data (LATitude and LONGitude)
    if venue_location and venue_specs:
        distance = calculate_distance(data[2], float(venue_location[1]),
                                    data[3], float(venue_location[0]))
        surcharge, price, ranges = venue_specs
        surcharge -= data[1]
        # Surcharge cannot be negative so it's set to 0
        if surcharge < 0:
            surcharge = 0
        fee = calculate_fees(distance, ranges, price)
        if len(_errors) == 0:
            return create_json_file(fee + data[1] + surcharge, surcharge, data[1], fee, distance)
    return _errors

if __name__ == "__main__":
    print("\033[0;31mPlease start the program via 'dopc_server.py'!\033[1;37m")