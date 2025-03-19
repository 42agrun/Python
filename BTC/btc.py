from collections import deque

class Datatable:
    # This function is used for debugging by printing out the table's content
    def print_table(self):
        x = 0
        print("\033[1;33mID       Date          BTC)")
        print("\033[1;33m==========================================\033[1;37m")
        while x < self.lines:
            print(f"{self.table[x].a}\t\t{self.table[x].b}\t\t{self.table[x].c}")
            x += 1

    # This will make sure no illegal BTC values are processed
    # Only positive numbers are accepted, only one dot is allowed
    def check_value(self, value):
        n = 0
        dot = 0
        if len(value) < 1 or value.find("-") > -1:
            return 1
        for x in value:
            if not x.isnumeric() and x != '.':
                return 1
            elif ((n == 0 and x == '.') or
                    (dot == 1 and x == '.')):
                return 1
            elif dot == 0 and x == '.':
                dot += 1
            n += 1
        return 0

    # This will check for faulty formats & illegal dates
    # Dates must be formatted as YYYY-MM-DD
    # Dates cannot have more than 12 months and 31 days
    def check_date(self, date):
        faulty = "\033[0;31m" + date + ": illegal date!"
        if len(date) != 10:
            return faulty
        n = 0
        for i in date:
            if n != 4 and n != 7 and not i.isnumeric():
                return faulty
            elif (n == 4 or n == 7) and i != "-":
                return faulty
            n += 1
        month = date[5]
        month += date[6]
        day = date[8]
        day += date[9]
        if (int(month) > 12 or int(month) < 1
                or int(day) > 31 or int(day) < 1):
            return faulty
        return date

    # Parsing lines and keeping track of their total number
    def scan_line(self, line):
        switch = 0
        date = ""
        value = ""
        for c in line:
            if c == ',' or c == '|':
                switch = 1
            else:
                if switch == 0:
                    date += c
                else:
                    value += c
        date = date.strip()
        value = value.strip()
        # Skipping the header of the parsed list
        if date == "date" and self.lines == 0:
            return 0
        self.lines += 1
        # Checking date and value for issues
        date = self.check_date(date)
        if self.check_value(value) == 1:
            value = "\033[0;31m" + value + ": faulty format!"
        self.table.append(TripleArray(self.lines, date, value))

    # Function for parsing the entire file
    def parse_table(self, path):
        file = open(path, "r")
        content = file.read()
        line = ""
        for c in content:
            if c != "\n":
                line += c
            else:
                self.scan_line(line)
                line = ""
        file.close()

    # Checking if the given file is viable/available
    def check_file(self, path):
        try:
            with open(path, "r") as file:
                file.close()
                return 0
        except FileNotFoundError:
            if path == "data.csv":
                print("Critical error within default datatable.\n")
            print("Error reading file \"" + path + "\". Quitting now.")
            exit(1)

    # Using a deque to store the TripleArray objects
    def __init__(self, path):
        self.table = deque()
        self.lines = 0
        self.check_file(path)
        self.parse_table(path)

########################################################################
# a = ID used for sorting input
# b = data linked to a certain date
# c = the BTC value at that point of time

class TripleArray:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c
########################################################################

# Making sure to skip the "-" signs within the date-string
def int_conv(string):
    number = ""
    for i in string:
        if i.isnumeric():
            number += i
    return int(number)

# Find the closest date from the default table for calculating
# Turn the dates into integer values for easier comparison
# Start checking with the newest dates, then go back in time
def calculate_btc(stocks, date, value):
    new_date = int_conv(date)
    diff = int_conv(stocks.table[0].b) - new_date
    if diff < 0:
        diff *= -1
    found = 0
    n = stocks.lines - 1
    while n > -1:
        if new_date == int_conv(stocks.table[n].b):
            found = n
            break
        if new_date < int_conv(stocks.table[n].b):
            filter = int_conv(stocks.table[n].b) - new_date
            if filter < 0:
                filter *= -1
            if filter < diff:
                diff = filter
                found = n
        n -= 1
    btc = float(stocks.table[found].c) * float(value)
    value = round(float(value), 2)
    btc = round(btc, 2)
    # One additional tabulator space to adjust the formatting
    if len(str(int(value))) == 1:
        return f"{date}\t{value}\t\t\t\033[0;32m{btc}\033[1;37m"
    return f"{date}\t{value}\t\t\033[0;32m{btc}\033[1;37m"

# This function optimizes the formatting for the printout
# How many IDs are there in total? How many zeroes are needed?
def generate_id(table, number):
    x = 1
    max_id = table.lines
    if max_id > 999:
        x = 4
    elif max_id > 99:
        x = 3
    elif max_id > 9:
        x = 2
    diff = x - len(number)
    id = "\033[0;34m("
    while diff > 0:
        id += '0'
        diff -= 1
    id += number + ")\033[1;37m"
    return id

def print_btc(d1, d2):
    n = 1
    print("\033[1;33m ID   Date          BTC         €")
    print("\033[1;33m==========================================\033[1;37m")
    for i in d2.table:
        if i.b.find("\033") > -1 or i.c.find("\033") > -1:
            print(f"{generate_id(d2, str(n))}  {i.b} \033[1;37m[{i.c}\033[1;37m]")
        else:
            print(f"{generate_id(d2, str(n))}  {calculate_btc(d1, i.b, i.c)}")
        n += 1

if __name__ == "__main__":
    stocks = Datatable("data.csv")
    path = input("Please enter a valid filepath for BTC calculation: ")
    userdata = Datatable(path)
    print_btc(stocks, userdata)