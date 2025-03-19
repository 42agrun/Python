This document and its described project ("**Delivery Order Price Calculator**")
were written by Adrian Grun (egrun@gmx.de) during January 2025.

The purpose of this app is to **generate a JSON-file as a server-response** with coherent 
delivery information after processing the data received from the URL-string.
*   **

# SETUP

1) Make sure Python is installed on your device!

2) **Install the external library "requests".** It is mandatory
   for accessing online-data which is obligatory for this project.
   * From within your command line interface, type the following:
      >pip install requests

3) Make sure that all application-files **share the same folder** during execution!

*   **

# HOW TO USE

1) This project consists of three files:
   >dopc_server.py, dopc_tester.py, delivery_calculator.py

* It is obligatory to run **DOPC_SERVER.PY** first to set up the server. 

>**DOPC_TESTER.PY** is OPTIONAL and serves for testing via randomized input.

>**DELIVERY_CALCULATOR.PY** will be accessed by the server however it cannot be started by the user itself, it will simply quit if you attempt to do so.


   * **Enter "*python3 dopc_server.py*" to start the app/server.**
   * If you wish to perform tests, open another instance of your console. 
   * Type "*python3 dopc_tester.py*" to access the testing tool.


2) You will be able to send information to the server from within your browser. **This is the input format you'll need to follow:**
        
        http://localhost:8000/api/v1/delivery-order-price?venue_slug=home-assignment-venue-[LOCATION]&cart_value=[VALUE]&user_lat=[X]&user_lon=[Y]

   Here's an example of how a proper input might look like:

        http://localhost:8000/api/v1/delivery-order-price?venue_slug=home-assignment-venue-helsinki&cart_value=1000&user_lat=60.17094&user_lon=24.93087

   For reference, here are the **coordinates of the supported venues**:

        BERLIN: 	52.5003197  / 13.4536149
        HELSINKI:	60.17012143 / 24.92813512
        STOCKHOLM:	59.3466978  / 18.0314984
        TOKYO:		35.6459122  / 139.7115264

      These coordinates are divided into **LATITUDE** and **LONGITUDE**. To increase readability, these attributes are referred to as "X" and "Y" in this project.


3) Cart value input should consist of POSITIVE integers, the **CART MUST NOT BE EMPTY (0)**.
    There can't be more than the four mentioned attributes (*LOCATION, VALUE, X, Y*).
        
        Allowed range for X: -90/90
        Allowed range for Y: -180/180

   * Submitting illegal/non-supported venues will disqualify the other attributes.
   * Breaking any of these conditions will result in an **appropriate error-feedback**
   displayed on the browser and console-interface.


4) **You can shut down the server** by simply closing the console window. 
   >There is no current support for recognizing manual key-input to shut down the application.