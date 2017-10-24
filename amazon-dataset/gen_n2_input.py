""" File Description """

# ============================================================================

# Imports the print function from Python v3
# Not necessary, but facilitates code to be portable to v3
from __future__ import print_function

# ============================================================================

import json
import sys
import math
import datetime

# ===========================================================================

JSON_INPUT_FILE_PATH = None # "./dataset/amazon-meta.json"
N2_OUTPUT_FILE_PATH = None # "./dataset/n2-input.json"

# ============================================================================

def gen_line_table(data, product_1, product_2):
    """ Returns the line of the table being product_1 and product_2 the two valid products """

    # unused variable
    _ = data

    salesrank_p1 = product_1['salesrank']
    salesrank_p2 = product_2['salesrank']

    return (salesrank_p1, salesrank_p2)

def compute_time_expected(time_before, percentage_done):
    """ Computes time expected to end in h, min, s """

    now_time = datetime.datetime.now()
    time_diff = now_time - time_before
    seconds_elapsed = time_diff.total_seconds()
    seconds_expected = (seconds_elapsed / percentage_done) * (100 - percentage_done)
    minutes_to_show, seconds_to_show = divmod(seconds_expected, 60)
    hours_to_show, minutes_to_show = divmod(minutes_to_show, 60)
    return (math.floor(hours_to_show), math.floor(minutes_to_show), math.floor(seconds_to_show))

def main():
    """ The main function """

    data = None
    print('Reading JSON (' + JSON_INPUT_FILE_PATH + ') file... This might take a while.')
    with open(JSON_INPUT_FILE_PATH) as infile:
        data = json.load(infile)
    print('Done!')

    valid_products = []
    for product in data['products']:
        if not 'discontinued' in product:
            valid_products.append(product)

    num_valid_products = len(valid_products)
    num_edges = num_valid_products * (num_valid_products - 1)

    print(str(num_valid_products) + ' valid products found...')
    print(str(num_edges) + ' edges expected...')

    entries = []

    num_edges_processed = 0
    percentage_done = 0
    last_printed_percentage = 0
    starting_time = datetime.datetime.now()

    for product_1 in valid_products:
        for product_2 in valid_products:
            if product_1['asin'] != product_2['asin']:

                # print percentage
                num_edges_processed += 1
                percentage_done = 100.0 * num_edges_processed / num_edges
                if percentage_done - last_printed_percentage > 0.001:
                    time_expected_to_show = compute_time_expected(starting_time, percentage_done)
                    last_printed_percentage = percentage_done
                    print(str(math.floor(percentage_done*10000)/10000) + '%. ' + \
                                    'Expected more ' + \
                                    str(time_expected_to_show[0]) + 'h ' + \
                                    str(time_expected_to_show[1]) + 'm ' + \
                                    str(time_expected_to_show[2]) + 's ')

                entries.append(gen_line_table(data, product_1, product_2))

    print(str(len(entries)) + ' entries found...')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('NOT ENOUGH ARGUMENTS!\n' + \
              'Usage:\n' + \
              '\tpython ./gen_n2_input.py <PATH TO INPUT JSON FILE> ' + \
              '<PATH TO OUTPUT JSON FILE>\n' + \
              'Example:\n' + \
              '\tpython ./gen_n2_input.py ./dataset/amazon-meta.json ' + \
              './dataset/n2_input.json')
    else:
        JSON_INPUT_FILE_PATH = sys.argv[1]
        N2_OUTPUT_FILE_PATH = sys.argv[2]
        main()
