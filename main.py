""" File Description """

# ============================================================================

# Imports the print function from Python v3
# Not necessary, but facilitates code to be portable to v3
from __future__ import print_function

# ============================================================================

import json

# ============================================================================

def main():
    """ The main function """

    data = None
    print('Reading JSON file... This might take a while.')
    with open('./dataset/amazon-meta.json') as infile:
        data = json.load(infile)
    print('Done!')

    print(data['groups'])

    avg_salesrank = 0.0
    total_products = len(data['products'])
    for product in data['products']:
        if not 'discontinued' in product:
            avg_salesrank += (product['salesrank'] / total_products)

    print(avg_salesrank)

if __name__ == "__main__":
    main()
