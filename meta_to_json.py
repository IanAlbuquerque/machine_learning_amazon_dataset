""" File Description """

# ============================================================================

# Imports the print function from Python v3
# Not necessary, but facilitates code to be portable to v3
from __future__ import print_function

# ============================================================================

import math
import json
import re
import sys

# ============================================================================

DATASET_FILE_PATH = None # "./dataset/amazon-meta.txt"
JSON_OUTPUT_FILE_PATH = None # "./dataset/amazon-meta.json"

# ============================================================================

def find_group(group_array, group_name):
    """ TODO
    """
    for group in group_array:
        if group_name == group['name']:
            return group['id']
    new_group = {}
    new_group['name'] = group_name
    if not group_array:
        new_group['id'] = 0
    else:
        new_group['id'] = group_array[-1]['id'] + 1
    group_array.append(new_group)
    return new_group['id']

def add_category(categories, category_path_parsed):
    """ Add the category path to the categories array

    Keyword arguments:
    categories -- an array where the categories will be added
    category_path_parsed -- an array of dictionaries with "id" and "name"
    """

    if not category_path_parsed:
        return

    for category in categories:
        if category['id'] == category_path_parsed[0]['id']:
            if category['name'] != category_path_parsed[0]['name']:
                raise Exception('Same category has different names!')
            add_category(category['subcategories'], category_path_parsed[1:])
            return

    new_category = {}
    new_category['id'] = category_path_parsed[0]['id']
    new_category['name'] = category_path_parsed[0]['name']
    new_category['subcategories'] = []

    categories.append(new_category)

def process_category_line(products, categories, product_index_being_processed, line):
    """ Processes a new review line for a given product

    Keyword arguments:
    products -- the product array being build
    categories -- the categories array being build
    product_index_being_processed -- the index of the product of the review
    line -- the new line string
    """

    category_path_parsed = []
    category_path = re.split(r'[\s]*[\t\n\|]+[\s]*', line)[1:-1]
    for category_with_id in category_path:
        category = {}
        category_with_id_splitted = re.split(r'[\]\[]+', category_with_id)
        category['name'] = category_with_id_splitted[0]
        category['id'] = category_with_id_splitted[1]
        category_path_parsed.append(category)

    add_category(categories, category_path_parsed)

    if not 'categories' in products[product_index_being_processed]:
        products[product_index_being_processed]['categories'] = []

    products[product_index_being_processed]['categories'].append(category_path_parsed[-1]['id'])


def process_review_line(products, product_index_being_processed, line_as_list):
    """ Processes a new review line for a given product

    Keyword arguments:
    products -- the product array being build
    product_index_being_processed -- the index of the product of the review
    line_as_list -- the new line to be processes as a list
    """

    if not line_as_list:
        return

    new_review = {}
    new_review['date'] = line_as_list[0]
    new_review['customer_id'] = line_as_list[2]
    new_review['rating'] = float(line_as_list[4])
    new_review['votes'] = int(line_as_list[6])
    new_review['helpful'] = int(line_as_list[8])

    if not 'reviews' in products[product_index_being_processed]:
        products[product_index_being_processed]['reviews'] = []

    products[product_index_being_processed]['reviews'].append(new_review)

def process_new_product_line(product_array, group_array, product_index, line_as_list):
    """ Processes a new line for a given product

    Keyword arguments:
    product_array -- the product array being build
    group_array -- the group array being build
    product_index -- the index of that product in the product array
    line_as_list -- the new line to be processes as a list

    Returns number_of_categories_to_parse, number_of_reviews_to_parse,
    the number of categories and reviews to parse next
    """

    if not line_as_list:
        return

    first_word = line_as_list[0]

    if first_word == 'ASIN:':
        product_array[product_index]['asin'] = line_as_list[1]

    elif first_word == 'title:':
        product_array[product_index]['title'] = ' '.join(line_as_list[1:])

    elif first_word == 'group:':
        product_array[product_index]['group'] = find_group(group_array, ' '.join(line_as_list[1:]))

    elif first_word == 'salesrank:':
        product_array[product_index]['salesrank'] = int(line_as_list[1])

    elif first_word == 'similar:':
        product_array[product_index]['similar'] = line_as_list[2:]

    elif first_word == 'categories:':
        number_of_categories_to_parse = int(line_as_list[1])
        return number_of_categories_to_parse, 0

    elif first_word == 'reviews:':
        product_array[product_index]['reviews_total'] = int(line_as_list[2])
        product_array[product_index]['reviews_downloaded'] = int(line_as_list[4])
        product_array[product_index]['reviews_avg_rating'] = float(line_as_list[7])
        return 0, product_array[product_index]['reviews_downloaded']

    elif first_word == 'discontinued':
        product_array[product_index]['discontinued'] = True

    return 0, 0

def process_dataset_file(path):
    """ Processes the dataset file

    Keyword arguments:
    path -- the path to the dataset file

    Returns products, groups, categories,
    the dictionaries of products, groups and categories
    """

    products = []
    groups = []
    categories = []

    is_processing_new_product = False
    product_index_being_processed = 0

    num_categories_to_process = 0
    num_reviews_to_process = 0

    num_products_processed = 0
    num_products_expected = 548552
    percentage_done = 0
    last_printed_percentage = 0

    print('Processing dataset file ' + path + '...')
    with open(path) as infile:
        for line in infile:
            line_words = line.split()
            if not line_words:
                continue
            if num_categories_to_process > 0:
                num_categories_to_process -= 1
                process_category_line(products,                           \
                                      categories,                         \
                                      product_index_being_processed,      \
                                      line)
                continue
            if num_reviews_to_process > 0:
                num_reviews_to_process -= 1
                process_review_line(products,                           \
                                    product_index_being_processed,      \
                                    line_words)
                continue
            if line_words[0] == 'Id:':
                # print percentage
                num_products_processed += 1
                percentage_done = 100.0 * num_products_processed / num_products_expected
                if percentage_done - last_printed_percentage > 1.0:
                    last_printed_percentage = percentage_done
                    print(str(math.floor(percentage_done)) + '%')
                is_processing_new_product = True
                products.append({})
                product_index_being_processed = len(products) - 1
                continue
            if is_processing_new_product:
                num_categories_to_process, num_reviews_to_process = \
                                process_new_product_line(products,  \
                                groups,                             \
                                product_index_being_processed,      \
                                line_words)
                continue

    print('100% complete')
    print(str(len(products)) + ' products processed')
    print(str(len(groups)) + ' groups found')
    return products, groups, categories

def main():
    """ The main function of the program """
    products, groups, categories = process_dataset_file(DATASET_FILE_PATH)

    json_out_dict = {}
    json_out_dict['products'] = products
    json_out_dict['groups'] = groups
    json_out_dict['categories'] = categories

    print('Saving JSON to file... ' + JSON_OUTPUT_FILE_PATH)
    with open(JSON_OUTPUT_FILE_PATH, 'w') as outfile:
        outfile.write(json.dumps(json_out_dict))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('NOT ENOUGH ARGUMENTS!\n' + \
              'Usage:\n' + \
              '\tpython ./meta_to_json.py <PATH TO INPUT DATASET FILE> ' + \
              '<PATH TO OUTPUT DATASET FILE>\n' + \
              'Example:\n' + \
              '\tpython ./meta_to_json.py ./dataset/amazon-meta.txt ' + \
              './dataset/amazon-meta.json')
    else:
        DATASET_FILE_PATH = sys.argv[1]
        JSON_OUTPUT_FILE_PATH = sys.argv[2]
        main()
