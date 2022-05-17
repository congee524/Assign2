import re
import sys

import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import torch


def remove_special_symbols(input_str):
    punctuation = ' -\''
    html_tokens = ['<br />', '<b>', '</b>']
    res = input_str
    for token in html_tokens:
        res = res.replace(token, ' ')
    res = str().join(e for e in res if e.isalnum() or e in punctuation)
    res = re.sub(r'\s+', ' ', res)
    return res


def process_desc(desc, nbhd, book):
    # process description info
    res = ''

    res += desc

    if nbhd != '':
        nbhd = ' locate at ' + nbhd + '. '
    res += nbhd

    if 't' in book:
        book = 'able to instantly book.'
    elif 'f' in book:
        book = 'unable to instantly book.'
    res += book

    return res


def process_fac(type, accommodate, bathroom, bedrooms, amenity):
    # process facility info
    res = ''

    if type == '':
        type = 'Private room'
    if accommodate == '':
        accommodate = '1'
    res += 'The ' + type + ' is accommodated for ' + str(
        accommodate) + ' persons. '

    if bathroom != '':
        res += 'The house has ' + bathroom
        if bedrooms != '':
            res += ' and ' + str(bedrooms) + ' bedrooms'
        res += '. '
    if amenity != '':
        amenity = str().join(e for e in amenity if e.isalnum() or e in ' ,-')
        res += 'The available facilities include ' + amenity + '.'

    return res


def process_rev(rev, rating, score_A, score_B, score_C, score_D):
    if rev == '0' or rev == '' or rating == '':
        return 'Has no review.'
    format_list = [
        int(float(item)) if item != '' else 0
        for item in [rating, rev, score_A, score_B, score_C, score_D]
    ]

    res = ' The average rating of reviews is {} from {} reviewers, and the ' \
        'average ratings of reviews on A, B, C, D are {}, {}, {}, {}.'.format(
        *format_list)
    return res


def data_processing(csv_fname):
    # replace all null with ''
    data = pd.read_csv(csv_fname, delimiter=',', keep_default_na=False)
    num_data = len(data)

    # process string
    data_desc = list(
        map(process_desc, data['description'], data['neighbourhood'],
            data['instant_bookable']))
    data_desc = list(map(remove_special_symbols, data_desc))
    data_desc = tf.reshape(tf.convert_to_tensor(data_desc), [-1, 1])

    data_fac = list(
        map(process_fac, data['type'], data['accommodates'], data['bathrooms'],
            data['bedrooms'], data['amenities']))
    data_fac = list(map(remove_special_symbols, data_fac))
    data_fac = tf.reshape(tf.convert_to_tensor(data_fac), [-1, 1])

    data_rev = list(
        map(process_rev, data['reviews'], data['review_rating'],
            data['review_scores_A'], data['review_scores_B'],
            data['review_scores_C'], data['review_scores_D']))
    data_rev = list(map(remove_special_symbols, data_rev))
    data_rev = tf.reshape(tf.convert_to_tensor(data_rev), [-1, 1])

    # extract features
    elmo = hub.load('src/elmo_3')

    desc_list = []
    fac_list = []
    rev_list = []
    for idx in range(num_data):
        print(f'processing {idx} / {num_data - 1}')
        desc_list.append(elmo.signatures['default'](data_desc[idx])['default'])
        fac_list.append(elmo.signatures['default'](data_fac[idx])['default'])
        rev_list.append(elmo.signatures['default'](data_rev[idx])['default'])

    desc_embed = tf.concat(desc_list, axis=0)
    fac_embed = tf.concat(fac_list, axis=0)
    rev_embed = tf.concat(rev_list, axis=0)

    feat_embed = tf.concat([desc_embed, fac_embed, rev_embed], axis=1)
    feat_embed = torch.tensor(feat_embed.numpy())
    torch.save(feat_embed, csv_fname[:-4] + '_feat_embed.pt')


if __name__ == '__main__':
    csv_fname = sys.argv[-1]
    data = data_processing(csv_fname)
