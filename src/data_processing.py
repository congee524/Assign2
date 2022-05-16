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


def process_review(reviews, rating, score_A, score_B, score_C, score_D):
    if reviews == '0' or reviews == '' or rating == '':
        return 'Has no review.'
    format_list = [
        int(float(item)) if item != '' else 0
        for item in [rating, reviews, score_A, score_B, score_C, score_D]
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

    data_review = list(
        map(process_review, data['reviews'], data['review_rating'],
            data['review_scores_A'], data['review_scores_B'],
            data['review_scores_C'], data['review_scores_D']))
    data_review = list(map(remove_special_symbols, data_review))
    data_review = tf.reshape(tf.convert_to_tensor(data_review), [-1, 1])

    elmo = hub.load('src/elmo_3')
    desc_embedding = elmo.signatures['default'](data_desc[0])['default']
    fac_embedding = elmo.signatures['default'](data_fac[0])['default']
    review_embedding = elmo.signatures['default'](data_review[0])['default']

    for idx in range(1, num_data):
        print(f'processing {idx} / {num_data - 1}')
        _desc = elmo.signatures['default'](data_desc[idx])['default']
        desc_embedding = tf.concat([desc_embedding, _desc], axis=0)

        _fac = elmo.signatures['default'](data_fac[idx])['default']
        fac_embedding = tf.concat([fac_embedding, _fac], axis=0)

        _review = elmo.signatures['default'](data_review[idx])['default']
        review_embedding = tf.concat([review_embedding, _review], axis=0)

    # desc_embedding = torch.tensor(desc_embedding.numpy())
    # fac_embedding = torch.tensor(fac_embedding.numpy())
    # review_embedding = torch.tensor(review_embedding.numpy())

    # torch.save(desc_embedding, csv_fname[:-4] + '_desc_embedding.pt')
    # torch.save(fac_embedding, csv_fname[:-4] + '_fac_embedding.pt')
    # torch.save(review_embedding, csv_fname[:-4] + '_review_embedding.pt')

    feat_embedding = tf.concat(
        [desc_embedding, fac_embedding, review_embedding], axis=1)
    feat_embedding = torch.tensor(feat_embedding.numpy())

    torch.save(feat_embedding, csv_fname[:-4] + '_feat_embedding.pt')


if __name__ == '__main__':
    csv_fname = sys.argv[-1]
    data = data_processing(csv_fname)