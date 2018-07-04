#!/usr/bin/env python

'''
Downloads Text8 dataset as zip file and unpacks it.
'''
from src.util.get_data import get_data

if __name__ == '__main__':
    get_data('http://mattmahoney.net/dc/text8.zip', 'text8')

