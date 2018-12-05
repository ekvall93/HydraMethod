import argparse
from .utils import get_fullpath_basename, is_valid_file
from typing import TextIO
import os
import csv
from typing import List
import numpy as np





def pad_factory(MAX_LENGTH: int):
    """
    >>> f = pad_factory(15)
    >>> f(["OxM", "A", "C", "A", "T"])
    [21, 1, 2, 1, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    """
    def pad_encode_peptide_ary(p: list):
        """
        >>> pad_encode_peptide_ary(["OxM", "A", "C", "A", "T"], 7)
        [21, 1, 2, 1, 17, 0, 0]
        """
        rt_ary = []
        for i in p:
            rt_ary.append(ALPHABET[i])
        while len(rt_ary) < MAX_LENGTH:
            rt_ary.append(0)

        return(rt_ary)
    return pad_encode_peptide_ary


def read_csv(fileOpener: TextIO) -> peptideVector:
    peptides = [i[0] for i in csv.reader(open(fileOpener, 'r'))]
    return peptides


if __name__ == "__main__":
    MAX_LENGTH = 30

    f = pad_factory(MAX_LENGTH)
    dir_, filebase_, _ = get_fullpath_basename("/root/data.csv")
    peptideList = read_csv("/root/data.csv")
    # peptideList = pd.read_csv(fullPath)["Modified sequence"].tolist()

    X = [np.array(f(peptide_parser_ary(i, MAX_LENGTH, 'mq'))) for i in peptideList]

    file_name = "_".join([filebase_, "tensor.npy"])
    out_path = os.path.join(dir_, file_name)
    np.save(out_path, X)

class transformData():
    def __init__(self,in_path, *args, **kwargs):
        is_valid_file(in_path)
        self.in_path = in_path
        self.out_path = out_path
        super(transformData, self).__init__(*args, **kwargs)

    peptideVector = List[str]

    ALPHABET = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5,
                'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
                'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
                'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20,
                'OxM': 21, 'CaC': 2,
                'M(ox)': 21}    # M(ox) in ProteomeTools
    ALPHABET_S = {integer: char for char, integer in ALPHABET.items()}

    def peptide_parser_pt(self, p: str):
        """
        >>> p = "-OxM-A-BOT-"
        >>> [i for i in peptide_parser_pt(p)]
        ['OxM', 'A', 'BOT']
        """
        if p[0] == '(':
            raise ValueError("sequence starts with '('")
        n = len(p)
        i = 0
        while i < n:
            if p[i] == "-":
                j = p[(i + 1):].index('-')
                offset = i + j + 1
                yield p[(i + 1):offset]
                i = offset + 1
            else:
                yield p[i]
                i += 1

    def peptide_parser_ary(self, p: str, MAX_LENGTH: int, input_type: str):
        """
        >>> peptide_parser_ary("-OxM-A-BOT-", 30, 'pt')
        ['OxM', 'A', 'BOT']
        >>> peptide_parser_ary("-OxM-A-BOT-", 1, 'pt')
        Traceback (most recent call last):
        ValueError: peptide ['OxM', 'A', 'BOT'] is too long by 3 in comparison to 1
        """
        X = [i for i in self.peptide_parser_pt(p)]
        if len(X) > MAX_LENGTH:
            raise ValueError('peptide {} is too long by {} in comparison to {}'.format(X, len(X), MAX_LENGTH))
        else:
            return X
