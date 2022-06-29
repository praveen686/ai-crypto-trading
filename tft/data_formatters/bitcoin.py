import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class BitcoinFormatter(GenericDataFormatter):
    _column_definition = []

    def __init__(self):
        pass

