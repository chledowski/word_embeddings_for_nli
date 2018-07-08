from src.models.bilstm import bilstm
from src.models.cbow import cbow
from src.models.esim import esim

def build_model(config, data):
    if config['model'] == 'cbow':
        return cbow(config, data)
    if config['model'] == 'bilstm':
        return bilstm(config, data)
    if config['model'] == 'esim':
        return esim(config, data)