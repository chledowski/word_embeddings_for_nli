from src.models.bilstm import bilstm
from src.models.cbow import cbow
from src.models.esim import esim

def build_model(config):
    if config['model'] == 'cbow':
        return cbow(config)
    if config['model'] == 'bilstm':
        return bilstm(config)
    if config['model'] == 'esim':
        return esim(config)