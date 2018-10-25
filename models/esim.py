
import logging

import keras.backend as K
from keras.layers import Activation, Dropout, Input, Lambda, Dot, Concatenate
from keras.models import Model

from modules.esim import EmbeddingLayer, EncoderLayer, ExternalKnowledgeLayer, \
    FeedForwardLayer, InferenceLayer, PoolingLayer, ProjectionLayer
from modules.residual import ResidualLayer
from modules.restorer import Restorer
from modules.utils import MaskMultiply

logger = logging.getLogger(__name__)


class ESIM(object):
    """
    ESIM model implementation.
    """

    @classmethod
    def from_config(cls, config, embeddings):
        """
        Builds ESIM model from config.

        :param config: ``dict`` containing model configuration
        :param embeddings: ``dict`` containing embedding matrices keyed by names.
        :return: ``Model`` object.
        """

        # Prepare inputs
        premise_input = Input(shape=(None,), dtype='int32', name='sentence1')
        premise = Lambda(lambda x: x, name='premise')(premise_input)
        premise_mask_input = Input(shape=(None,), dtype='int32', name='sentence1_mask')
        premise_mask = Lambda(lambda x: K.cast(x, 'float32'))(premise_mask_input)

        hypothesis_input = Input(shape=(None,), dtype='int32', name='sentence2')
        hypothesis = Lambda(lambda x: x, name='hypothesis')(hypothesis_input)
        hypothesis_mask_input = Input(shape=(None,), dtype='int32', name='sentence2_mask')
        hypothesis_mask = Lambda(lambda x: K.cast(x, 'float32'))(hypothesis_mask_input)

        premise_knowledge_input = Input(shape=(None, None, 5), dtype='float32', name='KBph')
        premise_knowledge = Lambda(lambda x: x)(premise_knowledge_input)
        hypothesis_knowledge_input = Input(shape=(None, None, 5), dtype='float32', name='KBhp')
        hypothesis_knowledge = Lambda(lambda x: x)(hypothesis_knowledge_input)

        # Set up model input & output
        model_inputs = [premise_input, premise_mask_input,
                       hypothesis_input, hypothesis_mask_input]

        # TODO(tomwesolowski): Get rid of this flag.
        if config['read_knowledge']:
            model_inputs += [premise_knowledge_input, hypothesis_knowledge_input]

        model_outputs = []

        # Create embeddings layers.
        embedded = dict()

        for emb_name, emb_config in config['embeddings'].items():
            embedding_layer = EmbeddingLayer.from_config(
                name=emb_name,
                config=emb_config,
                embeddings=embeddings
            )
            embedded[emb_name] = {
                'premise': embedding_layer(premise),
                'hypothesis': embedding_layer(hypothesis)
            }

        premise = Dropout(config["dropout"])(embedded['main']['premise'])
        hypothesis = Dropout(config["dropout"])(embedded['main']['hypothesis'])

        # Runs input encoding - Bidirectonal LSTM.
        input_encoder = EncoderLayer.from_config(config['encoder'])

        embedded['contextual'] = {
            'premise': input_encoder(premise),
            'hypothesis': input_encoder(hypothesis),
        }

        premise = embedded['contextual']['premise']
        hypothesis = embedded['contextual']['hypothesis']

        # Apply residual connection when needed.
        if config.get('residual_connection') is not None:
            residual_connection = ResidualLayer.from_config(config['residual_connection'])
            residual_embedding = config['residual_connection']['embedding']
            premise = residual_connection(
                contextualized=premise,
                residual=embedded[residual_embedding]['premise']
            )
            hypothesis = residual_connection(
                contextualized=hypothesis,
                residual=embedded[residual_embedding]['hypothesis']
            )

        # Restore contextualized embeddings
        if config.get('restorer') is not None:
            restorer = Restorer.from_config(config['restorer'])
            dot_restored = restorer.restore(
                premise_contextualized=embedded['contextual']['premise'],
                hypothesis_contextualized=embedded['contextual']['hypothesis']
            )
            dot_restored_target = Dot(axes=(2, 2), name='dot_restored_target')([
                embedded['main']['premise'],
                embedded['main']['hypothesis']
            ])
            # Without this lambda dot_restored_target is pruned from graph.
            dot_restored = Lambda(lambda x: x[0], name='dot_restored')(
                [dot_restored, dot_restored_target]
            )
            model_outputs.append(dot_restored)

        # Mask padding.
        premise = MaskMultiply()([premise, premise_mask])
        hypothesis = MaskMultiply()([hypothesis, hypothesis_mask])

        # Inference layer with attention mechanism.
        inference_layer = InferenceLayer.from_config(config['inference'])

        (premise_knowledge_vector, hypothesis_knowledge_vector,
         premise_soft_attention, hypothesis_soft_attention) = inference_layer(
            inputs=[premise, hypothesis],
            knowledge=[premise_knowledge, hypothesis_knowledge],
        )

        # External knowledge layer.
        premise_external_knowledge_vector = None
        hypothesis_external_knowledge_vector = None

        if config.get('external_knowledge') is not None:
            external_knowledge_layer = ExternalKnowledgeLayer.from_config(
                config['external_knowledge'])

            external_knowledge_embedding = embedded[
                config['external_knowledge'].get('embedding', 'contextual')]

            (premise_external_knowledge_vector,
             hypothesis_external_knowledge_vector) = external_knowledge_layer(
                inputs=[external_knowledge_embedding['premise'],
                        external_knowledge_embedding['hypothesis']],
                knowledge=[premise_knowledge, hypothesis_knowledge],
                soft_attention=[premise_soft_attention, hypothesis_soft_attention]
            )

        if (premise_external_knowledge_vector is not None
                and hypothesis_external_knowledge_vector is not None):
            premise_knowledge_vector += [premise_external_knowledge_vector]
            hypothesis_knowledge_vector += [hypothesis_external_knowledge_vector]

        # Project knowledge vector to reduce dimensionality.
        projection = ProjectionLayer.from_config(config['projection'])

        premise = projection(Concatenate()(premise_knowledge_vector))
        hypothesis = projection(Concatenate()(hypothesis_knowledge_vector))

        premise = Dropout(config["dropout"])(premise)
        hypothesis = Dropout(config["dropout"])(hypothesis)

        if (premise_external_knowledge_vector is not None
                and hypothesis_external_knowledge_vector is not None):
            premise = Concatenate()([premise, premise_external_knowledge_vector])
            hypothesis = Concatenate()([hypothesis, hypothesis_external_knowledge_vector])

        # Run inference encoder - Bidirectonal LSTM.
        inference_encoder = EncoderLayer.from_config(config['inference_encoder'])

        embedded['inference'] = {
            'premise': inference_encoder(premise),
            'hypothesis': inference_encoder(hypothesis),
        }

        premise = embedded['inference']['premise']
        hypothesis = embedded['inference']['hypothesis']

        # Mask padding once again.
        premise = MaskMultiply()([premise, premise_mask])
        hypothesis = MaskMultiply()([hypothesis, hypothesis_mask])

        pooling = PoolingLayer.from_config(config['pooling'])
        inference_final_vector = pooling(
            inputs=[premise, hypothesis],
            masks=[premise_mask, hypothesis_mask]
        )

        inference_final_vector = Dropout(config["dropout"])(inference_final_vector)

        # Run MLP to get final prediction.
        num_layers = len(config['feed_forwards'])
        for i in range(num_layers):
            ff_layer = FeedForwardLayer.from_config(config['feed_forwards'][i])
            inference_final_vector = ff_layer(inference_final_vector)
            if i+1 < num_layers:
                inference_final_vector = Dropout(config["dropout"])(inference_final_vector)

        prediction = Activation('linear', name='prediction')(inference_final_vector)
        model_outputs.insert(0, prediction)

        return Model(inputs=model_inputs, outputs=model_outputs)