import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from kerastuner import HyperModel

from Models.utils.pipeline import build_feature_fusion_layer, build_micro_exp_spatial_feature_extractor, build_micro_exp_temporal_inconsistency_detector, build_spatial_feature_extractor, build_temporal_feature_extractor

class FaceSwapHyperModel(HyperModel):
    def build(self, hp):
        # Inputs
        facial_frames = layers.Input(shape=(224, 224, 3), name='facial_frames')
        micro_expression_frames = layers.Input(shape=(64, 64, 3), name='micro_expression_frames')
        
        # Hyperparameters for spatial feature extractor
        spatial_dropout = hp.Float('spatial_dropout', min_value=0.3, max_value=0.7, step=0.1)
        spatial_model = build_spatial_feature_extractor()  # Assuming this function is defined as per your code
        for layer in spatial_model.layers[-30:]:
            layer.trainable = True
        spatial_features = spatial_model(facial_frames)
        
        # Hyperparameters for micro-expression spatial feature extractor
        micro_exp_dropout = hp.Float('micro_exp_dropout', min_value=0.3, max_value=0.7, step=0.1)
        micro_exp_spatial_model = build_micro_exp_spatial_feature_extractor()
        micro_exp_spatial_features = micro_exp_spatial_model(micro_expression_frames)
        
        # Reshape and tile spatial features
        reshaped_spatial = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(spatial_features)
        reshaped_spatial = layers.Lambda(lambda x: tf.tile(x, [1, 30, 1]))(reshaped_spatial)
        
        reshaped_micro_exp_spatial = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(micro_exp_spatial_features)
        reshaped_micro_exp_spatial = layers.Lambda(lambda x: tf.tile(x, [1, 30, 1]))(reshaped_micro_exp_spatial)
        
        # Temporal feature extractor hyperparameters
        temporal_lstm_units_1 = hp.Int('temporal_lstm_units_1', min_value=64, max_value=256, step=64)
        temporal_lstm_units_2 = hp.Int('temporal_lstm_units_2', min_value=32, max_value=128, step=32)
        
        temporal_model = build_temporal_feature_extractor()  # Define this function accordingly
        temporal_features = temporal_model(reshaped_spatial)
        
        # Micro-expression temporal inconsistency detector hyperparameters
        micro_exp_temporal_lstm_units_1 = hp.Int('micro_exp_temporal_lstm_units_1', min_value=64, max_value=256, step=64)
        micro_exp_temporal_lstm_units_2 = hp.Int('micro_exp_temporal_lstm_units_2', min_value=32, max_value=128, step=32)
        
        micro_exp_temporal_model = build_micro_exp_temporal_inconsistency_detector()
        micro_exp_temporal_features = micro_exp_temporal_model(reshaped_micro_exp_spatial)
        
        # Feature Fusion
        concatenated_feature_vector = build_feature_fusion_layer(
            spatial_features, 
            temporal_features, 
            micro_exp_spatial_features, 
            micro_exp_temporal_features
        )
        
        # Face Swap Detection Model
        dense_units = hp.Choice('dense_units', values=[256, 128, 64])
        x_face_swap = layers.GlobalAveragePooling1D()(concatenated_feature_vector)
        
        for unit in [dense_units]:
            x_face_swap = layers.Dense(unit, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x_face_swap)
            x_face_swap = layers.Dropout(spatial_dropout)(x_face_swap)
        
        op_face_swap = layers.Dense(1, activation='sigmoid')(x_face_swap)
        
        # Compile Model
        model = models.Model(inputs=[facial_frames, micro_expression_frames], outputs=op_face_swap)
        
        model.compile(
            optimizer=optimizers.Adam(
                hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
