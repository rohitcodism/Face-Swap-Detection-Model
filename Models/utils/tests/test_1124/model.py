import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, GlobalAveragePooling1D, GlobalMaxPooling1D,
    Concatenate, GlobalAveragePooling2D, Flatten, LSTM
)
from tensorflow.keras.models import Model

# Define feature extraction backbones (dummy examples)
def build_spatial_feature_extractor(input_shape):
    spatial_input = Input(shape=input_shape, name='Spatial_Input')
    x = Conv2D(32, (3, 3), activation='relu')(spatial_input)
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=spatial_input, outputs=x, name='Spatial_Feature_Extractor')

def build_temporal_feature_extractor(input_shape):
    temporal_input = Input(shape=input_shape, name='Temporal_Input')
    x = LSTM(64, return_sequences=True)(temporal_input)
    return Model(inputs=temporal_input, outputs=x, name='Temporal_Feature_Extractor')

def build_micro_expression_spatial_feature_extractor(input_shape):
    micro_exp_spatial_input = Input(shape=input_shape, name='MicroExp_Spatial_Input')
    x = Conv2D(32, (3, 3), activation='relu')(micro_exp_spatial_input)
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=micro_exp_spatial_input, outputs=x, name='MicroExp_Spatial_Feature_Extractor')

def build_micro_expression_temporal_feature_extractor(input_shape):
    micro_exp_temporal_input = Input(shape=input_shape, name='MicroExp_Temporal_Input')
    x = LSTM(64, return_sequences=True)(micro_exp_temporal_input)
    return Model(inputs=micro_exp_temporal_input, outputs=x, name='MicroExp_Temporal_Feature_Extractor')

# Feature fusion layer
def build_feature_fusion_layer(spatial_features, temporal_features, micro_exp_spatial_features, micro_exp_temporal_features):
    # Apply Dense layers to generate fixed-size vectors
    spatial_features_dense = Dense(512, activation='relu')(spatial_features)
    temporal_features_dense = Dense(512, activation='relu')(GlobalAveragePooling1D()(temporal_features))
    micro_exp_spatial_features_dense = Dense(512, activation='relu')(micro_exp_spatial_features)
    micro_exp_temporal_features_dense = Dense(512, activation='relu')(GlobalAveragePooling1D()(micro_exp_temporal_features))
    
    # Concatenate all processed features
    concatenated_features = Concatenate()([
        spatial_features_dense,
        temporal_features_dense,
        micro_exp_spatial_features_dense,
        micro_exp_temporal_features_dense
    ])
    return concatenated_features

# Final classification model
def build_deepfake_detection_model(spatial_input_shape, temporal_input_shape, micro_exp_spatial_input_shape, micro_exp_temporal_input_shape, num_classes=2):
    # Feature extractors
    spatial_feature_extractor = build_spatial_feature_extractor(spatial_input_shape)
    temporal_feature_extractor = build_temporal_feature_extractor(temporal_input_shape)
    micro_exp_spatial_feature_extractor = build_micro_expression_spatial_feature_extractor(micro_exp_spatial_input_shape)
    micro_exp_temporal_feature_extractor = build_micro_expression_temporal_feature_extractor(micro_exp_temporal_input_shape)
    
    # Inputs
    spatial_input = Input(shape=spatial_input_shape, name='Spatial_Input')
    temporal_input = Input(shape=temporal_input_shape, name='Temporal_Input')
    micro_exp_spatial_input = Input(shape=micro_exp_spatial_input_shape, name='MicroExp_Spatial_Input')
    micro_exp_temporal_input = Input(shape=micro_exp_temporal_input_shape, name='MicroExp_Temporal_Input')
    
    # Extract features
    spatial_features = spatial_feature_extractor(spatial_input)
    temporal_features = temporal_feature_extractor(temporal_input)
    micro_exp_spatial_features = micro_exp_spatial_feature_extractor(micro_exp_spatial_input)
    micro_exp_temporal_features = micro_exp_temporal_feature_extractor(micro_exp_temporal_input)
    
    # Fuse features
    fused_features = build_feature_fusion_layer(
        spatial_features, temporal_features, 
        micro_exp_spatial_features, micro_exp_temporal_features
    )
    
    # Classification head
    x = Dense(256, activation='relu')(fused_features)
    x = Dense(128, activation='relu')(x)
    output = Dense(num_classes, activation='softmax', name='Classification_Output')(x)
    
    # Define model
    model = Model(
        inputs=[spatial_input, temporal_input, micro_exp_spatial_input, micro_exp_temporal_input],
        outputs=output,
        name='Deepfake_Detection_Model'
    )
    return model