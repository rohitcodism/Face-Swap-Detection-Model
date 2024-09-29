import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Input, Dense, Dropout, Concatenate, Multiply, Lambda, Activation,
    Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, TimeDistributed,
    GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Bidirectional, Attention
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

# Define models
def build_spatial_feature_extractor():
    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3))
    spatial_model = Model(inputs=base_model.input, outputs=base_model.output)
    
    for layer in spatial_model.layers[-10:]:  # Unfreeze last 10 layers, for example
        layer.trainable = True

    return spatial_model

def build_temporal_feature_extractor():
    input_seq = Input(shape=(30,2048))

    lstm_1 = LSTM(128, return_sequences=True, dropout=0.2)
    lstm_2 = LSTM(64, return_sequences=True, dropout=0.2)

    lstm_out = Bidirectional(lstm_1)(input_seq)
    lstm_out = Bidirectional(lstm_2)(lstm_out)

    model = Model(inputs=input_seq, outputs=lstm_out)

    return model

def build_micro_exp_spatial_feature_extractor():
    spatial_inputs = Input(shape=(64,64,3))
    micro_exp_x = Conv2D(32, (3,3), padding='same')(spatial_inputs)
    micro_exp_x = BatchNormalization()(micro_exp_x)
    micro_exp_x = Activation('relu')(micro_exp_x)

    micro_exp_x = MaxPooling2D(pool_size=(2,2))(micro_exp_x)

    micro_exp_x = Conv2D(64, (3,3), padding='same')(micro_exp_x)
    micro_exp_x = BatchNormalization()(micro_exp_x)
    micro_exp_x = Activation('relu')(micro_exp_x)

    micro_exp_x = MaxPooling2D(pool_size=(2,2))(micro_exp_x)

    micro_exp_x = Conv2D(128, (3,3), padding='same')(micro_exp_x)
    micro_exp_x = BatchNormalization()(micro_exp_x)
    micro_exp_x = Activation('relu')(micro_exp_x)

    micro_exp_x = MaxPooling2D(pool_size=(2,2))(micro_exp_x)

    micro_exp_x = Flatten()(micro_exp_x)

    micro_exp_x = Dense(256, activation='relu')(micro_exp_x)
    micro_exp_output = Dense(128, activation='relu')(micro_exp_x)

    micro_exp_spatial_feature_extractor = Model(inputs=spatial_inputs, outputs=micro_exp_output)

    return micro_exp_spatial_feature_extractor

def build_micro_exp_temporal_inconsistency_detector():
    temp_inputs = Input(shape=(30,128))
    
    x_mic_exp = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(temp_inputs)
    x_mic_exp = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2))(x_mic_exp)

    x_mic_exp = Dense(256, activation='relu')(x_mic_exp)
    x_mic_exp = Dense(128, activation='relu')(x_mic_exp)

    mic_exp_temp_model = Model(inputs=temp_inputs, outputs=x_mic_exp)

    return mic_exp_temp_model

def build_feature_fusion_layer(spatial_features, temporal_features, micro_exp_spatial_features, micro_exp_temporal_features):
    concatenated_feature_vector = Concatenate(name='feature_fusion_layer')([
        spatial_features,
        temporal_features,
        micro_exp_spatial_features,
        micro_exp_temporal_features
    ])
    return concatenated_feature_vector

def build_face_swap_detection_model(concatenated_feature_vector):
    # Aggregate across temporal dimension
    aggregated_features = GlobalAveragePooling1D()(concatenated_feature_vector)  # Shape: (None, features)

    dense_units = [256, 128, 64]

    x_face_swap = aggregated_features

    for unit in dense_units:
        x_face_swap = Dense(unit, activation='relu', kernel_regularizer=l2(1e-4))(x_face_swap)
        x_face_swap = Dropout(0.5)(x_face_swap)

    op_face_swap = Dense(1, activation='sigmoid')(x_face_swap)

    return op_face_swap

# Combine everything into a single model pipeline
def build_full_model():
    # Inputs
    facial_frames = Input(shape=(30, 224, 224, 3), name='facial_frames')  # Sequence of 30 frames
    micro_expression_frames = Input(shape=(30, 64, 64, 3), name='micro_expression_frames')  # Sequence of 30 micro-expression frames
    
    # Feature Extractors with TimeDistributed
    spatial_model = build_spatial_feature_extractor()
    micro_exp_spatial_model = build_micro_exp_spatial_feature_extractor()

    spatial_features = TimeDistributed(spatial_model)(facial_frames)  # Shape: (None, 30, 2048)
    micro_exp_spatial_features = TimeDistributed(micro_exp_spatial_model)(micro_expression_frames)  # Shape: (None, 30, 128)
    
    # Temporal Features
    temporal_model = build_temporal_feature_extractor()
    micro_exp_temporal_model = build_micro_exp_temporal_inconsistency_detector()
    
    temporal_features = temporal_model(spatial_features)  # Shape: (None, 30, 256)
    micro_exp_temporal_features = micro_exp_temporal_model(micro_exp_spatial_features)  # Shape: (None, 30, 128)
    
    # Feature Fusion
    concatenated_feature_vector = build_feature_fusion_layer(
        spatial_features, 
        temporal_features, 
        micro_exp_spatial_features, 
        micro_exp_temporal_features
    )  # Shape: (None, 30, 2432)
    
    # Face Swap Detection
    face_swap_output = build_face_swap_detection_model(concatenated_feature_vector)  # Shape: (None, 1)
    
    # Build full model
    full_model = Model(inputs=[facial_frames, micro_expression_frames], outputs=face_swap_output)
    
    return full_model
