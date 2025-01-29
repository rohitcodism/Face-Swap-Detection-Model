import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Multiply, Lambda, Activation, Conv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, GlobalAveragePooling1D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Attention, RNN, Input, RandomFlip, RandomRotation, RandomZoom, RandomContrast, RandomBrightness
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2

data_augmentation = tf.keras.Sequential([
    RandomFlip('horizontal_and_vertical'),
    RandomRotation(0.1),  # Rotate by ±10%
    RandomZoom(0.1),      # Zoom by ±10%
    RandomContrast(0.2),  # Adjust contrast by ±20%
    RandomBrightness(0.2),# Adjust brightness by ±20%
    Lambda(lambda x: x + tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.01)),  # Add Gaussian noise
    Lambda(lambda x: tf.nn.depthwise_conv2d(x, tf.random.normal([3, 3, 3, 1], mean=0.0, stddev=1.0), strides=[1, 1, 1, 1], padding='SAME')),  # Apply Gaussian blur
], name='data_augmentation')

# Define models
def build_spatial_feature_extractor():
    base_model = ResNet50(include_top=False, weights='imagenet', pooling='avg', input_shape=(224,224,3))
    spatial_model = Model(inputs=base_model.input, outputs=base_model.output, name='spatial_feature_extractor')

    for layer in spatial_model.layers[-30:]:  # Unfreeze last 20 layers, for example
        layer.trainable = True

    return spatial_model

def build_temporal_feature_extractor():
    input_seq = Input(shape=(30,2048))

    lstm_1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))
    lstm_2 = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))

    lstm_out = lstm_1(input_seq)
    lstm_out = lstm_2(lstm_out)

    model = Model(inputs=input_seq, outputs=lstm_out, name='temporal_feature_extractor')

    return model

def build_micro_exp_spatial_feature_extractor():
    spatial_inputs = Input(shape=(64,64,3))
    micro_exp_x = Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(1e-4))(spatial_inputs)
    micro_exp_x = BatchNormalization()(micro_exp_x)
    micro_exp_x = Activation('relu')(micro_exp_x)
    micro_exp_x = Dropout(0.6)(micro_exp_x)

    micro_exp_x = MaxPooling2D(pool_size=(2,2))(micro_exp_x)

    micro_exp_x = Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(1e-4))(micro_exp_x)
    micro_exp_x = BatchNormalization()(micro_exp_x)
    micro_exp_x = Activation('relu')(micro_exp_x)
    micro_exp_x = Dropout(0.6)(micro_exp_x)

    micro_exp_x = MaxPooling2D(pool_size=(2,2))(micro_exp_x)

    micro_exp_x = Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(1e-4))(micro_exp_x)
    micro_exp_x = BatchNormalization()(micro_exp_x)
    micro_exp_x = Activation('relu')(micro_exp_x)
    micro_exp_x = Dropout(0.6)(micro_exp_x)

    micro_exp_x = MaxPooling2D(pool_size=(2,2))(micro_exp_x)

    micro_exp_x = Flatten()(micro_exp_x)

    micro_exp_x = Dense(256, activation='relu')(micro_exp_x)
    micro_exp_x = Dropout(0.6)(micro_exp_x)
    micro_exp_output = Dense(128, activation='relu')(micro_exp_x)

    micro_exp_spatial_feature_extractor = Model(inputs=spatial_inputs, outputs=micro_exp_output, name='micro_exp_spatial_feature_extractor')

    return micro_exp_spatial_feature_extractor

def build_micro_exp_temporal_inconsistency_detector():
    temp_inputs = Input(shape=(30,128))
    
    x_mic_exp = Bidirectional(LSTM(128, return_sequences=True, dropout=0.5))(temp_inputs)

    x_mic_exp = Bidirectional(LSTM(64, return_sequences=True, dropout=0.5))(x_mic_exp)

    # attention_output = Attention()([x_mic_exp, x_mic_exp])

    # x_mic_exp = Dense(256, activation='relu')(x_mic_exp)
    # x_mic_exp = Dense(128, activation='relu')(x_mic_exp)

    mic_exp_temp_model = Model(inputs=temp_inputs, outputs=x_mic_exp, name='micro_exp_temporal_feature_extractor')

    return mic_exp_temp_model

def build_spatial_attention_mechanism(feature_maps):
    """
    :param feature_maps: 
    :return: weighted feature maps
    """

    expanded_tensor = Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, axis=1),axis=2))(feature_maps)
    
    attention_map = Conv2D(1, kernel_size=(1,1), strides=(1,1), padding='same')(expanded_tensor)
    
    attention_map = Activation('sigmoid')(attention_map) # 'sigmoid' or 'softmax' can be used as an activation function
    
    # Element wise multiplication of feature_maps and attention_map
    weighted_feature_map = Multiply()([feature_maps, attention_map])
    
    # Convert the weighted feature map into a context vector
    spatial_context_vectors = GlobalAveragePooling2D()(weighted_feature_map)
    
    return spatial_context_vectors

def build_temporal_attention_mechanism(feature_maps):
    """
    :param feature_maps: 
    :return weighted_feature_maps: 
    """
    
    temporal_attention_scores = Dense(1, activation='tanh')(feature_maps)
    
    temporal_attention_weights = Activation('sigmoid')(temporal_attention_scores)
    
    weighted_temporal_features = Multiply()([feature_maps, temporal_attention_weights])
    
    context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(weighted_temporal_features)
    
    return context_vector

def build_spatial_micro_expression_attention_mechanism(micro_exp_spatial_feature_maps):
    """
    :param micro_exp_spatial_feature_maps: 
    :return weighted micro_exp feature maps : 
    """

    reshaped_map = Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, axis=1),axis=2))(micro_exp_spatial_feature_maps)
    
    attention_map = Conv2D(1,(1,1),strides=(1,1),padding="same")(reshaped_map)
    
    attention_map = Activation('sigmoid')(attention_map)
    
    weighted_micro_exp_feature_map = Multiply()([micro_exp_spatial_feature_maps,attention_map])
    
    micro_exp_spatial_context_vector = GlobalAveragePooling2D()(weighted_micro_exp_feature_map)
    
    return micro_exp_spatial_context_vector

def build_temporal_micro_expression_attention_mechanism(micro_exp_feature_vectors):
    """
    :param micro_exp_feature_vectors: 
    :return micro_exp_context_vectors: 
    """
    
    attention_scores = Dense(1,activation='tanh')(micro_exp_feature_vectors)
    
    attention_weights = Activation('sigmoid')(attention_scores)
    
    weighted_micro_exp_temporal_features = Multiply()([attention_weights, micro_exp_feature_vectors])
    
    micro_exp_context_vector = Lambda(lambda x:tf.reduce_sum(x, axis=1))(weighted_micro_exp_temporal_features)
    
    return micro_exp_context_vector

def build_feature_fusion_layer(spatial_features, temporal_features, micro_exp_spatial_features, micro_exp_temporal_features):
    # spatial_context_vectors = build_spatial_attention_mechanism(feature_maps=spatial_features)

    # temporal_context_vector = build_temporal_attention_mechanism(feature_maps=temporal_features)

    # micro_exp_spatial_context_vector = build_spatial_micro_expression_attention_mechanism(micro_exp_spatial_feature_maps=micro_exp_spatial_features)

    # micro_exp_temporal_context_vector = build_temporal_micro_expression_attention_mechanism(micro_exp_feature_vectors=micro_exp_temporal_features)
    
    reshaped_spatial_features = Lambda(lambda x: tf.expand_dims(x, axis=1))(spatial_features)  # (None, 1, 2048)
    reshaped_spatial_features = Lambda(lambda x: tf.tile(x, [1, 30, 1]))(reshaped_spatial_features)  # (None, 30, 2048)

    reshaped_micro_exp_spatial_features = Lambda(lambda x: tf.expand_dims(x, axis=1))(micro_exp_spatial_features)  # (None, 1, 128)
    reshaped_micro_exp_spatial_features = Lambda(lambda x: tf.tile(x, [1, 30, 1]))(reshaped_micro_exp_spatial_features)  # (None, 30, 128)

    concatenated_feature_vector = Concatenate()([
        reshaped_spatial_features,
        temporal_features,
        reshaped_micro_exp_spatial_features,
        micro_exp_temporal_features
    ])
    return concatenated_feature_vector

def build_face_swap_detection_model(concatenated_feature_vector):
    dense_units = [256,128,64]

    x_face_swap = GlobalAveragePooling1D()(concatenated_feature_vector)

    for unit in dense_units:
        x_face_swap = Dense(unit, activation='relu', kernel_regularizer=l2(1e-4))(x_face_swap)
        x_face_swap = Dropout(0.6)(x_face_swap)

    op_face_swap = Dense(1, activation='sigmoid')(x_face_swap)

    # face_swap_detector_model = Model(inputs=concatenated_feature_vector, outputs=op_face_swap)
    
    return op_face_swap

# Combine everything into a single model pipeline
def build_full_model():
    # Inputs
    facial_frames = data_augmentation(Input(shape=(224, 224, 3)))  # Example input shape for facial_frames
    micro_expression_frames = data_augmentation(Input(shape=(64, 64, 3)))  # Example input shape for micro_expression_frames
    
    # Feature Extractors
    spatial_model = build_spatial_feature_extractor()
    micro_exp_spatial_model = build_micro_exp_spatial_feature_extractor()
    temporal_model = build_temporal_feature_extractor()
    micro_exp_temporal_model = build_micro_exp_temporal_inconsistency_detector()
    
    # Extract features
    spatial_features = spatial_model(facial_frames)
    micro_exp_spatial_features = micro_exp_spatial_model(micro_expression_frames)

    reshaped_spatial = Lambda(lambda x: tf.expand_dims(x, axis=1))(spatial_features)
    reshaped_spatial = Lambda(lambda x: tf.tile(x, [1, 30, 1]))(reshaped_spatial)  # Create a sequence dimension with repeated frames

    reshaped_micro_exp_spatial = Lambda(lambda x: tf.expand_dims(x, axis=1))(micro_exp_spatial_features)
    reshaped_micro_exp_spatial = Lambda(lambda x: tf.tile(x, [1, 30, 1]))(reshaped_micro_exp_spatial)
    
    # Temporal features
    temporal_features = temporal_model(reshaped_spatial)
    micro_exp_temporal_features = micro_exp_temporal_model(reshaped_micro_exp_spatial)
    
    # Feature Fusion
    concatenated_feature_vector = build_feature_fusion_layer(
        spatial_features, 
        temporal_features, 
        micro_exp_spatial_features, 
        micro_exp_temporal_features
    )
    
    # Face Swap Detection
    face_swap_model = build_face_swap_detection_model(concatenated_feature_vector)
    
    # Build full model
    full_model = Model(inputs=[facial_frames, micro_expression_frames], outputs=face_swap_model, name='full_model')
    
    return full_model