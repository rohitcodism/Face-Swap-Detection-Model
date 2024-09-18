from pipeline import build_full_model

def deliver_model():
    model = build_full_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model