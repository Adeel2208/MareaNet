"""MAREA-Net model architecture."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .layers import ASPP_SP, MAREADecoderBlock
from .config import Config


def get_convnext_skip_outputs(encoder):
    """
    Resolve skip-connection outputs from ConvNeXtBase.
    
    ConvNeXtBase layout for 320×320 input:
    - Stem:   stride-4 Conv → 80×80 @ 128ch
    - Stage0: stride-1,  3 blocks → 80×80 @ 128ch
    - Stage1: stride-2,  3 blocks → 40×40 @ 256ch
    - Stage2: stride-2, 27 blocks → 20×20 @ 512ch
    - Stage3: stride-2,  3 blocks → 10×10 @ 1024ch
    
    Args:
        encoder: ConvNeXtBase model
        
    Returns:
        Tuple of (stem, stage0, stage1, stage2, stage3) outputs
    """
    def last_layer_matching(tag):
        candidates = [l for l in encoder.layers if tag in l.name]
        if not candidates:
            raise ValueError(f"No layer found with tag '{tag}'")
        return candidates[-1]
    
    stem_l = last_layer_matching('stem')
    stage0_l = last_layer_matching('stage_0')
    stage1_l = last_layer_matching('stage_1')
    stage2_l = last_layer_matching('stage_2')
    stage3_l = last_layer_matching('stage_3')
    
    print("\n🔍 ConvNeXtBase resolved skip layers:")
    for label, layer in [
        ("Stem   80×80  128ch", stem_l),
        ("Stage0 80×80  128ch", stage0_l),
        ("Stage1 40×40  256ch", stage1_l),
        ("Stage2 20×20  512ch", stage2_l),
        ("Stage3 10×10 1024ch", stage3_l),
    ]:
        print(f"  {label:24s} → {layer.name}")
    
    return (stem_l.output, stage0_l.output, stage1_l.output, 
            stage2_l.output, stage3_l.output)


def build_marea_net(config: Config = None):
    """
    Build MAREA-Net v5.5 with ConvNeXtBase backbone.
    
    Architecture:
    - ConvNeXtBase encoder with ImageNet weights
    - ASPP with Strip Pooling on bottleneck
    - 4-stage decoder with CGA-Fusion
    - Auxiliary head for deep supervision
    - Plant presence binary classification head
    
    Args:
        config: Configuration object. If None, uses default config.
        
    Returns:
        Keras Model with outputs: [seg_output, aux_output, plant_output]
    """
    if config is None:
        config = Config()
    
    img_h = config.get('model.input_height', 320)
    img_w = config.get('model.input_width', 320)
    n_cls = config.get('model.num_classes', 8)
    
    inp = layers.Input(shape=(img_h, img_w, 3), name="input_image")
    
    # ConvNeXtBase encoder
    encoder = keras.applications.ConvNeXtBase(
        include_top=False,
        weights='imagenet',
        input_tensor=inp,
        include_preprocessing=False,  # preprocessing done in data pipeline
    )
    
    skip1, skip2, skip3, skip4, bottleneck = get_convnext_skip_outputs(encoder)
    
    # ASPP + SimAM on bottleneck
    aspp = ASPP_SP(out_channels=256)(bottleneck)  # 10×10, 256ch
    
    # Auxiliary head (deep supervision)
    aux_out = layers.Conv2D(n_cls, 1, activation='softmax', name='aux_output')(
        layers.UpSampling2D(32, interpolation='bilinear')(aspp)
    )
    
    # Plant presence binary head
    plant_out = layers.Dense(1, activation='sigmoid', name='plant_output')(
        layers.Dense(64, activation='relu')(
            layers.GlobalAveragePooling2D()(aspp)
        )
    )
    
    # 4-stage decoder
    d4 = MAREADecoderBlock(256, use_sbcc=True, name='decoder_l4')(aspp, skip4)
    d3 = MAREADecoderBlock(128, use_wdts=True, name='decoder_l3')(d4, skip3)
    d2 = MAREADecoderBlock(64, name='decoder_l2')(d3, skip2)
    d1 = MAREADecoderBlock(32, name='decoder_l1')(d2, skip1)
    
    # Final head — TWO ×2 upsamples to reach 320×320
    seg = layers.UpSampling2D(2, interpolation='bilinear')(d1)  # 80 → 160
    seg = layers.Conv2D(32, 3, padding='same', activation='relu',
                        kernel_initializer='he_normal')(seg)
    seg = layers.BatchNormalization()(seg)
    
    seg = layers.UpSampling2D(2, interpolation='bilinear')(seg)  # 160 → 320
    seg = layers.Conv2D(16, 3, padding='same', activation='relu',
                        kernel_initializer='he_normal')(seg)
    seg = layers.BatchNormalization()(seg)
    
    seg_out = layers.Conv2D(n_cls, 1, activation='softmax', name='seg_output')(seg)
    
    model = keras.Model(
        inputs=inp,
        outputs=[seg_out, aux_out, plant_out],
        name="MAREA_Net_v5.5_ConvNeXtBase"
    )
    
    return model


def set_encoder_trainable(model, trainable: bool):
    """
    Freeze or unfreeze the ConvNeXt encoder.
    
    Args:
        model: MAREA-Net model
        trainable: If True, unfreeze encoder; if False, freeze it
    """
    for layer in model.layers:
        if 'convnext' in layer.name.lower():
            layer.trainable = trainable
            print(f"  {'🔓 Unfroze' if trainable else '🧊 Froze'} encoder: {layer.name}")
