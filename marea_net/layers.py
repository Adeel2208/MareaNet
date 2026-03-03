"""Custom layers for MAREA-Net architecture."""

import tensorflow as tf
from tensorflow.keras import layers


class SimAM(layers.Layer):
    """SimAM: Simple, Parameter-Free Attention Module."""
    
    def __init__(self, lam=1e-4, **kwargs):
        super().__init__(**kwargs)
        self.lam = lam
    
    def call(self, x, training=None):
        mu = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        var = tf.math.reduce_variance(x, axis=[1, 2], keepdims=True)
        e_inv = tf.square(x - mu) / (4.0 * (var + self.lam) + self.lam)
        return x * tf.nn.sigmoid(e_inv)
    
    def get_config(self):
        config = super().get_config()
        config["lam"] = self.lam
        return config


class StripPooling(layers.Layer):
    """Strip Pooling for capturing long-range dependencies."""
    
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.h_conv = layers.Conv2D(channels, 1, padding='same', activation='relu')
        self.v_conv = layers.Conv2D(channels, 1, padding='same', activation='relu')
        self.fuse_conv = layers.Conv2D(channels, 1, padding='same')
    
    def call(self, x, training=None):
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        h_feat = tf.tile(self.h_conv(tf.reduce_mean(x, axis=2, keepdims=True)), [1, 1, w, 1])
        v_feat = tf.tile(self.v_conv(tf.reduce_mean(x, axis=1, keepdims=True)), [1, h, 1, 1])
        combined = h_feat + v_feat
        return combined * tf.nn.sigmoid(self.fuse_conv(combined))
    
    def get_config(self):
        config = super().get_config()
        config["channels"] = self.channels
        return config


class SBCC(layers.Layer):
    """Spatial-Boundary Channel Calibration."""
    
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        mid = max(channels // reduction, 8)
        self.gap = layers.GlobalAveragePooling2D(keepdims=True)
        self.fc1 = layers.Dense(mid, activation='relu', kernel_initializer='he_normal')
        self.fc2 = layers.Dense(channels, activation='sigmoid', kernel_initializer='zeros')
        self.atten_conv = layers.Conv2D(channels, 1, padding='same', activation='sigmoid')
    
    def call(self, x, training=None):
        excite = self.fc2(self.fc1(tf.squeeze(self.gap(x), axis=[1, 2])))
        excite = tf.reshape(excite, [tf.shape(x)[0], 1, 1, self.channels])
        return x * excite * self.atten_conv(x)
    
    def get_config(self):
        config = super().get_config()
        config["channels"] = self.channels
        return config


class WDTS(layers.Layer):
    """Wavelet-inspired Dual-scale Texture Sharpening."""
    
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.dw3 = layers.DepthwiseConv2D(3, padding='same', activation='relu')
        self.dw5 = layers.DepthwiseConv2D(5, padding='same', activation='relu')
        self.fuse = layers.Conv2D(channels, 1, padding='same')
        self.gate = layers.Conv2D(channels, 1, padding='same', activation='sigmoid')
        self.bn = layers.BatchNormalization()
    
    def call(self, x, training=None):
        combined = self.bn(self.fuse(self.dw3(x) + self.dw5(x)))
        return x * self.gate(combined) + x
    
    def get_config(self):
        return super().get_config()


class CGAFusion(layers.Layer):
    """Cross-scale Gated Attention Fusion."""
    
    def __init__(self, channels, reduction=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        mid = max(channels // reduction, 8)
        self.ca_fc1 = layers.Dense(mid, activation='relu', kernel_initializer='he_normal')
        self.ca_fc2 = layers.Dense(channels, kernel_initializer='zeros')
        self.sa_conv = layers.Conv2D(1, 7, padding='same')
        self.refine_conv1 = layers.Conv2D(channels, 1, padding='same')
        self.refine_conv2 = layers.Conv2D(channels, 7, padding='same')
        self.proj_skip = layers.Conv2D(channels, 1, padding='same')
        self.proj_decoder = layers.Conv2D(channels, 1, padding='same')
        self.out_conv = layers.Conv2D(channels, 3, padding='same', activation='relu')
        self.bn = layers.BatchNormalization()
    
    def call(self, decoder_feat, skip_feat, training=None):
        skip = self.proj_skip(skip_feat)
        dec = self.proj_decoder(decoder_feat)
        combined = skip + dec
        
        gap = tf.reduce_mean(combined, axis=[1, 2])
        ca = self.ca_fc2(self.ca_fc1(gap))
        ca = tf.reshape(ca, [tf.shape(combined)[0], 1, 1, self.channels])
        
        sa = self.sa_conv(tf.concat([
            tf.reduce_mean(combined, axis=-1, keepdims=True),
            tf.reduce_max(combined, axis=-1, keepdims=True)
        ], axis=-1))
        
        w = tf.nn.sigmoid(self.refine_conv2(self.refine_conv1(
            tf.concat([combined, ca + sa], axis=-1))))
        fused = w * skip + (1.0 - w) * dec
        return self.bn(self.out_conv(fused), training=training)
    
    def get_config(self):
        config = super().get_config()
        config["channels"] = self.channels
        return config


class MAREADecoderBlock(layers.Layer):
    """MAREA Decoder Block with optional SBCC and WDTS modules."""
    
    def __init__(self, filters, use_sbcc=False, use_wdts=False, dropout=0.15, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.use_sbcc = use_sbcc
        self.use_wdts = use_wdts
        
        self.up = layers.UpSampling2D(2, interpolation='bilinear')
        self.cga = CGAFusion(filters)
        self.conv1 = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')
        self.bn2 = layers.BatchNormalization()
        self.drop = layers.Dropout(dropout)
        self.sbcc = SBCC(filters) if use_sbcc else None
        self.wdts = WDTS(filters) if use_wdts else None
    
    def call(self, x, skip, training=None):
        x = self.up(x)
        x = tf.cast(tf.image.resize(x, [tf.shape(skip)[1], tf.shape(skip)[2]], 
                                     method='bilinear'), x.dtype)
        fused = self.cga(x, skip, training=training)
        fused = tf.nn.relu(self.bn1(self.conv1(fused), training=training))
        
        if self.sbcc is not None:
            fused = self.sbcc(fused, training=training)
        if self.wdts is not None:
            fused = self.wdts(fused, training=training)
        
        fused = tf.nn.relu(self.bn2(self.conv2(fused), training=training))
        return self.drop(fused, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "use_sbcc": self.use_sbcc,
            "use_wdts": self.use_wdts
        })
        return config


class ASPP_SP(layers.Layer):
    """Atrous Spatial Pyramid Pooling with Strip Pooling."""
    
    def __init__(self, out_channels=256, rates=(6, 12, 18), **kwargs):
        super().__init__(**kwargs)
        self.out_channels = out_channels
        
        self.conv1x1 = layers.Conv2D(out_channels, 1, padding='same', 
                                      kernel_initializer='he_normal')
        self.bn0 = layers.BatchNormalization()
        
        self.atrous_convs = [
            layers.Conv2D(out_channels, 3, padding='same', dilation_rate=r,
                         kernel_initializer='he_normal') for r in rates
        ]
        self.atrous_bns = [layers.BatchNormalization() for _ in rates]
        
        self.strip_pool = StripPooling(out_channels)
        self.strip_conv = layers.Conv2D(out_channels, 1, padding='same', 
                                        kernel_initializer='he_normal')
        self.strip_bn = layers.BatchNormalization()
        
        self.fuse_conv = layers.Conv2D(out_channels, 1, padding='same', 
                                       kernel_initializer='he_normal')
        self.fuse_bn = layers.BatchNormalization()
        self.fuse_drop = layers.Dropout(0.1)
        self.simam = SimAM()
    
    def call(self, x, training=None):
        b0 = tf.nn.relu(self.bn0(self.conv1x1(x), training=training))
        branches = [b0] + [
            tf.nn.relu(bn(conv(x), training=training))
            for conv, bn in zip(self.atrous_convs, self.atrous_bns)
        ]
        
        sp = tf.nn.relu(self.strip_bn(
            self.strip_conv(self.strip_pool(x, training=training)), 
            training=training))
        branches.append(sp)
        
        out = tf.nn.relu(self.fuse_bn(
            self.fuse_conv(tf.concat(branches, axis=-1)), 
            training=training))
        return self.simam(self.fuse_drop(out, training=training))
    
    def get_config(self):
        config = super().get_config()
        config["out_channels"] = self.out_channels
        return config
