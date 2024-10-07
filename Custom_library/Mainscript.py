import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D,GlobalAveragePooling2D
from tensorflow.keras.models import Model
from IPython.display import Image
import cv2 as cv
import os
import pydot
from tensorflow.keras.layers import Dropout
from skimage.metrics import peak_signal_noise_ratio ,structural_similarity,mean_squared_error
import pickle
import sys
sys.path.append("/home/karthikl/Harish/Data_processing_zero.py")
from Data_processing_zero import Data_Processing as dp
from Data_processing_zero import Training as tn
from keras.models import load_model

def load_data():
    SR_file_path = "/scratch/karthikl_NEW/karthikl/Harish/data.pkl"
    with open(SR_file_path, 'rb') as file:
        SR_data = pickle.load(file)
    SR_mask = np.isnan(SR_data)
    SR_data_base,Metadata = dp.process_base_data(SR_data)
    SR_data_processed,LR_data_processed,Metadata,daily_max = dp.generate_dataset(SR_data_base,Metadata)
    
    return SR_data_processed, LR_data_processed,SR_mask

def generator():
    class DepthToSpaceLayer(tf.keras.layers.Layer):
        def __init__(self, block_size, **kwargs):
            super(DepthToSpaceLayer, self).__init__(**kwargs)
            self.block_size = block_size

        def call(self, inputs):
            return tf.nn.depth_to_space(inputs, self.block_size)
    def residual_block_gen(ch=64, k_s=3, st=1):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope = 0.2),
            tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(negative_slope = 0.2),
        ])
        return model
    def Upsample_block(x, ch=256, k_s=3, st=1):
        x = tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same')(x)
        x = DepthToSpaceLayer(block_size=2)(x)  
        x = tf.keras.layers.LeakyReLU(negative_slope = 0.2)(x)
        return x
    input_lr = tf.keras.layers.Input(shape=(None, None, 1))
    input_conv = tf.keras.layers.Conv2D(64, 3, padding='same')(input_lr)
    input_conv = tf.keras.layers.LeakyReLU(negative_slope = 0.2)(input_conv)
    Generator = input_conv
    for _ in range(8):
        res_output = residual_block_gen()(Generator)
        res_output_1 = residual_block_gen(64,5)(Generator)
        res_output_2 = residual_block_gen(64,7)(Generator)
        res_output_3= residual_block_gen(64,9)(Generator)
        Generator = tf.keras.layers.Add()([Generator, res_output,res_output_1,res_output_2,res_output_3])
    Generator = tf.keras.layers.Conv2D(64, 3, padding='same')(Generator)
    Generator = tf.keras.layers.BatchNormalization()(Generator)

    Generator = tf.keras.layers.Add()([Generator, input_conv])

    Generator = Upsample_block(Generator)  
    Generator = Upsample_block(Generator) 
    output_sr = tf.keras.layers.Conv2D(1, 3, activation='sigmoid', padding='same')(Generator)

    Generator = tf.keras.models.Model(input_lr, output_sr)
    
    return Generator
    
def Discriminator():
    def residual_block_disc(ch=64, k_s=3, st=1):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(ch, k_s, strides=(st, st), padding='same',
                                kernel_initializer=tf.keras.initializers.HeNormal()),
            tf.keras.layers.LeakyReLU(negative_slope=0.2),
            # Optional: Move BatchNormalization after activation or remove
            # tf.keras.layers.BatchNormalization(),
        ])
        return model

    input_lr = tf.keras.layers.Input(shape=(128, 128, 1))
    input_conv = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer=tf.keras.initializers.HeNormal())(input_lr)
    input_conv = tf.keras.layers.LeakyReLU(negative_slope=0.2)(input_conv)

    channel_nums = [64, 128, 128, 256, 256, 512, 512]
    stride_sizes = [2, 1, 2, 1, 2, 1, 2]

    disc = input_conv
    for x in range(7):
        disc = residual_block_disc(ch=channel_nums[x], st=stride_sizes[x])(disc)

    disc = tf.keras.layers.Flatten()(disc)
    #disc = tf.keras.layers.Droput(0.1)(disc)
    disc = tf.keras.layers.Dense(1024, kernel_initializer=tf.keras.initializers.HeNormal())(disc)
    disc = tf.keras.layers.LeakyReLU(negative_slope=0.2)(disc)
    disc_output = tf.keras.layers.Dense(1)(disc) 
    discriminator = tf.keras.models.Model(input_lr, disc_output)
    return discriminator

def autoencoder():
    def my_custom_loss(y_true, y_pred):
        ssim_loss = 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return 0.9 * ssim_loss + 0.1* mse_loss
    model = load_model('/scratch/karthikl_NEW/karthikl/Harish/autoencoder_fe (1).keras',custom_objects={'combined_loss': my_custom_loss})
    feature_extractor = Model(inputs=model.input, outputs=model.get_layer('latent_space').output)
    return feature_extractor
        

def training(Generator, discriminator, feature_extractor, SR_data_processed, LR_data_processed, SR_mask, epochs, batch_size):
    PSNR = []
    SSIM = []
    MSE = []

    generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.0, beta_2=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.0, beta_2=0.5)

    def train_step(SR_images, LR_images, batch_size):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = Generator(LR_images, training=True)
            real_output = discriminator(SR_images, training=True)
            fake_output = discriminator(fake_images, training=True)
            gp = tn.gradient_penalty(discriminator, SR_images, fake_images)
            gen_loss = tn.generator_loss(fake_output, SR_images, fake_images, feature_extractor)
            disc_loss = tn.discriminator_loss(real_output, fake_output, gp)
        
        # Compute gradients for the generator, discriminator
        gradients_of_generator = gen_tape.gradient(gen_loss, Generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        # Apply gradients to update the weights
        generator_optimizer.apply_gradients(zip(gradients_of_generator, Generator.trainable_variables ))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
        
        return gen_loss, disc_loss

    # Training loop
    def train(SR_data, LR_data,PSNR,SSIM, epochs, batch_size):
        for epoch in range(epochs):
            indices = np.arange(len(SR_data))
            np.random.shuffle(indices)
            for i in range(len(SR_data)//batch_size):
                batch = indices[i:i+batch_size]
                lr = np.array([LR_data[j] for j in batch]).reshape((batch_size, 32, 32, 1))
                sr = np.array([SR_data[j] for j in batch]).reshape((batch_size, 128, 128, 1))
                gen_loss, disc_loss = train_step(sr, lr, batch_size)
                if i%100==0 and i!=0:
                    print(f'{i} batches completed in epoch:{epoch+1}')
            print(f'Epoch {epoch+1}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}')
            Generator.save(f'/home/karthikl/Harish/generator_epoch_{epoch+1}.h5')
            print(f'Models saved after epoch {epoch+1}')
            psnr,ssim , mse = tn.calculate_metrics(Generator,SR_data,LR_data ,SR_mask[0])
            print(f'PSNR:{psnr},SSIM:{ssim} , MSE:{mse}')
            PSNR.append(psnr)
            SSIM.append(ssim)
            MSE.append(mse)
    train(SR_data_processed,LR_data_processed,PSNR,SSIM,epochs, batch_size)
    return PSNR,SSIM,MSE

def main():
    SR_data_processed, LR_data_processed, SR_mask = load_data()
    Generator = generator()
    discriminator = Discriminator()
    feature_extractor = autoencoder()
    epochs = 20
    batch_size = 32
    PSNR, SSIM, MSE = training(Generator, discriminator, feature_extractor, SR_data_processed, LR_data_processed, SR_mask, epochs, batch_size)
    print(PSNR, SSIM, MSE)

if __name__ == "__main__":
    main()