import numpy as np
import tensorflow as tf
import cv2 as cv
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio ,structural_similarity,mean_squared_error

class Data_Processing:
    @staticmethod
    def process_base_data(SR_data):
        SR_Data_Processed = []
        Metadata = []
        for key in SR_data:
            rainfall_array_final = [] 
            meta = []
            for j in range(SR_data[key].shape[0]):
                k = SR_data[key][j][1:,:]
                k = k[:,3:-4]
                rainfall_array_final.append(k)
                meta.append([key,j])
            rainfall_array_final = np.array(rainfall_array_final)
            SR_Data_Processed.extend(rainfall_array_final)
            Metadata.extend(meta)
        return SR_Data_Processed, Metadata
    
    @staticmethod
    def downsample(image):
        max_pooled_image = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(image)
        downsampled_image = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(max_pooled_image) 
        return downsampled_image

    @staticmethod
    def Generating_Low_resolution_data(SR_data):
        LR = []
        for i in range(len(SR_data)):
            nan_value = -9999
            filled_data = np.nan_to_num(SR_data[i], nan=nan_value)
            input_image = filled_data.reshape((1,128,128,1))
            downsampled_image = np.array(Data_Processing.downsample(input_image))
            downsampled_image[downsampled_image==nan_value] =np.nan
            LR.append(downsampled_image.reshape((32,32)))
        return LR
    
    @staticmethod
    def calculate_daily_max(SR_data):
        daily_max = []
        for i in range(len(SR_data)):
            daily_max.append(np.nanmax(SR_data[i], axis=(0,1)))
        return daily_max
    
    @staticmethod
    def normalize_with_daily_max(SR_data, daily_max):
        mask = np.isnan(SR_data[0])
        mask = mask.astype('uint8') * 255
        SR = []
        for i in range(len(SR_data)):
            max_value = daily_max[i]
            if max_value != 0:
                image = np.copy(SR_data[i])
                image /= max_value
                image = image.astype('float32')
                dst = cv.inpaint(image, mask, 3, cv.INPAINT_TELEA)
                dst = np.clip(dst, 0, 1)
                SR.append(dst)
        return SR

    
    @staticmethod
    def check_for_nans(dataset):
        to_be_removed = []
        for i in range(len(dataset)):
            if np.isnan(dataset[i]).any():
                to_be_removed.append(i)
        return to_be_removed
    
    @staticmethod
    def remove_nans(dataset,meta_data, to_be_removed):
        final_dataset = []
        final_meta_data = []
        for i in range(len(dataset)):
            if i not in to_be_removed:
                final_dataset.append(dataset[i])
                final_meta_data.append(meta_data[i])
        return final_dataset, final_meta_data
    
    @staticmethod
    def generate_dataset(SR_Data_Processed, Metadata):
        print('Step 1 completed')
        daily_max = Data_Processing.calculate_daily_max(SR_Data_Processed)
        print('Step 2 completed')
        SR_final = Data_Processing.normalize_with_daily_max(SR_Data_Processed, daily_max)
        LR_final = Data_Processing.Generating_Low_resolution_data(SR_final)
        print('Step 3 completed')
        to_be_removed = Data_Processing.check_for_nans(SR_final)
        print('Step 4 completed')
        SR_final, metadata = Data_Processing.remove_nans(SR_final, Metadata, to_be_removed)
        print('Step 5 completed')
        LR_final, metadata = Data_Processing.remove_nans(LR_final, Metadata, to_be_removed)
        return SR_final, LR_final, metadata, daily_max
    
    @staticmethod
    def create_mask(SR_data):
        SR_mask = np.isnan(SR_data)
        LR = Data_Processing.Generating_Low_resolution_data(SR_data)
        LR_mask = np.isnan(LR)
        return SR_mask, LR_mask
    
    @staticmethod
    def visualize_LR_data(LR_data,label='Low Resolution Rainfall Data'):
        lat1 = np.load('/kaggle/input/coordinates/1lat.npy')
        lon1 = np.load('/kaggle/input/coordinates/1lon.npy')
        lon1 = lon1[1:-1]
        print(len(lat1),len(lon1))
        print(LR_data.shape)
        plt.figure(figsize=(10, 6))
        X_LR, Y_LR = np.meshgrid(lon1, lat1)
        plt.contourf(X_LR, Y_LR, LR_data, cmap='Blues')
        plt.colorbar(label='Rainfall (mm/day)')
        plt.title(label)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
    
    @staticmethod
    def visualize_SR_data(SR_data,label='High Resolution Rainfall Data'):
        lat25 = np.load('/kaggle/input/coordinates/0.25lat.npy')
        lon25 = np.load('/kaggle/input/coordinates/0.25lon.npy')
        lon25 = lon25[4:-4]
        print(len(lat25),len(lon25))
        plt.figure(figsize=(10, 6))
        X_SR, Y_SR = np.meshgrid(lon25, lat25)
        plt.contourf(X_SR, Y_SR, SR_data, cmap='Blues')
        plt.colorbar(label='Rainfall (mm/day)')
        plt.title(label)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
        
    @staticmethod
    def visualize_LR_masked_data(LR_data,mask,label='unmasked Low Resolution Rainfall Data'):
        rainfall = np.copy(LR_data)
        rainfall[mask] = np.nan
        lat1 = np.load('/kaggle/input/coordinates/1lat.npy')
        lon1 = np.load('/kaggle/input/coordinates/1lon.npy')
        lon1 = lon1[1:-1]
        print(len(lat1),len(lon1))
        print(LR_data.shape)
        plt.figure(figsize=(10, 6))
        X_LR, Y_LR = np.meshgrid(lon1, lat1)
        plt.contourf(X_LR, Y_LR, rainfall, cmap='Blues')
        plt.colorbar(label='Rainfall (mm/day)')
        plt.title(label)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
    
    @staticmethod
    def visualize_SR_masked_data(SR_data,mask,label = 'unmasked High Resolution Rainfall Data'):
        rainfall = np.copy(SR_data)
        rainfall[mask] = np.nan
        lat25 = np.load('/kaggle/input/coordinates/0.25lat.npy')
        lon25 = np.load('/kaggle/input/coordinates/0.25lon.npy')
        lon25 = lon25[4:-4]
        print(len(lat25),len(lon25))
        plt.figure(figsize=(10, 6))
        X_SR, Y_SR = np.meshgrid(lon25, lat25)
        plt.contourf(X_SR, Y_SR, rainfall, cmap='Blues')
        plt.colorbar(label='Rainfall (mm/day)')
        plt.title(label)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()
    
    @staticmethod
    def visualize_hr(SR_data,SR_vanilla,Generator):
        upsampled_image = tf.keras.layers.UpSampling2D(size=(2, 2))(SR_vanilla)
        upsampled_image = tf.keras.layers.UpSampling2D(size=(2, 2))(upsampled_image)
        generated = Generator(SR_data).numpy() 
        array = np.copy(generated[0])
        mask = np.isnan(upsampled_image)
        array[mask] = np.nan
        lat25 = np.load('/kaggle/input/coordinates/0.25lat.npy')
        lon25 = np.load('/kaggle/input/coordinates/0.25lon.npy')
        lon = np.arange(len(lon25) * 4)
        lat = np.arange(len(lat25) * 4)
        X_SR, Y_SR = np.meshgrid(lon, lat)
        plt.figure(figsize=(7, 6))   
        plt.contourf(X_SR, Y_SR,array , cmap='Blues')
        plt.colorbar(label='Rainfall (mm/day)')
        plt.title('Generated High Resolution Rainfall Data')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()

class Training:
    @staticmethod
    def gradient_penalty(discriminator, real_images, fake_images):
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated_images = alpha * real_images + (1 - alpha) * fake_images

        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            predictions = discriminator(interpolated_images)

        gradients = tape.gradient(predictions, interpolated_images)
        gradients_sqr_sum = tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3])
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum + 1e-8)
        gradient_penalty = tf.reduce_mean((gradient_l2_norm - 1.0) ** 2)

        return gradient_penalty
    
    @staticmethod
    def discriminator_loss(real_output, fake_output, gradient_penalty):
        real_loss = tf.reduce_mean(real_output)
        fake_loss = tf.reduce_mean(fake_output)
        total_loss = fake_loss - real_loss + gradient_penalty
        return total_loss
    
    @staticmethod
    def generator_loss(fake_output, real_images, fake_images, feature_extractor,adv_c=0.1,fe_c = 1,mse_c = 0.01):
        adv_loss = -tf.reduce_mean(fake_output)
        feature_loss = 0
        real_feature = feature_extractor(real_images)
        fake_feature = feature_extractor(fake_images)
        for real, fake in zip(real_feature, fake_feature):
            feature_loss += tf.reduce_mean(tf.abs(real - fake))
        mse_loss = tf.reduce_mean(tf.square(real_images - fake_images))
        total_loss =  adv_c*adv_loss + fe_c*feature_loss+mse_c* mse_loss
        return total_loss
    
    @staticmethod
    def calculate_metrics(Generator, SR_data, LR_data,SR_mask):
        PSNR = []
        SSIM = []
        MSE = []

        def generate_data():
            for i in range(len(SR_data)):
                SR_data_ = np.copy(SR_data[i].reshape(128, 128))
                LR_data_ = np.copy(LR_data[i].reshape(32, 32))
                yield LR_data_, SR_data_

        dataset = tf.data.Dataset.from_generator(
            generate_data,
            output_signature=(
                tf.TensorSpec(shape=(32, 32), dtype=tf.float32),
                tf.TensorSpec(shape=(128, 128), dtype=tf.float32)
            )
        )
        dataset = dataset.batch(64) 

        for lr_batch, sr_batch in dataset:
            sr_images = Generator(lr_batch)
            sr_images_np = sr_images.numpy()

            for i in range(sr_images_np.shape[0]):
                gt_image_np = sr_batch[i].numpy().reshape(128, 128)
                sr_image_np = sr_images_np[i].reshape(128, 128)
                gt_image_np[SR_mask] = 0
                sr_image_np[SR_mask] = 0
                data_range = gt_image_np.max() - gt_image_np.min()
                psnr_value = peak_signal_noise_ratio(gt_image_np, sr_image_np, data_range=data_range)
                PSNR.append(psnr_value)
                ssim_value = structural_similarity(gt_image_np, sr_image_np, multichannel=False, data_range=data_range)
                SSIM.append(ssim_value)
                mse_value = mean_squared_error(gt_image_np, sr_image_np)
                MSE.append(mse_value)

        return np.mean(PSNR), np.mean(SSIM), np.mean(MSE)
    
    @staticmethod
    def plot_metrics(PSNR, SSIM, MSE):
        epochs = np.arange(1, len(PSNR) + 1)

        plt.figure(figsize=(18, 6))

        # Plot PSNR
        plt.subplot(1, 3, 1)
        plt.bar(epochs, PSNR, color='b', alpha=0.6, label='PSNR')
        plt.plot(epochs, PSNR, color='b', marker='o', linestyle='-', linewidth=2, label='PSNR Line')
        plt.xlabel('Epoch')
        plt.ylabel('PSNR')
        plt.title('PSNR per Epoch')
        plt.legend()
        plt.grid(True)

        # Plot SSIM
        plt.subplot(1, 3, 2)
        plt.bar(epochs, SSIM, color='r', alpha=0.6, label='SSIM')
        plt.plot(epochs, SSIM, color='r', marker='o', linestyle='-', linewidth=2, label='SSIM Line')
        plt.xlabel('Epoch')
        plt.ylabel('SSIM')
        plt.title('SSIM per Epoch')
        plt.legend()
        plt.grid(True)

        # Plot MSE
        plt.subplot(1, 3, 3)
        plt.bar(epochs, MSE, color='g', alpha=0.6, label='MSE')
        plt.plot(epochs, MSE, color='g', marker='o', linestyle='-', linewidth=2, label='MSE Line')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.title('MSE per Epoch')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()