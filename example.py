import smooth_3d as s3d
import matplotlib.pyplot as plt
import numpy as np

# creating distribution of points
X = np.random.normal(0,20,size=(3, 100000))
flux = np.random.uniform(10,50,size=(1,100000))
Z = np.zeros_like(flux)
Z[:,X[1,:] > 0] = -1; Z[:,X[1,:] < 0] = 1; Z[:,(X[1,:] < 0) & (X[2,:] > 0)] = 0 

# binning without smoothing
imgs0 = s3d.bin_particles(X, flux, projection='yz', extent=120, res=0.5)
imgs0_Z = s3d.bin_particles(X, Z, projection='yz', extent=120, res=0.5, statistic='mean')
# binning with smoothing
imgs, imgs_Z = s3d.smooth_3d(X, flux, Z, projection='yz', njobs=4, nsteps=250, k=5,
                         n_resample=500, extent=120, res=0.5, upper_threshold=False)

# plotting result
fig, axs = plt.subplots(2,2,figsize=(8,8))
axs[0,0].imshow(np.log10(imgs0[0]), vmin=-1, vmax=2.5, cmap='inferno')
axs[0,0].axis('off');
axs[0,0].set_title('Sum')
axs[0,1].imshow((imgs0_Z[0]), vmin=-1, vmax=1, cmap='viridis')
axs[0,1].axis('off');
axs[0,1].set_title('Mean')
axs[1,0].imshow(np.log10(imgs[0]), vmin=-1, vmax=2.5, cmap='inferno')
axs[1,0].axis('off');
axs[1,1].imshow((imgs_Z[0]), vmin=-1, vmax=1, cmap='viridis')
axs[1,1].axis('off');

plt.savefig('example.png', bbox_inches='tight')