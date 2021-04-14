#!/usr/bin/python
# -*- coding: utf-8 -*-


def img_extent(x, y):
    import numpy as np

    return int(np.max([np.max(np.abs(x)), np.max(np.abs(y))]))


def project(X, projection='xy'):
    import numpy as np

    if projection == 'xy':
        x = X[0, :]
        y = X[1, :]
    if projection == 'xz':
        x = X[0, :]
        y = X[2, :]
    if projection == 'yz':
        x = X[1, :]
        y = X[2, :]
    return (x, y)


def bin_particles(
    X,
    quantity,
    projection='xy',
    res=1.,
    statistic='sum',
    extent=False,
    ):

    from scipy.stats import binned_statistic_2d
    import numpy as np
    (x, y) = project(X, projection=projection)
    if extent is False:
        extent = img_extent(x, y)
    (h, x_edges, y_edges, binno) = binned_statistic_2d(x, y, quantity,
            statistic=statistic, bins=np.linspace(-extent, extent,
            int(extent * 2. / res)))
    return h


def nearest_neighbour_density(
    X,
    weight=[False],
    k=5,
    njobs=12,
    ):

    from sklearn.neighbors import NearestNeighbors
    import numpy as np
    neigh = NearestNeighbors(n_neighbors=k, radius=1e30, n_jobs=njobs)
    neigh.fit(X)
    (dist, ind) = neigh.kneighbors(X, k, return_distance=True)
    volume = 4. / 3. * np.pi * dist[:, -1] ** 3
    if weight[0] is not False:
        density = np.nansum(weight[ind], axis=1) / volume
    else:
        density = k / volume
    return (density, dist[:, -1])


def resample_particles(args):
    from fast_histogram import histogram2d
    import numpy as np

    # Unpack arguments (need to do this so it works with tqdm)

    (
        Xi,
        Qi,
        Qi2,
        Di,
        extent,
        res,
        n_resample,
        ) = args

    sample_points = np.random.normal(scale=Di, size=n_resample)
    if Qi is not False:
        if Qi.shape[1] > 0:
            dX = np.random.choice(sample_points, size=(2, Qi.shape[1]
                                  * n_resample))
            X2 = np.repeat(Xi, n_resample).reshape(2, Qi.shape[1]
                    * n_resample) + dX
            Q2 = np.repeat(Qi / n_resample,
                           n_resample).reshape(Qi.shape[0], Qi.shape[1]
                    * n_resample)

            h = np.asarray([histogram2d(X2[0, :], X2[1, :],
                           weights=Q2i, bins=int(extent * 2. / res)
                           - 1, range=[[-extent, extent], [-extent,
                           extent]]) for Q2i in Q2])

            # add to the shared histogram

            global shared_h
            tmp = np.ctypeslib.as_array(shared_h)
            tmp += h

    if Qi2 is not False:
        if Qi2.shape[1] > 0:
            dX = np.random.choice(sample_points, size=(2, Qi2.shape[1]
                                  * n_resample))
            X2 = np.repeat(Xi, n_resample).reshape(2, Qi2.shape[1]
                    * n_resample) + dX
            Q2 = np.repeat(Qi2, n_resample).reshape(Qi2.shape[0],
                    Qi2.shape[1] * n_resample)
            h2 = np.asarray([histogram2d(X2[0, :], X2[1, :],
                            weights=Q2i, bins=int(extent * 2. / res)
                            - 1, range=[[-extent, extent], [-extent,
                            extent]]) for Q2i in Q2])

            h3 = np.asarray([histogram2d(X2[0, :], X2[1, :],
                            bins=int(extent * 2. / res) - 1,
                            range=[[-extent, extent], [-extent,
                            extent]]) for Q2i in Q2])

            # add to the shared histogram

            global shared_h_2
            tmp2 = np.ctypeslib.as_array(shared_h_2)
            tmp2 += h2

            global shared_h_3
            tmp3 = np.ctypeslib.as_array(shared_h_3)
            tmp3 += h3


def smooth_3d(
    X,
    quantity_sum=[False],
    quantity_average=[False],
    res=1.,
    upper_threshold=False,
    extent=False,
    lower_threshold=False,
    njobs=8,
    nsteps=250,
    k=5,
    n_resample=500,
    projection='xy',
    verbose=True,
    antialias=True,
    ):
    '''
       purpose:
           adaptively smooths sparsely sampled particles according to their local density. 
           Similar to the approach described in Merritt+2020, Section 3.1
           (https://ui.adsabs.harvard.edu/abs/2020MNRAS.495.4570M/abstract)
       inputs:
           X:            particle co-ordinates: array shape=(3, Nparticles)
           quantity_sum: particle quantities to be summed: array shape=(Nquantities, Nparticles); ignore if [False]
           quantity_average: particle quantities to be averaged: array shape=(Nquantities, Nparticles); ignore if [False]
           res:          desired size of resolution unit in same units as X
           extent:       desired size of the image in the same units as X
           upper_threshold: number density of particles above which they will no longer be smoothed
           lower_threshold: number density of particles below which they will no longer be smoothed
           njobs:        number of workers to assign
           nsteps:       total number of density bins to do smoothing over, keep this fairly large or the result will lose accuracy
           k:            k nearest neighbour density estimate
           n_resample:   number of sub-particles to split each particle into for smoothing
           projection:   axis of projection used to produce the images
           antialias:    specifies whether the final image is antialiased
        outputs:
            img:         smoothed image for each summed quantity: shape = (Nquantities, Npixels, Npixels)
            average_img: smoothed image for each averaged quantity: shape = (Nquantities, Npixels, Npixels)
    '''

    assert quantity_sum[0] is not False or quantity_average[0] \
        is not False, \
        'both quantity_sum and quantity_average cannot be false'

    from multiprocessing import Pool, sharedctypes
    import tqdm
    import numpy as np
    from fast_histogram import histogram2d
    from scipy.ndimage import gaussian_filter

    (x, y) = project(X, projection=projection)
    if extent is False:
        extent = img_extent(x, y)

    hsize = int(extent * 2. / res)
    if verbose:
        print ('      calculating local density of each particle...')
    (rho, dist) = nearest_neighbour_density(X.T, k=k, njobs=njobs)
    d_element = rho / res ** 3
    if (upper_threshold is not False) & (lower_threshold is not False):
        pick = (d_element < upper_threshold) & (d_element
                > lower_threshold)
    if (upper_threshold is not False) & (lower_threshold is False):
        pick = d_element < upper_threshold
    if (upper_threshold is False) & (lower_threshold is not False):
        pick = d_element > lower_threshold
    if (upper_threshold is False) & (lower_threshold is False):
        pick = d_element > 0
    sort = np.argsort(rho[pick])
    sample_points = np.asarray([np.random.normal(scale=np.median(s),
                               size=n_resample * 10) for s in
                               np.array_split(dist[pick][sort],
                               nsteps)])
    Ds = np.asarray([np.median(s) for s in
                    np.array_split(dist[pick][sort], nsteps)])
    Xs = np.array_split(np.vstack([x[pick], y[pick]])[:, sort], nsteps,
                        axis=-1)
    if quantity_sum[0] is not False:
        Qs = np.array_split(quantity_sum[:, pick][:, sort], nsteps,
                            axis=-1)
    else:
        Qs = [False] * nsteps
    if quantity_average[0] is not False:
        Qs2 = np.array_split(quantity_average[:, pick][:, sort],
                             nsteps, axis=-1)
    else:
        Qs2 = [False] * nsteps
    args = []
    for (Xi, Qi, Qi2, Si) in zip(Xs, Qs, Qs2, Ds):
        args.append([
            Xi,
            Qi,
            Qi2,
            Si,
            extent,
            res,
            n_resample,
            ])

    if verbose:
        print ('      smoothing particle distribution...')
    if quantity_sum[0] is not False:
        global shared_h
        result = \
            np.ctypeslib.as_ctypes(np.zeros((quantity_sum.shape[0],
                                   hsize - 1, hsize - 1)))
        shared_h = sharedctypes.RawArray(result._type_, result)

    if quantity_average[0] is not False:
        global shared_h_2
        result_2 = \
            np.ctypeslib.as_ctypes(np.zeros((quantity_average.shape[0],
                                   hsize - 1, hsize - 1)))
        shared_h_2 = sharedctypes.RawArray(result_2._type_, result_2)
        global shared_h_3
        result_3 = \
            np.ctypeslib.as_ctypes(np.zeros((quantity_average.shape[0],
                                   hsize - 1, hsize - 1)))
        shared_h_3 = sharedctypes.RawArray(result_3._type_, result_3)

    pool = Pool(processes=njobs)
    with pool as p:
        if verbose:
            job = \
                np.asarray(list(tqdm.tqdm(p.imap_unordered(resample_particles,
                           args), total=len(args))))
        else:
            job = np.asarray(list(p.imap_unordered(resample_particles,
                             args)))

    # now put the images together

    sigma = (1. if antialias else 0.)
    if quantity_sum[0] is not False:
        h = np.ctypeslib.as_array(shared_h)
        img = []
        for i in range(Qs[0].shape[0]):
            h_ld = h[i]
            if np.count_nonzero(~pick) > 0:
                h_hd = histogram2d(x[~pick], y[~pick],
                                   weights=quantity_sum[i][~pick],
                                   bins=int(extent * 2. / res) - 1,
                                   range=[[-extent, extent], [-extent,
                                   extent]])
                h_ld[~np.isfinite(h_ld)] = 0.
                h_hd[~np.isfinite(h_hd)] = 0.
                img.append(gaussian_filter(h_ld + h_hd, sigma))
            else:
                h_ld[~np.isfinite(h_ld)] = 0.
                img.append(gaussian_filter(h_ld, sigma))
    else:
        img = np.nan

    if quantity_average[0] is not False:
        h2 = np.ctypeslib.as_array(shared_h_2)
        h3 = np.ctypeslib.as_array(shared_h_3)

        average_img = []
        for i in range(Qs2[0].shape[0]):
            h_ld = h2[i]
            if np.count_nonzero(~pick) > 0:
                h_hd = histogram2d(x[~pick], y[~pick],
                                   weights=quantity_average[i][~pick],
                                   bins=int(extent * 2. / res) - 1,
                                   range=[[-extent, extent], [-extent,
                                   extent]])
                n_hd = histogram2d(x[~pick], y[~pick], bins=int(extent
                                   * 2. / res) - 1, range=[[-extent,
                                   extent], [-extent, extent]])

                h_ld[~np.isfinite(h_ld)] = 0.
                h_hd[~np.isfinite(h_hd)] = 0.

                average_img.append(gaussian_filter((h_ld + h_hd)
                                   / (h3[i] + n_hd), sigma))
            else:
                h_ld[~np.isfinite(h_ld)] = 0.
                average_img.append(gaussian_filter(h_ld / h3[i], sigma))
    else:
        average_img = np.nan

    return (img, average_img)
