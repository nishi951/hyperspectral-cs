from matplotlib.colors import LinearSegmentedColormap
# Visualize hyperspectral images
def plot_all_channels(data, channels_to_wavelengths):
    """Plot the channels in the data, color coded by wavelength.
    channels_to_wavelengths[i] gives the wavelength in nm of the ith channel."""
    plt.figure(figsize=(24, 13))
    inchannels = data.shape[1]
    for channel in range(inchannels):
        wavelength = channels_to_wavelengths[channel]
        r, g, b = spectral_color(wavelength)
        
#         low = hsv_to_rgb(h, s, 0)
#         high = hsv_to_rgb(h, s, 1)
        cdict = {"red": ((0.0, 0.0, 0.0),
                         (1.0, r, r)),
                 "green": ((0.0, 0.0, 0.0),
                           (1.0, g, g)),
                 "blue":  ((0.0, 0.0, 0.0),
                           (1.0, b, b))
        }
        lambd = LinearSegmentedColormap('lambd', cdict)

        ax = plt.subplot(4, inchannels//4+1, channel+1)

        ax.imshow(data[0,channel,:,:], cmap=lambd)
        plt.axis('off')
        plt.tight_layout

# Adapted from
# https://stackoverflow.com/questions/3407942/rgb-values-of-visible-spectrum/22681410#22681410
def spectral_color(l): # RGB <0,1> <- lambda l <400,700> [nm]
    # R
    if l>=400.0 and l<410.0:
        t = (l-400.0)/(410.0-400.0)
        r = (0.33*t)-(0.20*t*t)
    elif l>=410.0 and l<475.0:
        t = (l-410.0)/(475.0-410.0)
        r = 0.14 - (0.13*t*t)
    elif l>=545.0 and l<595.0:
        t = (l-545.0)/(595.0-545.0)
        r = (1.98*t)-(     t*t)
    elif l>=595.0 and l<650.0:
        t = (l-595.0)/(650.0-595.0)
        r = 0.98+(0.06*t)-(0.40*t*t)
    elif l>=650.0 and l<700.0:
        t = (l-650.0)/(700.0-650.0)
        r = 0.65-(0.84*t)+(0.20*t*t)
    else:
        r = 0.0
    # G
    if l>=415.0 and l<475.0:
        t = (l-415.0)/(475.0-415.0)
        g = (0.80*t*t)
    elif l>=475.0 and l<590.0:
        t = (l-475.0)/(590.0-475.0)
        g = 0.8 +(0.76*t)-(0.80*t*t)
    elif l>=585.0 and l<639.0:
        t = (l-585.0)/(639.0-585.0)
        g = 0.84-(0.84*t)
    else:
        g = 0.0
    # B
    if l>=400.0 and l<475.0:
        t = (l-400.0)/(475.0-400.0)
        b = (2.20*t)-(1.50*t*t)
    elif l>=475.0 and l<560.0:
        t = (l -475.0)/(560.0-475.0)
        b = 0.7 - t+(0.30*t*t)
    else:
        b = 0.0
    return r, g, b