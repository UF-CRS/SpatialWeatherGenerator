# Usually we use `%matplotlib inline`. However we need `notebook` for the anim to render in the notebook.
import matplotlib

import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr

weather = xr.load_dataset("test_weather.nc")


fps = 4
nSeconds = 100
rain = weather["RAIN"].to_numpy()
tmax = weather["TMAX"].to_numpy()

# First set up the figure, the axis, and the plot element we want to animate
fig, axs = plt.subplots(figsize=(8, 8), ncols=2)

rain1 = rain[0]
tmax1 = tmax[0]
im_rain = axs[0].imshow(
    rain1, interpolation="none", aspect="auto", vmin=np.nanmin(rain), vmax=40
)
plt.colorbar(im_rain)
im_tmax = axs[1].imshow(
    tmax1,
    interpolation="none",
    aspect="auto",
    vmin=np.nanmin(tmax),
    vmax=np.nanmax(tmax),
)
plt.colorbar(im_tmax)


def animate_func(i):
    if i % fps == 0:
        print(".", end="")

    im_rain.set_array(rain[i])
    im_tmax.set_array(tmax[i])
    return [im_rain, im_tmax]


anim = animation.FuncAnimation(
    fig, animate_func, frames=nSeconds * fps, interval=1000 / fps  # in ms
)

anim.save("test_anim.mp4", fps=fps, extra_args=["-vcodec", "libx264"])

print("Done!")
