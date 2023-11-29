---
title: Mapping Paths and Gaze in Virtual Spaces using Structure from Motion
date: 2023-11-23 13:20:00 -0800
categories: [Computer Vision, Structure From Motion]
tags: [computer-vision, c++, video-analysis, sfm, math, tools]     # TAG names should always be lowercase
toc: true
img_path: /assets/img/sfm
---


## Motivating Problem

I was presented with a problem where we needed to understand how people moved around in VR and what they attended to.
However, only videos collected from the perspective of the person navigating the Google Earth VR environment were
available via screen recordings of video headset data. Ideally, we would have collected more than just videos of these
experiences; it would have been useful to know the position of the person recorded directly from Google Earth VR.
Unfortunately this was not available.

## General Solution

It turns out the field of Computer Vision has some tools that can help with this problem. Known as [Structure from
Motion (SfM)](https://en.wikipedia.org/wiki/Structure_from_motion), tools can employ this process to reconstruct the 3d
structure of a scene based on overlapping images of the scene, such as those on might collect from a video captured by
a GoPro cameras or the like. SfM can recreate environments, and subsequently indicate pathways through environments. 

### A Very Brief Explanation of Structure from Motion

In SfM, as the name of the technique implies, it is possible to derive the 3d structure of objects and the environment
by using video recorded by one or more cameras moving around the environment. As the camera is taking overlapping 2d
images of the objects from a variety of angles (the frames of the video), SfM extracts the shared information from those
images to create 3d reconstructions.

![Structure from Motion](sfm.png)
_3 images of a tree taken from different positions_

The steps involved in the SfM process are:
- Feature detection and extraction
- Feature matching and geometric verification
- Incremental structure and motion reconstruction (bundle adjustment)

With two images, and a pixel in each image that corresponds to the same 3d point in space, there exists a triangle in 3d
space, because the pixel points also represent 3d points in space where the light intersected the camera's sensor at the
time the photo was taken.

Once many such triangle correspondences are collected, it becomes possible to solve a large constraint problem and
derive where all the points are with respect to each other. This process is known as
[bundle adjustment](https://en.wikipedia.org/wiki/Bundle_adjustment)
and provides not only the 3d points on the object, but also the location and orientation of the camera in space when it
captured what it was looking at – also known as the camera's
[pose](https://en.wikipedia.org/wiki/Pose_(computer_vision)).

There are limitations to this process because we are dealing with imperfect quantized data. The correspondences found in
the images will almost never be an exact match to a single point in 3d. There are a number of reasons for this: pixels
by their very nature are discrete/averaged samples of the light coming into the camera’s sensor. In addition, the
camera's lens plays a part in how to interpret what light was received by the camera’s sensor and recorded as pixels.
Modeling aspects like the optical center and focal length of the camera all play into what are known as a camera’s
intrinsics and can be used to derive more accuracy from the calculations.

Also, when moving around the scene, light can vary over the surface of objects. For example, a bright spot seen from one
angle can be in a totally different 3d position when seen from a different angle due to reflection (i.e. the scene is
not [Lambertian](https://en.wikipedia.org/wiki/Lambertian_reflectance)). For this reason there are some clever ways of
looking at a neighborhood of pixels for other structure and trying to find matches on the entire neighborhood -- these
are called image or feature descriptors, examples include
[SIFT](https://www.vlfeat.org/api/sift.html#:~:text=The%20SIFT%20descriptor%20is%20a,orientation%20on%20the%20image%20plane.)
and [SURF](https://docs.opencv.org/3.4/df/dd2/tutorial_py_surf_intro.html). The process of deriving the descriptors in
the images is often known as "feature detection or feature extraction", and then finding the correspondences between
images is known as "feature matching".

This means that in the best case scenario everything will be a little off. In the worst case scenario, the algorithm can
mistakenly match two distinct 3d points instead of one shared point between the two images. This leads to the types of
errors that should be expected from the process -- both the 3d points and the camera pose can sometimes be very wrong.
The bundle adjustment is only attempting to minimize the error from matching all the correspondences against all the
other correspondences to the best of its ability -- there can still be some large errors.

To mitigate some of the issues of trying to fit all the correspondences (including the outliers), SfM usually
employs an additional set of steps for [Random Sample Consensus,
RANSAC](https://en.wikipedia.org/wiki/Random_sample_consensus#:~:text=Random%20sample%20consensus%20(RANSAC)%20is,as%20an%20outlier%20detection%20method.),
where different subsets of the correspondences are checked for consensus. However, errors will always be present and can
vary dramatically given the quality of the data.

To read more about all of this, there are a lot of great papers and resources cited in the [COLMAP
docs](https://colmap.github.io/bibliography.html).


### Structure from Motion Tools

There are a variety of good open source tools and libraries out there for solving this SfM problem. To list a few:

- [COLMAP](https://colmap.github.io/)
- [OpenCV](https://opencv.org/)
- [Bundler](https://www.cs.cornell.edu/~snavely/bundler/)
- [VisualSFM](http://ccwu.me/vsfm/)

I am not going to delve into the pros and cons of all of them in this post. The reason I ended up choosing
[COLMAP](https://colmap.github.io/)
was its support for GPU acceleration, which helped with processing speed, and the fact that it was being actively
maintained at the time of this writing.

In order to use COLMAP, you will need a COLMAP executable. The project maintains some [pre-built
executables](https://colmap.github.io/install.html#installation);
however, I ended up building my own which I will cover a bit later in this post.

There is a version of COLMAP that is pre-built with [CUDA](https://developer.nvidia.com/cuda-toolkit) support. CUDA is
a toolset released by NVidia for enabling computationally intensive tasks on your GPU. When a problem is massively
parallel, as is the case for the bundle adjustment, CUDA can speed it up considerably, but it requires you have an
NVidia GPU on which to run it. If you want to target older GPUs it might make sense to build COLMAP yourself with
support for an older [compute capability](https://en.wikipedia.org/wiki/CUDA#GPUs_supported).

A fun side-note I learned about COLMAP during this process is that it helps power [NeRF
Studio](https://docs.nerf.studio/), which is a tool for generating novel 2d views, using [Neural Radiance Fields or
NeRFs](https://en.wikipedia.org/wiki/Neural_radiance_field), from the 3d scene reconstruction problem we are solving
with SfM. It is possible to use NeRFs to capture dense 3d spaces and then fly around them somewhat arbitrarily.

## Using the Tools

### Use of the Terminal and Windows

For brevity, I will be assuming the use of Windows. While much of what I post is also possible from Linux or Mac, I will
not be covering the differences.

I am not sure how familiar the reader is with using the terminal, so I'll add a few brief notes below to get you started
if you're a beginner in this space -- and if you're already familiar, well feel free to skip the next paragraph or two.

The [Windows Terminal](https://apps.microsoft.com/detail/windows-terminal/9N0DX20HK701?hl=en-us&gl=US) is a good
terminal program from which to run various "shells". Shells are programs that allow a user to start other programs from
within them. This is usually accomplished by the shell providing a prompt, at which a command-line is typed in that will
execute another program. Output and errors from the program will then be passed to the shell which usually will show
them in the terminal window. After executing the program, the shell will display another prompt, and so the cycle
continues. Examples of shells in Windows include [Git Bash](https://gitforwindows.org/),
[Powershell](https://learn.microsoft.com/en-us/powershell/scripting/overview?view=powershell-7.4) and Windows built-in
[cmd](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/cmd).

When executing a command, the shell will usually check a specified list of directories for executable programs to see if
any match the name of the typed in command. This list of directories is usually known as the path. On Windows it is
possible to add new directories to the path by tweaking the environment variables, specifically one named
[PATH](https://www.maketecheasier.com/what-is-the-windows-path/). Often for the PATH to take effect, it is necessary to
restart the shell and/or terminal program.


### Extracting Image Frames with FFMpeg

The starting point for the SfM problem is a set of images that are looking at a scene. We had a bunch of videos. So
I first needed to get a set of images from a given video.

To do this I chose to use [ffmpeg](https://ffmpeg.org/), as it really is one of the most versatile tools for dealing
with anything video related. It powers [more of the
internet](https://en.wikipedia.org/wiki/FFmpeg#Projects_using_FFmpeg) than I think most people realize. The "tool" is
both a set of libraries and a set of command-line executables, there is no GUI packaged with it. So what follows are the
commands I used, which I typed into a terminal to execute.

While I could have exported every single frame from each video, the videos were long, on the order of hours. With video
content usually being at least 30 frames per second (so the eye doesn't experience any flicker), that would result in
a prohibitively large number of images to process for my purposes. So the first order of business was to subsample the
videos.

To output one image every second.
```terminal
ffmpeg -i input.mp4 -vf fps=1 out%04d.png
```

To output one image ever minute.
```terminal
ffmpeg -i input.mp4 -vf fps=1/60 out%04d.png
```

The `%04d` is [printf()
syntax](https://learn.microsoft.com/en-us/cpp/c-runtime-library/format-specification-syntax-printf-and-wprintf-functions?view=msvc-170)
that says to print out an integer number and prepend zeros where necessary to ensure that it is 4 digits long. So we
will end up with `out0001.png`, `out0002.png` and so on.

To start one minute into a given video and copy two minutes worth of content.
```terminal
ffmpeg -ss 00:01:00 -to 00:03:00 -i input.mp4 -c copy output.mp4
```

To specify a set of custom ranges use a select filter -- in the following case it will select between 2 and 6 seconds
and then between 15 to 24 seconds.
```terminal
ffmpeg -i in.mp4 -vf select='between(t,2,6)+between(t,15,24)' -vsync 0 out%04d.png
```

Further documentation on the ffmpeg command-line tool and its arguments can be found [here](https://ffmpeg.org/ffmpeg.html).

### Using COLMAP

Initially the program can be a bit confusing. I will briefly outline how I used it. If you want more detail than
I provide below, they have [a great tutorial](https://colmap.github.io/tutorial.html) and more on the [COLMAP docs
site](https://colmap.github.io/index.html).

We start by specifying a new project as follows.

![New Project](new_project.png)

The "Database" field is for a location in which to store a sql database of image info (like the descriptors mentioned
previously). The "Images" field is for a directory in which to find all the images for the scene -- which I created
using the above ffmpeg command to extract the images from a video.

![Automatic Reconstruction](auto_reconstruction.png)

Next we select "Automatic reconstruction" from the "Reconstruction" menu item, and up will pop the following dialog.

![Auto Reconstruction Dialog](auto_reconstruction_dialog2.png){: .right }

The fields here are seemingly a bit redundant with the project settings, though I'm sure there is a good reason for it.

- The "Workspace folder" is likely just the root folder in which you are storing everything.
- The "Images folder" is likely the same as the "Images" field that was specified above for the project previously -- it
    is just a directory in which to find all the images to use in the reconstruction.
- The "Vocabulary tree" is a way of specifying to match against nearest neighbors based on traversing the pre-trained
    tree. This can speed up matching and you can download trees from [here](https://demuc.de/colmap/).
- The "Shared intrinsics" being checked is because I am using images taken from one video (which uses the same camera
    throughout).
- I also do not care about generating a dense model since, for our particular use case, I'm more interested in the
    camera's pose that gets derived than the reconstructed 3d points.

Once all of that is specified and you click "Run", it should go ahead and automatically do feature detection and
extraction, followed by matching and geometric verification. This can all take quite a bit of time, even on a GPU. 

At the end of all the computation you will end up with something like the following.

![Auto Reconstruction Dialog](reconstruction_default.png)

The red pyramids above are called [viewing frustums](https://en.wikipedia.org/wiki/Viewing_frustum) and represent the
camera at that point in space. The flat rectangle portion is meant to represent the image that is being looked at. The
sparse set of 3d points that were detected will be scattered around the scene, and it is pretty intuitive the path the
camera/person took. The speed of the camera will be evident in the size of the gaps between frustums since they were
sampled from the video at a specific rate -- the larger the gap, the faster the camera is moving.

![Select Matches](select_image.png)

You can even double click on the individual points or the frustums and get information on what was matched and why (as
seen above).

All of this is great and very useful, but there were some quality of life improvements that were asked for after our
team played with the default program for a bit, which I will cover in the subsequent sections.

## Further Requirements and Features

While COLMAP as it exists is great, it was seemingly designed more for extracting the 3d points or the dense meshes
created from those points than for analyzing the camera pose in a scene, or its passage through time. To help categorize
and understand our date we needed some more functionality, notably:

- It was difficult to find specific images in all the data in the large scene space.
- Wanted to be able to see the path taken by the one camera as it was sometimes ambiguous.
- Wanted to be able to show certain sub-sections of the path taken to highlight various behaviors, and hide the rest of the path.
- Needed a better way to deal with wild jumps in the path and remove those outliers.

### Building COLMAP from Source

To add the new features required that I build COLMAP from source. On Windows this means installing [Visual
Studio](https://visualstudio.microsoft.com/). While 2022 is the latest version, I ended up using 2019 because I ran into
a compiler bug with v17.5.3 of VS 2022. There are newer versions of 2022 out, and they may work, but I have not yet
tested them.

You will also likely need to install `git`, which can be found [here](https://gitforwindows.org/).

I started by trying to build every dependency from scratch, and while this is probably technically possible, I managed
to fail somewhere around the SuiteSparse dependency. I think it was SuiteSparse's CHOLMOD dependency that really caused
me issues, but the details are a bit fuzzy at this point.

What I did have luck with was going the [VCPKG route](https://colmap.github.io/install.html#vcpkg) with Visual Studio
2019. I also tried building with Visual Studio 2022 (v17.5.3) but ran into a compiler bug -- presumably the same
one mentioned in the COLMAP docs.

```terminal
git clone https://github.com/microsoft/vcpkg
cd vcpkg
bootstrap-vcpkg.bat -disableMetrics
vcpkg install colmap:x64-windows --editable
```

It may also require installing FLANN, though my notes are unclear on this.

```terminal
vcpkg install FLANN:x64-windows
```

Once everything is set up, you can change directory to the colmap directory that was checked out by vcpkg, or you can
check out your own version by running `git clone https://github.com/colmap/colmap.git`.

The following will specify to build with [CUDA](https://developer.nvidia.com/cuda-toolkit) support and requires that it
be installed. I compiled with v11.6 in this instance, but something newer in the v12 range is probably also fine. It
also specifies that an Nvidia GPU with a Pascal architecture or newer be available (see [CUDA
gencodes](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)).

```terminal
cd colmap
mkdir build
cd build
cmake .. -G "Visual Studio 17 2022" -A x64 -T cuda=11.6 -DCMAKE_CUDA_ARCHITECTURES=60 -DCMAKE_TOOLCHAIN_FILE=C:/Dev/vcpkg/scripts/buildsystems/vcpkg.cmake
```

The command above will create a Visual Studio 2019 solution file (with a `.sln` extension) in that `colmap/build`
directory. Open it with Visual Studio and build the project. If you build a `Release` version of the application it will
end up in the `colmap\build\src\colmap\exe\Release` directory.


### Building My Modified COLMAP

I made a variety of changes, which I will outline in subsequent sections, that have not been merged into the main COLMAP
repo. These changes can be retrieved from [my forked version](https://github.com/samr/colmap/tree/viz-ui-changes-2) of
the COLMAP repo. The steps to use my repo instead of the main one are as follows.

```terminal
git clone https://github.com/samr/colmap
cd colmap
git checkout viz-ui-changes-2
```

Then build as outlined above in [Building COLMAP from Source](/posts/mapping-virtual-paths/#building-colmap-from-source).

I fixed [a blocking bug](https://github.com/colmap/colmap/issues/1964) in my version of the repo that was preventing me
from compiling. This bug may or may not be fixed in the main repo at time of this reading. If you get an error that
starts with `error nvcc fatal : A single input file is required for...` then the bug is still present. My specific fix
for the bug can be found [here](https://github.com/colmap/colmap/compare/main...samr:colmap:win-compile-fixes), but it
is also rolled into the above `viz-ui-changes-2` branch, so you shouldn't encounter it if you checked out that branch
and built it as outlined above. Unfortunately it is a bit of a hack and probably shouldn't be upstreamed as-is into the
main repo.

### My Modifications

The code diff that shows all my modifications can be found
[here](https://github.com/colmap/colmap/compare/main...samr:colmap:viz-ui-changes-2).

#### Link consecutive frames

The first change I made was to create a line linking consecutive images taken from the video. This shows the path that
camera took and is useful for identifying images that did not get matched properly.

![Reconstruction with travel line](with_line.png)
_The light blue line links consecutive frames from the video_

The order of the consecutive frames is determined from sorting the filenames lexicographically. So it is important that
the filenames with numbers in the name have prepended zeros (e.g. `image0024.png`).

#### Color a range of frames

The next change I made was to provide filtering based on a range of images using those image numbers. If you select
"Render" from the file menu, then go to "Render options", a dialog will pop up as follows.

![Render options and filters](image_ranges_and_filters.png)

In the dialog there is a "Image Colormap" entry in which you can select "Images filename in number range". Then you can
specify to "Add" a "Num Range to Color". Clicking add will present a couple of dialog boxes to specify a starting image
number and an image number at which to stop, then a color for the frustum's image plane and wire frame.

![Range of frames colored](range_colored.png)
_Images 766 to 866 were selected for coloring_

After clicking "Apply" it will color the range of images like is shown above. To clear the color and return it to the
default red, select the "Clear" button as shown below and click "Apply".

![Clearing the range of images](range_clear.png)

#### Hide a range of frames

You can see that there is another option which is "Num Range to Hide", and it functions in a similar way where you can
specify a range of image numbers present in the filename and then hides those frustums.

![Hiding a range of images](range_hidden.png)

#### Selecting a specific image by number

Sometimes you know the specific image number you would like to see and want to find it in the scene.

![Selecting a specific image by number](select_specific_image.png)
_Selecting image 850 in this dataset_

This can be accomplished by clicking "Select Image" in the "Render options" and then typing in the number. It will pop
up the image info dialog and highlight the frustum in the scene.

#### Filtering out outliers

I [made
a change](https://github.com/colmap/colmap/compare/main...samr:colmap:viz-ui-changes-2#diff-5c58dc05b92fa473674e34ed47895d99358e39024f844d65c2895663d5f84c42R1409)
that does not currently have a GUI toggle (though it probably should). I filter out large inconsistent spatial jumps in
the camera path. I make the simplifying assumption that given this is a video, most of the motion is relatively
consistent throughout the video.

For each image in the scene, I take a window of 6 previous image camera positions and calculate the mean position.
I then subtract the mean position from the current camera position to get a mean-adjusted position. I then calculate the
mean and standard deviation over all the mean-adjusted positions in the scene. I consider any values in the [upper
quartile](https://en.wikipedia.org/wiki/Quartile) of the distribution to be outliers and remove them from the path. This
removes image positions that exhibit large changes in velocity (i.e. acceleration).

I did this to combat the occasional bad pose that would show up in the path. I think this is likely a byproduct of
working with images of virtual spaces, which are themselves pixelated and can have varying levels of detail from frame
to frame -- leading to false matches.


## Conclusions

This tool was surprisingly helpful in solving our challenges around analyzing gaze and movement in our videos of VR
experiences. It was also noisy and prone to "hallucinations" at times, falsely connecting various parts of the space due
to bad matches. So definitely do not blindly trust the results -- visually verify your results.

My goal with this post was to lay out my journey in solving this problem and making the process clear for others. I hope
the tool and the modifications are useful.

You might argue that I should get the changes into the main COLMAP tool, but I added some big UI changes that are really
only useful for a particular use-case -- that of a single video, where the derived images have a specific file naming
convention. I think it would require a bit more thought on how to best integrate it before putting forth a pull request
and merging it into the main COLMAP tool.

Instead of working directly in COLMAP, there are other tools that understand the results that COLMAP generates, and
perhaps they already have some of the above functionality. However, I have not had time to experiment with them. The one
that immediately comes to mind is [rerun.io](https://www.rerun.io/examples/real-data/structure-from-motion). Perhaps
that will be a future post.
