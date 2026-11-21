# Submission text

## Project title

GPU Batch Image Enhancement & Edge Detection with CUDA Streams

## Proof of execution

`proof_of_execution.zip` contains the same files as the `results/` folder:

* `logs/01_pipeline_all_images.log`: one execution that processes all 141
  SIPI images (80.7 MP). It has one line per image with size, channels,
  stream and GPU time, followed by a summary and a CPU-vs-GPU check
  (20/20 bit-exact).
* `logs/02_*_all_images.log`: the equalize, blur and Sobel operations, each
  run on all 141 images.
* `logs/03_pipeline_threshold60.log`: binary edge maps for all 141 images.
* `logs/04_stream_scaling.log`: throughput with 1, 2, 4 and 8 CUDA streams.
* `logs/pipeline_timings.csv`: per-image GPU timings.
* `logs/00_nvidia_smi.log`: the GPU the runs used.
* `previews/*_before_after.png`: input | equalized | blurred | Sobel edges
  for 14 images from all three SIPI volumes.
* `previews_binary_edges/*_before_after.png`: input | thresholded edge map.

## Code project description

I built a CUDA C++ command-line tool that batch-processes a whole directory
of images on the GPU. The test data is all three volumes of the USC-SIPI
database (textures, misc and aerials): 141 images, from 256x256 to
2250x2250, grayscale and color, 80.7 megapixels in total. Each image goes
through a four-stage pipeline: RGB to grayscale, histogram equalization,
Gaussian blur and Sobel edge detection. The tool can also run any single
stage on its own. Command-line flags select the operation, Gaussian sigma,
edge threshold, number of CUDA streams, number of images, CPU verification,
saving of intermediate stages, benchmark mode and a CSV timing report.

I wrote all the kernels myself instead of calling NPP, because I wanted to
practise the main CUDA patterns:

* The histogram uses per-block shared-memory histograms with shared atomics,
  merged into global memory once per block.
* The equalization lookup table comes from a Hillis-Steele parallel prefix
  scan over the 256 bins.
* The Gaussian blur is separable. The weights sit in `__constant__` memory,
  and each pass stages a 16x16 tile plus its apron in shared memory.
* The Sobel kernel works on an 18x18 shared-memory tile.

On the host, a BatchProcessor keeps a pool of slots. Each slot has its own
CUDA stream, pinned host buffers, device buffers and timing events. Images
are assigned to slots round-robin, so the host reads the next image from
disk while earlier images are still being copied and processed on other
streams.

To show that the GPU results are correct, I wrote a single-threaded CPU
reference and a --verify flag that compares the two outputs pixel by pixel.
This was the hardest part. nvcc fuses multiply-add into FMA instructions by
default and g++ does not, so a straightforward blur would differ by ±1 on
some pixels, and the Sobel stage would magnify that. I used the `__fmul_rn`
and `__fadd_rn` intrinsics in the blur kernels, `-ffp-contract=off` for the
host code, and integer arithmetic for luma and the equalization LUT. With
those changes every verified image matches bit for bit
(differing_pixels=0), which makes exact equality a strict test that catches
any indexing or apron bug.

Results on my RTX 2050 laptop GPU (CUDA 12.6, WSL2):

* All 141 images processed in one execution, 0 failures.
* The full pipeline averages 0.40 ms per 512x512 image and 1.09 ms per
  1024x1024 image, including both PCIe transfers.
* The GPU pipeline is about 20x faster than the CPU reference, up to 32x
  on 1024x1024 images.

The more interesting lesson was that the whole program is I/O-bound. Reading
PGMs and writing three outputs per image takes several times longer than the
GPU work, so end-to-end throughput is about 110-390 MP/s. With the images
preloaded into RAM (--benchmark) the GPU path reaches about 650 MP/s. Stream
scaling was also not linear. Four streams were best (645 MP/s against 556
MP/s with one stream), and eight streams were slower (407 MP/s). My reading
is that the extra per-stream buffer allocations and the WDDM driver's
batching of submissions on a small 16-SM laptop GPU eat up the benefit.
Next I would add multi-threaded host I/O so the GPU is no longer starved,
and try NPP's built-in filters for comparison.

The repository includes a Makefile, a run.sh that rebuilds every artifact,
the data download/conversion script, a .clang-format set to Google style, and
a README with build and run instructions.
