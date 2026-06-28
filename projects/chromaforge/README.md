# ChromaForge

A small computer vision tool that flattens a photo into blocks of flat colour
(like a posterized illustration), then splits each colour region into its
own transparent PNG layer. Layers can also be recombined back into a single
image.

## How it works

1. **Flatten** ([segment.py](segment.py)) - the source image is run through
   mean-shift filtering (`cv2.pyrMeanShiftFiltering`), which merges pixels
   that are close in both position and colour into spatially-coherent
   blobs. K-means then caps the result to a fixed number of colours,
   painting each blob with its cluster's average colour.
2. **Split** ([extract.py](extract.py)) - takes an already-flattened image
   and pulls each distinct colour out into its own image, with everything
   else made transparent (alpha = 0), so each colour becomes a standalone
   PNG layer.
3. **Merge** ([merge.py](merge.py)) - takes multiple PNG layers and stacks
   them back into one image. Where layers overlap, the later one in the
   list overwrites the earlier one (painter's algorithm).

## Usage

Run the GUI:

```
python main.py
```

The window lets you pick one of three modes, choose your input file(s),
and run the operation:

- **Flatten colours** - pick one image, tune the parameters below, and save
  the flattened result as a PNG.
  - *Blob size* (`sp`) - spatial window radius; how far away (in pixels) to
    look for similar neighbours when merging.
  - *Colour tolerance* (`sr`) - colour window radius; how different two
    colours can be and still count as "the same".
  - *How many colours?* (`k`) - caps the final result to this many distinct
    colours.
- **Split into layers** - pick a flattened image (i.e. output from the
  Flatten mode) and a destination folder; saves one transparent PNG per
  colour found in the image.
- **Merge layers** - pick multiple layer PNGs and save the combined result.

## Requirements

- Python 3.12
- [opencv-python](https://pypi.org/project/opencv-python/) (`cv2`)
- [numpy](https://numpy.org/)
- `tkinter` (ships with most standard Python installs)

```
pip install opencv-python numpy
```

## Project structure

```
segment.py    # colourSegment: photo -> flat-colour image + per-pixel labels
extract.py    # colourExtract: flat-colour image + labels -> list of PNG layers
merge.py      # mergeLayers: list of PNG layers -> single merged image
main.py       # Tkinter GUI tying the three operations together
test_imgs/    # sample images used while developing
```

## Status

Personal/learning project - parameters are tuned by hand per image rather
than auto-detected, and the GUI is intentionally minimal.
