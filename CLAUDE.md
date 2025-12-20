# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements **lubrication theory simulations for droplet coalescence with insoluble surfactants** using the [pyoomph](https://pyoomph.github.io/) finite element framework. The physics models thin liquid films on substrates where two droplets merge, optionally with surfactant transport affecting surface tension.

## Running Simulations

### Main Coalescence with Surfactants
```bash
python coalescence.py --beta 0.1 --Pe 1.0 --Gamma0 0.8 --theta 20 --hp 1e-4
```
Key parameters:
- `--beta`: Surfactant strength (fractional surface tension reduction)
- `--Pe`: Péclet number (advection/diffusion ratio)
- `--Gamma0`: Initial surfactant concentration on left droplet
- `--theta`: Contact angle in degrees
- `--hp`: Precursor film thickness

### Clean Coalescence (No Surfactant)
```bash
python coalescence_clean.py
```

### Droplet Spreading (Axisymmetric)
```bash
python spreading_clean.py
```

## Postprocessing

```bash
# Generate analysis plots for surfactant simulation
python postprocess.py coalescence

# Generate plots for clean simulation
python postprocess_clean.py coalescence_clean

# Generate plots for spreading simulation
python postprocess_clean.py spreading_clean

# Generate video frames
python make_video-interfaceOnly.py coalescence
# Then create video:
ffmpeg -framerate 30 -i coalescence_frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p coalescence_video.mp4
```

## Architecture

### Equation Modules
- **`lubrication.py`**: Lubrication equations with surfactant transport (fields: h, p, Γ). Implements Marangoni stresses via β parameter.
- **`lubrication_clean.py`**: Clean lubrication equations without surfactant (fields: h, p). Supports optional disjoining pressure.

### Problem Classes
- **`coalescence.py`** → `DropletCoalescence`: Two spherical-cap droplets with surfactant initially on left droplet only
- **`coalescence_clean.py`** → `DropletCoalescence`: Same geometry, no surfactant
- **`spreading_clean.py`** → `DropletSpreading`: Single droplet spreading with disjoining pressure (axisymmetric)

### Physics Details
The lubrication equations (from paper §2.1-2.2):
- Mass conservation: ∂h/∂t + ∇·(h³/3·∇p + βh²/2·∇Γ) = 0
- Pressure-curvature: p = (1-βΓ)∇²h
- Surfactant transport: ∂Γ/∂t + ∇·(h²Γ/2·∇p + βhΓ∇Γ + 1/Pe·∇Γ) = 0

Initial geometry: two touching spherical caps with centers at x = ±√(2RH-H²), where R is the sphere radius derived from contact angle θ.

## Output Structure

Simulations create output directories named after the script:
```
coalescence/
├── domain/
│   ├── domain_000000.txt  # Text output: x, h, p, Gamma
│   ├── domain_000001.txt
│   └── ...
└── _ccode/                # Generated C code (pyoomph internals)
```

The text files contain columns: `coordinate_x`, `h`, `p`, `Gamma` (or just h, p for clean cases), with a header line containing `@time=<value>`.

## Dependencies

- pyoomph (finite element framework)
- numpy, matplotlib (postprocessing)
- ffmpeg (video creation, optional)
