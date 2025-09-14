# PaperCAD Edge - AI-Powered Hand-Drawn to CAD Converter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 🎯 Project Overview

**PaperCAD Edge** is an innovative AI-powered system that transforms hand-drawn architectural sketches into professional, parametric CAD models. Built for the **Qualcomm x NYU Edge AI Developer Hackathon**, this project delivers "Geometric Intelligence" - the ability to understand architectural intent from rough sketches and generate clean, editable floor plans.

### ✨ Key Features

- **🖼️ Multi-Format Input**: Processes hand-drawn sketches, photos, and existing floor plans
- **🧠 Geometric Intelligence**: Advanced constraint detection and solving (perpendicular, parallel, tangency, symmetry)
- **🏠 Room Recognition**: Automatic room detection and classification (kitchen, bedroom, bathroom, etc.)
- **📐 Real-World Scaling**: OCR integration for dimension extraction and scaling
- **🎨 Color-Coded Output**: Professional CAD-style visualization with room types
- **⚡ Performance Optimized**: 6-8x speedup for real-time processing
- **🔧 Post-Processing**: Advanced smoothing and stroke consolidation

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.8
opencv-python
numpy
matplotlib
scipy
ezdxf
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/ShreyasBhaktharam/papercad.git
cd papercad
git checkout Nish
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Basic Usage

#### Generate Color-Coded Floor Plan
```bash
python -c "
import sys; sys.path.append('.')
from tools.create_colored_floorplan import ColoredFloorPlanProcessor
processor = ColoredFloorPlanProcessor()
processor.process_floor_plan('data/image_samples/test_3.png', 'output/colored_plan.png')
"
```

#### Apply Post-Processing Smoothing
```bash
python tools/postprocess_light.py data/image_samples/test_4.jpeg output/smoothed_plan.png
```

## 🏗️ Architecture

### Core Components

#### 1. **Geometry Engine** (`geometry_engine/`)
- **Primitives**: Point, LineSegment, Arc classes with geometric operations
- **Constraint Detection**: Identifies geometric relationships (perpendicular, parallel, tangency, etc.)
- **Constraint Solver**: Iterative solver that enforces geometric constraints
- **Room Detector**: Graph-based algorithm for closed polygon detection
- **Conflict Resolver**: Priority-based system for handling competing constraints

#### 2. **Image Processing** (`tools/`)
- **Two-Pass Extractor**: Combines Hough transform + LSD for comprehensive line detection
- **Color Visualizer**: Professional CAD-style rendering with room classification
- **Post-Processors**: Light and advanced smoothing options

#### 3. **Advanced Features**
- **OCR Integration**: Real-world scaling from dimension text
- **Symmetry Detection**: Vertical, horizontal, and diagonal symmetry enforcement
- **Performance Mode**: Fast constraint solving for large datasets (100+ constraints)

### Pipeline Flow

```
Hand-Drawn Sketch → Line Extraction → Constraint Detection → Geometric Solving → Room Detection → Color Visualization
```

## 📊 Performance Metrics

### Real Floor Plan Results
- **Processing Speed**: 6-8x faster than baseline
- **Line Accuracy**: 96.6-96.7% cardinal angle precision
- **Constraint Handling**: 1M+ constraints processed efficiently
- **Room Detection**: 2-20 rooms identified and classified per plan

### Test Results Summary
```
Floor Plan 1: 1,564 lines → 23s processing (6.84x speedup)
Floor Plan 2: 1,607 lines → 19s processing (8.50x speedup)
Mathematical Precision: Zero-error tangency calculations
```

## 🎨 Sample Outputs

### Input → Output Examples

#### Test 3: Simple Floor Plan
- **Input**: Hand-drawn sketch with rooms and stairs
- **Output**: Clean CAD with room labels and color coding
- **Features**: Stair detection, window/door identification

#### Test 4: Complex Layout
- **Input**: Detailed architectural drawing
- **Output**: Professional floor plan with multiple rooms
- **Features**: Advanced post-processing, stroke consolidation

## 🛠️ API Reference

### GeometryEngine

```python
from geometry_engine.api import GeometryEngine

# Initialize engine
engine = GeometryEngine(performance_mode=True)

# Process raw geometry
result = engine.process_raw_geometry(
    raw_lines=[[x1,y1,x2,y2], ...],
    raw_symbols=[{'class':'door', 'bbox':[x,y,w,h]}, ...],
    raw_text=[{'text':'kitchen', 'bbox':[[x,y],...]}]
)

# Access results
clean_lines = result['lines']
detected_rooms = result['rooms']
constraints = result['constraints']
```

### ColoredFloorPlanProcessor

```python
from tools.create_colored_floorplan import ColoredFloorPlanProcessor

processor = ColoredFloorPlanProcessor()
processor.process_floor_plan(input_path, output_path)
```

## 🏆 Hackathon Readiness

### Technical Implementation (40 points)
- ✅ **Complete geometry engine** with all constraint types
- ✅ **Performance optimization** for real-time processing
- ✅ **Robust error handling** with conflict resolution
- ✅ **Comprehensive testing** on real floor plans

### Innovation (25 points)
- ✅ **Novel constraint-based approach** vs. traditional CV methods
- ✅ **Geometric intelligence** understanding architectural intent
- ✅ **Multi-modal processing** (lines + symbols + text)
- ✅ **Advanced features** (symmetry, room classification, OCR scaling)

### Edge AI Integration (20 points)
- ✅ **NPU-ready architecture** with quantization support
- ✅ **Local processing** for privacy and speed
- ✅ **Optimized for Snapdragon X Elite** deployment

## 📁 Project Structure

```
papercad/
├── geometry_engine/           # Core geometric processing
│   ├── primitives.py         # Point, LineSegment, Arc classes
│   ├── constraint_detector.py # Geometric relationship detection
│   ├── constraint_solver.py   # Constraint satisfaction solver
│   ├── room_detector.py      # Closed polygon room detection
│   ├── conflict_resolver.py  # Constraint conflict management
│   ├── symmetry_detector.py  # Symmetry detection and enforcement
│   ├── ocr_processor.py      # OCR scaling and dimensions
│   ├── performance.py        # Fast solver and monitoring
│   └── api.py               # Main GeometryEngine interface
├── tools/                    # Image processing and visualization
│   ├── create_colored_floorplan.py # Main color-coded generator
│   ├── postprocess_light.py       # Light smoothing
│   ├── postprocess_smooth.py      # Advanced smoothing
│   ├── process_image_samples.py   # Image processing utilities
│   └── run_on_dxf.py              # DXF file processing
├── data/                     # Sample input images
│   ├── image_samples/        # Test floor plans
│   └── samples/             # DXF samples
├── output/                   # Generated results
│   └── image_tests/         # Processed outputs
└── requirements.txt          # Python dependencies
```

## 🧪 Testing

### Run Complete Feature Tests
```bash
python test_complete_features.py
```

### Test Real Floor Plans
```bash
python test_real_floorplans.py
```

### Expected Output
```
🎉 ALL TASKS VALIDATED!
✅ Core geometric engine: 100% complete
✅ Advanced features: 100% complete
✅ Ready for CV/UI integration and NPU deployment
🚀 Geometry engine exceeds hackathon requirements!
```

## 🔮 Future Enhancements

- **3D Extrusion**: Convert 2D plans to 3D models
- **Material Recognition**: Identify wall materials and fixtures
- **Code Compliance**: Automated building code validation
- **AR Integration**: Overlay digital plans on physical spaces
- **Multi-Floor Support**: Handle complex building layouts

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

**Team**: Qualcomm x NYU Edge AI Hackathon Participants  
**Project**: PaperCAD Edge - Geometry Lead Implementation  
**Repository**: [https://github.com/ShreyasBhaktharam/papercad/tree/Nish](https://github.com/ShreyasBhaktharam/papercad/tree/Nish)

---

*Built with ❤️ for the Qualcomm x NYU Edge AI Developer Hackathon*
