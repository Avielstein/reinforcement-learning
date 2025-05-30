# Dot Follow RL - Web Interface

A hybrid Python-Web interface that leverages the existing Python RL utilities while providing an interactive web-based visualization and control system.

## Architecture

### Python Backend (Flask Server)
- **All AI Logic**: Uses existing `DotFollowEnv`, `DotFollowLearner`, and utility classes
- **Model Management**: Load/save trained PyTorch models
- **Real-time Simulation**: Runs environment simulation in background thread
- **Parameter Control**: Adjust environment and model parameters on-the-fly
- **API Endpoints**: RESTful API for web interface communication

### Web Frontend (HTML/JavaScript)
- **Pure Visualization**: Only handles display and user interface
- **Real-time Updates**: Polls Python backend for simulation state
- **Interactive Controls**: Buttons and sliders for parameter adjustment
- **Live Charts**: Performance metrics visualization
- **Model Upload**: File upload interface for trained models

## Quick Start

### 1. Train a Model (Python)
```bash
cd tank-sim/dot-follow
python simple_demo.py  # Creates best_dot_follow_circular.pt
```

### 2. Start Web Interface
```bash
python start_web_interface.py
```

### 3. Open Browser
Navigate to: `http://localhost:5001`

### 4. Load Model
- Click "Choose File" and select your `.pt` model file
- Click "Load Model"
- Start the simulation and adjust parameters

## Features

### 🧠 Model Management
- **Load Trained Models**: Upload `.pt` files from Python training
- **Real-time Inference**: Uses loaded PyTorch models for fish behavior
- **Parameter Tuning**: Adjust exploration noise and action scaling
- **Model Status**: Visual indicator of loaded model state

### 🎮 Simulation Control
- **Start/Pause/Reset**: Full simulation control
- **Real-time Updates**: 20 FPS simulation, 10 FPS visualization
- **Episode Management**: Automatic episode reset and tracking

### 🎯 Movement Patterns
- **5 Target Patterns**: Circular, Figure-8, Random Walk, Zigzag, Spiral
- **Live Pattern Switching**: Change patterns during simulation
- **Pattern Visualization**: See target movement trails

### ⚙️ Environment Parameters
- **Target Speed**: Adjust how fast the target moves (2-20)
- **Pattern Size**: Control the radius/scale of movement patterns (10-40)
- **Water Current**: Adjust current strength affecting fish movement (0-8)

### 🧪 Model Parameters
- **Exploration Noise**: Add randomness to model actions (0-1)
- **Action Scaling**: Scale model output actions (0.1-2)
- **Real-time Adjustment**: Changes take effect immediately

### 📊 Live Metrics
- **Performance Charts**: Real-time reward and distance plots
- **Statistics Panel**: Episode count, steps, current metrics
- **Visual Feedback**: Fish trails, target paths, current indicators

## API Endpoints

### State Management
- `GET /api/state` - Get current simulation state
- `POST /api/control/<action>` - Control simulation (start/pause/reset)
- `POST /api/pattern/<pattern>` - Set movement pattern
- `POST /api/params` - Update parameters

### Model Management
- `POST /api/model/load` - Upload and load model file
- `GET /api/model/weights` - Export model weights as JSON

## File Structure

```
dot-follow/
├── web_server.py              # Flask backend server
├── start_web_interface.py     # Startup script
├── templates/
│   └── index.html             # Web interface template
├── uploads/                   # Temporary model upload directory
├── best_dot_follow_*.pt       # Trained model files
└── [existing Python files]   # All existing RL utilities
```

## Usage Examples

### Training and Testing Workflow

1. **Train Model in Python**:
```bash
python simple_demo.py
# Creates: best_dot_follow_circular.pt
```

2. **Start Web Interface**:
```bash
python start_web_interface.py
```

3. **Load and Test Model**:
- Upload `best_dot_follow_circular.pt` via web interface
- Start simulation
- Try different movement patterns
- Adjust parameters to see behavior changes

### Parameter Experimentation

1. **Load a trained model**
2. **Start with circular pattern**
3. **Gradually increase target speed** - see how fish adapts
4. **Switch to random_walk pattern** - observe generalization
5. **Add exploration noise** - see effect on behavior smoothness
6. **Adjust action scaling** - control fish responsiveness

### Fine-tuning Workflow

1. **Load base model**
2. **Test on different patterns**
3. **Identify weak performance areas**
4. **Adjust environment parameters**
5. **Use insights to improve Python training**

## Technical Details

### Backend Architecture
- **Threading**: Simulation runs in background thread
- **State Management**: Centralized WebInterface class
- **Model Loading**: Direct PyTorch model loading and inference
- **Parameter Updates**: Real-time environment parameter adjustment

### Frontend Architecture
- **Polling**: 100ms intervals for state updates
- **Canvas Rendering**: HTML5 Canvas for simulation visualization
- **Chart Rendering**: Custom JavaScript for performance metrics
- **Responsive Design**: Works on desktop and tablet devices

### Communication Protocol
- **JSON API**: All communication via JSON REST endpoints
- **File Upload**: Multipart form data for model files
- **Real-time Updates**: Polling-based state synchronization

## Troubleshooting

### Common Issues

**Port 5000 in use**:
- Web interface uses port 5001 to avoid AirPlay conflicts
- Check firewall settings if accessing remotely

**Model loading fails**:
- Ensure model was trained with compatible Python environment
- Check that model file is a valid PyTorch `.pt` file
- Verify model architecture matches expected input/output dimensions

**Simulation not updating**:
- Check browser console for JavaScript errors
- Verify Flask server is running and accessible
- Ensure all Python dependencies are installed

**Poor performance**:
- Reduce simulation speed if visualization is laggy
- Check system resources (CPU/memory usage)
- Try different browser if rendering is slow

### Dependencies

**Python Backend**:
```bash
pip install flask torch numpy matplotlib
```

**Web Frontend**:
- Modern browser with HTML5 Canvas support
- JavaScript enabled
- No additional dependencies required

## Advantages of This Architecture

### 🐍 Python Strengths
- **Reuse Existing Code**: Leverages all existing RL utilities
- **Model Compatibility**: Direct PyTorch model loading
- **Performance**: Fast simulation in native Python
- **Extensibility**: Easy to add new features using existing codebase

### 🌐 Web Strengths
- **Interactive Interface**: Rich UI with real-time controls
- **Cross-platform**: Works on any device with a browser
- **Visual Feedback**: Immediate response to parameter changes
- **User-friendly**: Intuitive interface for non-programmers

### 🔄 Hybrid Benefits
- **Best of Both Worlds**: Python performance + Web usability
- **Rapid Prototyping**: Quick parameter testing and model evaluation
- **Educational Tool**: Great for demonstrations and learning
- **Research Platform**: Ideal for hypothesis testing and exploration

This architecture allows you to train sophisticated models in Python and then interactively explore their behavior through an intuitive web interface, making it perfect for both research and educational applications.
