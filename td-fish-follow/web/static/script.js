// TD Fish Follow - JavaScript Interface

class TDFishInterface {
    constructor() {
        this.isTraining = false;
        this.updateInterval = null;
        this.rewardChart = null;
        this.distanceChart = null;
        this.fishTrail = [];
        this.maxTrailLength = 20;
        
        this.initializeElements();
        this.setupEventListeners();
        this.initializeCharts();
        this.startUpdateLoop();
    }
    
    initializeElements() {
        // Get DOM elements
        this.elements = {
            // Controls
            startBtn: document.getElementById('start-training'),
            stopBtn: document.getElementById('stop-training'),
            resetBtn: document.getElementById('reset'),
            stepBtn: document.getElementById('manual-step'),
            patternSelect: document.getElementById('pattern'),
            tdMethodSelect: document.getElementById('td-method'),
            status: document.getElementById('status'),
            
            // Tank elements
            tank: document.getElementById('tank'),
            fish: document.getElementById('fish'),
            target: document.getElementById('target'),
            
            // Stats
            episode: document.getElementById('episode'),
            step: document.getElementById('step'),
            distance: document.getElementById('distance'),
            reward: document.getElementById('reward'),
            tdError: document.getElementById('td-error'),
            action: document.getElementById('action'),
            avgReward: document.getElementById('avg-reward'),
            avgDistance: document.getElementById('avg-distance'),
            bestDistance: document.getElementById('best-distance'),
            learningRate: document.getElementById('learning-rate'),
            
            // Charts
            rewardChart: document.getElementById('reward-chart'),
            distanceChart: document.getElementById('distance-chart')
        };
    }
    
    setupEventListeners() {
        this.elements.startBtn.addEventListener('click', () => this.startTraining());
        this.elements.stopBtn.addEventListener('click', () => this.stopTraining());
        this.elements.resetBtn.addEventListener('click', () => this.resetSimulation());
        this.elements.stepBtn.addEventListener('click', () => this.manualStep());
    }
    
    async startTraining() {
        const pattern = this.elements.patternSelect.value;
        const tdMethod = this.elements.tdMethodSelect.value;
        
        try {
            const response = await fetch('/api/start_training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    pattern: pattern,
                    td_method: tdMethod
                })
            });
            
            const result = await response.json();
            
            if (result.status === 'started') {
                this.isTraining = true;
                this.updateControlsState();
                this.updateStatus(`Training ${result.method.toUpperCase()} on ${result.pattern} pattern`, 'training');
                this.clearCharts();
            }
        } catch (error) {
            console.error('Error starting training:', error);
            this.updateStatus('Error starting training', 'stopped');
        }
    }
    
    async stopTraining() {
        try {
            const response = await fetch('/api/stop_training', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.status === 'stopped') {
                this.isTraining = false;
                this.updateControlsState();
                this.updateStatus('Training stopped', 'stopped');
            }
        } catch (error) {
            console.error('Error stopping training:', error);
        }
    }
    
    async resetSimulation() {
        try {
            const response = await fetch('/api/reset', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.status === 'reset') {
                this.clearTrail();
                this.updateStatus('Simulation reset', 'stopped');
            }
        } catch (error) {
            console.error('Error resetting simulation:', error);
        }
    }
    
    async manualStep() {
        try {
            const response = await fetch('/api/step', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (!result.error) {
                this.updateVisualization(result);
            }
        } catch (error) {
            console.error('Error taking manual step:', error);
        }
    }
    
    updateControlsState() {
        this.elements.startBtn.disabled = this.isTraining;
        this.elements.stopBtn.disabled = !this.isTraining;
        this.elements.patternSelect.disabled = this.isTraining;
        this.elements.tdMethodSelect.disabled = this.isTraining;
    }
    
    updateStatus(message, type) {
        this.elements.status.textContent = message;
        this.elements.status.className = `status ${type}`;
    }
    
    startUpdateLoop() {
        this.updateInterval = setInterval(() => {
            this.updateCurrentState();
            this.updateTrainingStats();
        }, 100); // Update every 100ms
    }
    
    async updateCurrentState() {
        try {
            const response = await fetch('/api/current_state');
            const state = await response.json();
            
            this.updateVisualization(state);
            this.updateStats(state);
        } catch (error) {
            console.error('Error updating current state:', error);
        }
    }
    
    async updateTrainingStats() {
        if (!this.isTraining) return;
        
        try {
            const response = await fetch('/api/training_stats');
            const data = await response.json();
            
            this.updateProgressStats(data);
            this.updateCharts(data);
        } catch (error) {
            console.error('Error updating training stats:', error);
        }
    }
    
    updateVisualization(state) {
        if (!state.fish_position || !state.target_position) return;
        
        const tankRect = this.elements.tank.getBoundingClientRect();
        const tankWidth = state.tank_width || 800;
        const tankHeight = state.tank_height || 600;
        
        // Scale positions to tank size
        const scaleX = this.elements.tank.offsetWidth / tankWidth;
        const scaleY = this.elements.tank.offsetHeight / tankHeight;
        
        // Update fish position
        const fishX = state.fish_position[0] * scaleX - 8; // Center the fish
        const fishY = state.fish_position[1] * scaleY - 8;
        
        this.elements.fish.style.left = `${fishX}px`;
        this.elements.fish.style.top = `${fishY}px`;
        
        // Update target position
        const targetX = state.target_position[0] * scaleX - 6; // Center the target
        const targetY = state.target_position[1] * scaleY - 6;
        
        this.elements.target.style.left = `${targetX}px`;
        this.elements.target.style.top = `${targetY}px`;
        
        // Add fish trail
        this.addTrailPoint(fishX + 8, fishY + 8); // Center of fish
        
        // Rotate fish based on movement direction
        if (state.action && state.action.length >= 2) {
            const angle = Math.atan2(state.action[1], state.action[0]) * 180 / Math.PI;
            this.elements.fish.style.transform = `rotate(${angle}deg)`;
        }
    }
    
    updateStats(state) {
        this.elements.step.textContent = state.step_count || 0;
        this.elements.distance.textContent = (state.distance_to_target || 0).toFixed(1);
        this.elements.reward.textContent = (state.episode_reward || 0).toFixed(2);
        this.elements.tdError.textContent = (state.td_error || 0).toFixed(4);
        
        if (state.action) {
            this.elements.action.textContent = `[${state.action[0].toFixed(2)}, ${state.action[1].toFixed(2)}]`;
        }
    }
    
    updateProgressStats(data) {
        this.elements.episode.textContent = data.episode || 0;
        
        if (data.stats && data.stats.length > 0) {
            const recentStats = data.stats.slice(-10);
            const avgReward = recentStats.reduce((sum, s) => sum + s.reward, 0) / recentStats.length;
            const avgDistance = recentStats.reduce((sum, s) => sum + s.avg_distance, 0) / recentStats.length;
            const bestDistance = Math.min(...data.stats.map(s => s.avg_distance));
            
            this.elements.avgReward.textContent = avgReward.toFixed(2);
            this.elements.avgDistance.textContent = avgDistance.toFixed(1);
            this.elements.bestDistance.textContent = bestDistance.toFixed(1);
        }
    }
    
    addTrailPoint(x, y) {
        // Add new trail point
        this.fishTrail.push({ x, y, time: Date.now() });
        
        // Remove old trail points
        if (this.fishTrail.length > this.maxTrailLength) {
            this.fishTrail.shift();
        }
        
        // Update trail visualization
        this.updateTrailVisualization();
    }
    
    updateTrailVisualization() {
        // Remove old trail elements
        const oldTrails = this.elements.tank.querySelectorAll('.trail');
        oldTrails.forEach(trail => trail.remove());
        
        // Add new trail elements
        this.fishTrail.forEach((point, index) => {
            const trail = document.createElement('div');
            trail.className = 'trail';
            trail.style.left = `${point.x - 2}px`;
            trail.style.top = `${point.y - 2}px`;
            trail.style.opacity = (index / this.fishTrail.length) * 0.6;
            this.elements.tank.appendChild(trail);
        });
    }
    
    clearTrail() {
        this.fishTrail = [];
        const trails = this.elements.tank.querySelectorAll('.trail');
        trails.forEach(trail => trail.remove());
    }
    
    initializeCharts() {
        // Initialize reward chart
        const rewardCtx = this.elements.rewardChart.getContext('2d');
        this.rewardChart = new SimpleChart(rewardCtx, {
            color: '#48dbfb',
            backgroundColor: 'rgba(72, 219, 251, 0.1)',
            maxPoints: 50
        });
        
        // Initialize distance chart
        const distanceCtx = this.elements.distanceChart.getContext('2d');
        this.distanceChart = new SimpleChart(distanceCtx, {
            color: '#feca57',
            backgroundColor: 'rgba(254, 202, 87, 0.1)',
            maxPoints: 50
        });
    }
    
    updateCharts(data) {
        if (data.stats && data.stats.length > 0) {
            // Update reward chart
            const rewardData = data.stats.map(s => s.reward);
            this.rewardChart.updateData(rewardData);
            
            // Update distance chart
            const distanceData = data.stats.map(s => s.avg_distance);
            this.distanceChart.updateData(distanceData);
        }
    }
    
    clearCharts() {
        this.rewardChart.clear();
        this.distanceChart.clear();
    }
}

// Simple chart implementation
class SimpleChart {
    constructor(ctx, options = {}) {
        this.ctx = ctx;
        this.canvas = ctx.canvas;
        this.data = [];
        this.options = {
            color: options.color || '#48dbfb',
            backgroundColor: options.backgroundColor || 'rgba(72, 219, 251, 0.1)',
            maxPoints: options.maxPoints || 50,
            padding: 20
        };
        
        this.resize();
    }
    
    resize() {
        const rect = this.canvas.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
    }
    
    updateData(newData) {
        this.data = newData.slice(-this.options.maxPoints);
        this.draw();
    }
    
    clear() {
        this.data = [];
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    draw() {
        if (this.data.length < 2) return;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        const width = this.canvas.width - 2 * this.options.padding;
        const height = this.canvas.height - 2 * this.options.padding;
        
        const minValue = Math.min(...this.data);
        const maxValue = Math.max(...this.data);
        const range = maxValue - minValue || 1;
        
        // Draw background area
        this.ctx.beginPath();
        this.ctx.moveTo(this.options.padding, this.canvas.height - this.options.padding);
        
        for (let i = 0; i < this.data.length; i++) {
            const x = this.options.padding + (i / (this.data.length - 1)) * width;
            const y = this.canvas.height - this.options.padding - 
                     ((this.data[i] - minValue) / range) * height;
            
            if (i === 0) {
                this.ctx.lineTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.lineTo(this.canvas.width - this.options.padding, this.canvas.height - this.options.padding);
        this.ctx.closePath();
        this.ctx.fillStyle = this.options.backgroundColor;
        this.ctx.fill();
        
        // Draw line
        this.ctx.beginPath();
        for (let i = 0; i < this.data.length; i++) {
            const x = this.options.padding + (i / (this.data.length - 1)) * width;
            const y = this.canvas.height - this.options.padding - 
                     ((this.data[i] - minValue) / range) * height;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
        }
        
        this.ctx.strokeStyle = this.options.color;
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Draw current value
        if (this.data.length > 0) {
            const lastValue = this.data[this.data.length - 1];
            this.ctx.fillStyle = this.options.color;
            this.ctx.font = '12px monospace';
            this.ctx.textAlign = 'right';
            this.ctx.fillText(lastValue.toFixed(1), this.canvas.width - 5, 15);
        }
    }
}

// Initialize the interface when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new TDFishInterface();
});

// Handle window resize
window.addEventListener('resize', () => {
    // Trigger chart resize if needed
    setTimeout(() => {
        const charts = document.querySelectorAll('canvas');
        charts.forEach(canvas => {
            const ctx = canvas.getContext('2d');
            if (ctx.chart) {
                ctx.chart.resize();
            }
        });
    }, 100);
});
