// Global variables
let updateInterval;
let canvas, ctx;
let distributionCanvas, distributionCtx;
let analyticsCanvas, analyticsCtx;
let simulationData = {};
let familyTreeData = {};
let currentZoom = 1.0;
let currentTab = 'simulation';

// Initialize canvas and components
window.onload = function() {
    canvas = document.getElementById('simulationCanvas');
    ctx = canvas.getContext('2d');
    
    distributionCanvas = document.getElementById('distributionChart');
    distributionCtx = distributionCanvas.getContext('2d');
    
    analyticsCanvas = document.getElementById('analyticsChart');
    analyticsCtx = analyticsCanvas.getContext('2d');
    
    // Start updating data more frequently for better real-time feel
    updateInterval = setInterval(updateData, 200); // 5 times per second
    updateData(); // Initial update
    
    // Initialize family tree
    initializeFamilyTree();
};

// Tab Management
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Remove active class from all tab buttons
    document.querySelectorAll('.tab-button').forEach(button => {
        button.classList.remove('active');
    });
    
    // Show selected tab content
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    currentTab = tabName;
    
    // Update content based on tab
    if (tabName === 'teams') {
        updateTeamsDistribution();
    } else if (tabName === 'family-tree') {
        updateFamilyTree();
    } else if (tabName === 'analytics') {
        updateAnalytics();
    }
}

// API calls
async function startSimulation() {
    try {
        const response = await fetch('/api/start', { method: 'POST' });
        const data = await response.json();
        addLog(data.message, 'info');
    } catch (error) {
        addLog('Error starting simulation: ' + error.message, 'error');
    }
}

async function pauseSimulation() {
    try {
        const response = await fetch('/api/pause', { method: 'POST' });
        const data = await response.json();
        addLog(data.message, 'warning');
    } catch (error) {
        addLog('Error pausing simulation: ' + error.message, 'error');
    }
}

async function resumeSimulation() {
    try {
        const response = await fetch('/api/resume', { method: 'POST' });
        const data = await response.json();
        addLog(data.message, 'info');
    } catch (error) {
        addLog('Error resuming simulation: ' + error.message, 'error');
    }
}

async function stopSimulation() {
    try {
        const response = await fetch('/api/stop', { method: 'POST' });
        const data = await response.json();
        addLog(data.message, 'warning');
    } catch (error) {
        addLog('Error stopping simulation: ' + error.message, 'error');
    }
}

async function resetSimulation() {
    try {
        const response = await fetch('/api/reset', { method: 'POST' });
        const data = await response.json();
        addLog(data.message, 'info');
        // Reset family tree data
        familyTreeData = {};
        updateFamilyTree();
    } catch (error) {
        addLog('Error resetting simulation: ' + error.message, 'error');
    }
}

// Update data from server
async function updateData() {
    try {
        const response = await fetch('/api/state');
        const data = await response.json();
        simulationData = data;
        updateUI(data);
        
        // Update family tree data
        updateFamilyTreeData(data);
        
        // Update current tab if needed
        if (currentTab === 'teams') {
            updateTeamsDistribution();
        } else if (currentTab === 'family-tree') {
            updateFamilyTree();
        } else if (currentTab === 'analytics') {
            updateAnalytics();
        }
    } catch (error) {
        console.error('Error updating data:', error);
    }
}

// Update UI elements
function updateUI(data) {
    // Update status
    const statusElement = document.getElementById('simulationStatus');
    
    if (data.is_running && !data.is_paused) {
        statusElement.innerHTML = '<span class="status-indicator status-running"></span>Running';
    } else if (data.is_paused) {
        statusElement.innerHTML = '<span class="status-indicator status-paused"></span>Paused';
    } else {
        statusElement.innerHTML = '<span class="status-indicator status-stopped"></span>Stopped';
    }

    // Update basic stats
    document.getElementById('currentEpisode').textContent = data.episode || 0;
    document.getElementById('currentStep').textContent = data.step || 0;

    // Update population stats
    if (data.population) {
        document.getElementById('activeTeams').textContent = data.population.total_teams || 0;
        document.getElementById('totalAgents').textContent = data.population.total_agents || 0;
        document.getElementById('aliveAgents').textContent = data.population.alive_agents || 0;
        document.getElementById('generation').textContent = data.population.statistics?.generation || 0;
    }

    // Update evolution stats
    if (data.population && data.population.insights) {
        document.getElementById('growingTeams').textContent = data.population.insights.growing_teams || 0;
        document.getElementById('avgDiversity').textContent = (data.population.insights.average_diversity || 0).toFixed(3);
    }

    // Update performance stats
    if (data.performance && data.performance.simulation) {
        document.getElementById('episodesPerMin').textContent = (data.performance.simulation.episodes_per_minute || 0).toFixed(1);
    }
    
    if (data.performance && data.performance.population) {
        const survivalRate = (data.performance.population.average_survival_rate || 0) * 100;
        document.getElementById('avgSurvivalRate').textContent = survivalRate.toFixed(1) + '%';
    }

    // Update visualization
    drawSimulation(data);
}

// Enhanced Teams Display for Teams Tab
function updateTeamsDistribution() {
    if (!simulationData.population) return;
    
    const teams = simulationData.population.teams || {};
    updateTeamsGrid(teams);
    drawDistributionChart(teams);
}

// Update teams grid with enhanced cards
function updateTeamsGrid(teams) {
    const teamsGrid = document.getElementById('teamsGrid');
    teamsGrid.innerHTML = '';

    const teamArray = Object.values(teams);
    
    // Sort teams by performance
    teamArray.sort((a, b) => (b.survival_rate || 0) - (a.survival_rate || 0));

    teamArray.forEach((team, index) => {
        const teamCard = document.createElement('div');
        teamCard.className = 'team-card';
        teamCard.style.setProperty('--team-color', team.color || '#4ecdc4');
        
        const performancePercent = ((team.survival_rate || 0) * 100);
        const rank = index + 1;
        const aliveCount = team.alive_count || 0;
        const teamSize = team.size || 0;
        
        // Determine team status
        let statusClass = '';
        let statusText = 'Active';
        
        if (team.status === 'eliminated' || aliveCount === 0) {
            statusClass = 'eliminated';
            statusText = 'Eliminated';
            teamCard.classList.add('eliminated');
        } else if (aliveCount < 2) {
            statusClass = 'at-risk';
            statusText = 'At Risk';
        }
        
        teamCard.innerHTML = `
            <div class="team-header">
                <div class="team-info">
                    <div class="team-color" style="background-color: ${team.color || '#4ecdc4'}"></div>
                    <div class="team-name">Team ${team.id}</div>
                </div>
                <div class="team-generation">Gen ${team.generation || 0}</div>
            </div>
            <div class="team-status ${statusClass}">${statusText}</div>
            <div class="team-stats">
                <div class="team-stat">
                    <div class="team-stat-label">Size</div>
                    <div class="team-stat-value">${teamSize}</div>
                </div>
                <div class="team-stat">
                    <div class="team-stat-label">Alive</div>
                    <div class="team-stat-value">${aliveCount}</div>
                </div>
                <div class="team-stat">
                    <div class="team-stat-label">Rank</div>
                    <div class="team-stat-value">#${rank}</div>
                </div>
                <div class="team-stat">
                    <div class="team-stat-label">Diversity</div>
                    <div class="team-stat-value">${((team.diversity_score || 0) * 100).toFixed(1)}%</div>
                </div>
            </div>
            <div class="team-performance-bar">
                <div class="team-performance-fill" style="width: ${performancePercent}%"></div>
            </div>
            <div style="text-align: center; margin-top: 8px; font-size: 0.9em; opacity: 0.8;">
                Survival: ${performancePercent.toFixed(1)}%
                ${aliveCount < 2 && aliveCount > 0 ? ' ⚠️ Needs 2+ survivors' : ''}
            </div>
        `;
        
        teamsGrid.appendChild(teamCard);
    });
}

// Draw distribution chart
function drawDistributionChart(teams) {
    const canvas = distributionCanvas;
    const ctx = distributionCtx;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    const teamArray = Object.values(teams);
    if (teamArray.length === 0) {
        ctx.fillStyle = '#4ecdc4';
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No teams to display', canvas.width / 2, canvas.height / 2);
        return;
    }

    // Draw pie chart of team sizes
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(canvas.width, canvas.height) * 0.3;
    
    const totalAgents = teamArray.reduce((sum, team) => sum + (team.size || 0), 0);
    let currentAngle = 0;
    
    teamArray.forEach((team, index) => {
        const teamSize = team.size || 0;
        const sliceAngle = (teamSize / totalAgents) * 2 * Math.PI;
        
        // Draw slice
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
        ctx.closePath();
        ctx.fillStyle = team.color || `hsl(${index * 360 / teamArray.length}, 70%, 60%)`;
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw label
        const labelAngle = currentAngle + sliceAngle / 2;
        const labelX = centerX + Math.cos(labelAngle) * (radius * 0.7);
        const labelY = centerY + Math.sin(labelAngle) * (radius * 0.7);
        
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`T${team.id}`, labelX, labelY);
        ctx.fillText(`${teamSize}`, labelX, labelY + 15);
        
        currentAngle += sliceAngle;
    });
    
    // Draw legend
    let legendY = 20;
    teamArray.forEach((team, index) => {
        ctx.fillStyle = team.color || `hsl(${index * 360 / teamArray.length}, 70%, 60%)`;
        ctx.fillRect(10, legendY, 15, 15);
        
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.textAlign = 'left';
        const status = (team.alive_count || 0) < 2 && (team.alive_count || 0) > 0 ? ' (At Risk)' : 
                      (team.alive_count || 0) === 0 ? ' (Eliminated)' : '';
        ctx.fillText(`Team ${team.id} (${team.size || 0} agents)${status}`, 30, legendY + 12);
        
        legendY += 25;
    });
}

// Family Tree Functions
function initializeFamilyTree() {
    familyTreeData = {
        nodes: [],
        links: [],
        generations: {}
    };
}

function updateFamilyTreeData(data) {
    if (!data.population || !data.population.teams) return;
    
    const teams = data.population.teams;
    
    // Update family tree data structure
    Object.values(teams).forEach(team => {
        const nodeId = `team_${team.id}`;
        
        // Check if node already exists
        let existingNode = familyTreeData.nodes.find(n => n.id === nodeId);
        
        if (!existingNode) {
            // Create new node
            const newNode = {
                id: nodeId,
                teamId: team.id,
                generation: team.generation || 0,
                color: team.color || '#4ecdc4',
                size: team.size || 0,
                alive: team.alive_count || 0,
                survival_rate: team.survival_rate || 0,
                status: team.status || 'active',
                created_at: Date.now(),
                children: [],
                parent: null
            };
            
            familyTreeData.nodes.push(newNode);
            
            // Group by generation
            if (!familyTreeData.generations[newNode.generation]) {
                familyTreeData.generations[newNode.generation] = [];
            }
            familyTreeData.generations[newNode.generation].push(newNode);
        } else {
            // Update existing node
            existingNode.size = team.size || 0;
            existingNode.alive = team.alive_count || 0;
            existingNode.survival_rate = team.survival_rate || 0;
            existingNode.status = team.status || 'active';
        }
    });
}

function updateFamilyTree() {
    const svg = document.getElementById('familyTreeSvg');
    
    // Clear existing content
    svg.innerHTML = '';
    
    if (familyTreeData.nodes.length === 0) {
        // Show placeholder
        const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        text.setAttribute('x', '50%');
        text.setAttribute('y', '50%');
        text.setAttribute('text-anchor', 'middle');
        text.setAttribute('fill', '#4ecdc4');
        text.setAttribute('font-size', '18');
        text.textContent = 'Start simulation to see family tree';
        svg.appendChild(text);
        return;
    }
    
    drawFamilyTreeSVG();
}

function drawFamilyTreeSVG() {
    const svg = document.getElementById('familyTreeSvg');
    const svgRect = svg.getBoundingClientRect();
    const width = svgRect.width || 800;
    const height = Math.max(500, Object.keys(familyTreeData.generations).length * 120);
    
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    
    const generations = Object.keys(familyTreeData.generations).sort((a, b) => parseInt(a) - parseInt(b));
    const levelHeight = height / (generations.length + 1);
    
    // Draw nodes by generation
    generations.forEach((gen, genIndex) => {
        const nodes = familyTreeData.generations[gen];
        const nodeWidth = width / (nodes.length + 1);
        
        nodes.forEach((node, nodeIndex) => {
            const x = (nodeIndex + 1) * nodeWidth;
            const y = (genIndex + 1) * levelHeight;
            
            // Draw node circle
            const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
            circle.setAttribute('cx', x);
            circle.setAttribute('cy', y);
            circle.setAttribute('r', 20 + (node.size * 2)); // Size based on team size
            circle.setAttribute('fill', node.color);
            circle.setAttribute('class', 'node-circle');
            circle.setAttribute('data-team-id', node.teamId);
            
            // Add eliminated styling
            if (node.status === 'eliminated' || node.alive === 0) {
                circle.classList.add('eliminated');
            }
            
            // Add hover events
            circle.addEventListener('mouseenter', (e) => showNodeDetails(e, node));
            circle.addEventListener('mouseleave', hideNodeDetails);
            
            svg.appendChild(circle);
            
            // Draw node text
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            text.setAttribute('x', x);
            text.setAttribute('y', y);
            text.setAttribute('class', 'node-text');
            text.textContent = `T${node.teamId}`;
            
            svg.appendChild(text);
            
            // Draw generation label
            if (nodeIndex === 0) {
                const genLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                genLabel.setAttribute('x', 10);
                genLabel.setAttribute('y', y);
                genLabel.setAttribute('fill', 'rgba(255, 255, 255, 0.7)');
                genLabel.setAttribute('font-size', '14');
                genLabel.textContent = `Gen ${gen}`;
                svg.appendChild(genLabel);
            }
        });
    });
}

function showNodeDetails(event, node) {
    const details = document.getElementById('nodeDetails');
    details.style.display = 'block';
    details.style.left = event.pageX + 10 + 'px';
    details.style.top = event.pageY + 10 + 'px';
    
    const status = node.status === 'eliminated' || node.alive === 0 ? 'Eliminated' :
                   node.alive < 2 ? 'At Risk' : 'Active';
    
    details.innerHTML = `
        <strong>Team ${node.teamId}</strong><br>
        Status: ${status}<br>
        Generation: ${node.generation}<br>
        Size: ${node.size}<br>
        Alive: ${node.alive}<br>
        Survival: ${(node.survival_rate * 100).toFixed(1)}%
    `;
}

function hideNodeDetails() {
    document.getElementById('nodeDetails').style.display = 'none';
}

// Tree control functions
function zoomTree(delta) {
    currentZoom = Math.max(0.5, Math.min(2.0, currentZoom + delta));
    document.getElementById('zoomLevel').textContent = Math.round(currentZoom * 100) + '%';
    
    const svg = document.getElementById('familyTreeSvg');
    svg.style.transform = `scale(${currentZoom})`;
}

function resetTreeView() {
    currentZoom = 1.0;
    document.getElementById('zoomLevel').textContent = '100%';
    document.getElementById('familyTreeSvg').style.transform = 'scale(1)';
}

function expandAllNodes() {
    addLog('All nodes expanded', 'info');
}

function collapseAllNodes() {
    addLog('All nodes collapsed', 'info');
}

// Analytics functions
function updateAnalytics() {
    if (!simulationData.population) return;
    
    drawAnalyticsChart();
}

function drawAnalyticsChart() {
    const canvas = analyticsCanvas;
    const ctx = analyticsCtx;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Draw placeholder analytics
    ctx.fillStyle = '#4ecdc4';
    ctx.font = '20px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Advanced Analytics Coming Soon', canvas.width / 2, canvas.height / 2);
    
    ctx.font = '14px Arial';
    ctx.fillText('Population trends, survival rates, and evolution metrics', canvas.width / 2, canvas.height / 2 + 30);
}

// Draw simulation visualization
function drawSimulation(data) {
    ctx.fillStyle = '#1a1a2e';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    if (!data.environment || !data.environment.agent_positions) {
        // Draw placeholder
        ctx.fillStyle = '#4ecdc4';
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Start simulation to see agents', canvas.width / 2, canvas.height / 2);
        return;
    }

    // Draw agents
    const agents = data.environment.agent_positions;
    const worldWidth = 800;
    const worldHeight = 600;
    const scaleX = canvas.width / worldWidth;
    const scaleY = canvas.height / worldHeight;

    Object.values(agents).forEach(agent => {
        if (!agent.alive) return;

        const x = agent.position[0] * scaleX;
        const y = agent.position[1] * scaleY;
        
        // Get team color
        const teams = data.population?.teams || {};
        const team = teams[agent.team_id];
        const color = team?.color || '#4ecdc4';

        // Draw agent
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, 2 * Math.PI);
        ctx.fill();

        // Draw health bar
        const healthRatio = agent.health / 100;
        ctx.fillStyle = `rgba(255, ${255 * healthRatio}, ${255 * healthRatio}, 0.8)`;
        ctx.fillRect(x - 6, y - 10, 12 * healthRatio, 2);
    });

    // Draw team territories (simple visualization)
    const teamCounts = data.environment?.team_counts || {};
    let yOffset = 10;
    Object.entries(teamCounts).forEach(([teamId, count]) => {
        const team = (data.population?.teams || {})[teamId];
        if (team) {
            ctx.fillStyle = team.color || '#4ecdc4';
            ctx.font = '12px Arial';
            ctx.textAlign = 'left';
            ctx.fillText(`Team ${teamId}: ${count} agents`, 10, yOffset);
            yOffset += 15;
        }
    });
}

// Add log entry
function addLog(message, type = 'info') {
    const logPanel = document.getElementById('activityLog');
    const logEntry = document.createElement('div');
    logEntry.className = `log-entry log-${type}`;
    logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    
    logPanel.appendChild(logEntry);
    logPanel.scrollTop = logPanel.scrollHeight;

    // Keep only last 50 entries
    while (logPanel.children.length > 50) {
        logPanel.removeChild(logPanel.firstChild);
    }
}

// Configuration panel functions
function toggleConfigPanel() {
    const panel = document.getElementById('configPanel');
    if (panel.style.display === 'none') {
        panel.style.display = 'block';
        loadCurrentConfig();
    } else {
        panel.style.display = 'none';
    }
}

async function loadCurrentConfig() {
    try {
        const response = await fetch('/api/config');
        const config = await response.json();
        
        document.getElementById('episodeLength').value = config.EPISODE_LENGTH || 1000;
        document.getElementById('simulationSpeed').value = config.SIMULATION_SPEED || 0.001;
        document.getElementById('initialTeams').value = config.INITIAL_TEAMS || 5;
        document.getElementById('startingTeamSize').value = config.STARTING_TEAM_SIZE || 4;
        document.getElementById('maxTeamSize').value = config.MAX_TEAM_SIZE || 12;
        document.getElementById('mutationRate').value = config.MUTATION_RATE || 0.1;
    } catch (error) {
        addLog('Error loading configuration: ' + error.message, 'error');
    }
}

async function applyConfig() {
    try {
        const config = {
            EPISODE_LENGTH: parseInt(document.getElementById('episodeLength').value),
            SIMULATION_SPEED: parseFloat(document.getElementById('simulationSpeed').value),
            INITIAL_TEAMS: parseInt(document.getElementById('initialTeams').value),
            STARTING_TEAM_SIZE: parseInt(document.getElementById('startingTeamSize').value),
            MAX_TEAM_SIZE: parseInt(document.getElementById('maxTeamSize').value),
            MUTATION_RATE: parseFloat(document.getElementById('mutationRate').value)
        };

        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });

        const result = await response.json();
        if (result.status === 'updated') {
            addLog('Configuration updated successfully', 'info');
            addLog(`Episode Length: ${config.EPISODE_LENGTH} steps`, 'info');
            addLog(`Simulation Speed: ${config.SIMULATION_SPEED}s delay`, 'info');
            addLog(`Teams: ${config.INITIAL_TEAMS}, Size: ${config.STARTING_TEAM_SIZE}`, 'info');
        } else {
            addLog('Error updating configuration: ' + result.message, 'error');
        }
    } catch (error) {
        addLog('Error applying configuration: ' + error.message, 'error');
    }
}

function resetConfig() {
    document.getElementById('episodeLength').value = 1000;
    document.getElementById('simulationSpeed').value = 0.001;
    document.getElementById('initialTeams').value = 5;
    document.getElementById('startingTeamSize').value = 4;
    document.getElementById('maxTeamSize').value = 12;
    document.getElementById('mutationRate').value = 0.1;
    
    addLog('Configuration reset to defaults', 'info');
}
