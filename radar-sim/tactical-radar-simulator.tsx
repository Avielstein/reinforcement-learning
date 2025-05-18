import React, { useState, useEffect, useRef } from 'react';

const TacticalRadarSimulator = () => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const [isPaused, setIsPaused] = useState(false);
  const [teams, setTeams] = useState({
    red: 3,
    blue: 3,
    green: 3
  });
  const [mapSize, setMapSize] = useState(1000); // meters
  const [radarRange, setRadarRange] = useState(250); // meters
  const [gameSpeed, setGameSpeed] = useState(1);
  
  const unitsRef = useRef([]);
  const projectilesRef = useRef([]);
  const explosionsRef = useRef([]);
  const lastUpdateTimeRef = useRef(Date.now());
  const battleLogRef = useRef([]);
  
  // Initialize game
  useEffect(() => {
    resetSimulation();
  }, [teams]);
  
  const resetSimulation = () => {
    unitsRef.current = [];
    projectilesRef.current = [];
    explosionsRef.current = [];
    battleLogRef.current = ["Simulation started"];
    
    // Create units for each team
    Object.entries(teams).forEach(([teamName, count]) => {
      if (count > 0) {
        for (let i = 0; i < count; i++) {
          const position = getTeamStartPosition(teamName, i);
          unitsRef.current.push({
            id: `${teamName}-${i}`,
            team: teamName,
            x: position.x,
            y: position.y,
            size: 8,
            heading: Math.random() * Math.PI * 2,
            speed: 40 + Math.random() * 20,
            radarRange: radarRange * (0.8 + Math.random() * 0.4),
            radarActive: true,
            health: 100,
            reload: 0,
            reloadTime: 2000,
            firingRange: 180,
            state: 'patrolling',
            detected: [],
            targetId: null,
            lastFired: 0
          });
        }
      }
    });
  };
  
  const getTeamStartPosition = (team, index) => {
    switch(team) {
      case 'red':
        return { x: mapSize * 0.2, y: mapSize * 0.2 + (index * 50) };
      case 'blue':
        return { x: mapSize * 0.8, y: mapSize * 0.8 - (index * 50) };
      case 'green':
        return { x: mapSize * 0.2, y: mapSize * 0.8 - (index * 50) };
      default:
        return { x: mapSize / 2, y: mapSize / 2 };
    }
  };
  
  // Main game loop
  useEffect(() => {
    if (!canvasRef.current) return;
    
    const updateGame = () => {
      if (!isPaused) {
        const currentTime = Date.now();
        const deltaTime = (currentTime - lastUpdateTimeRef.current) / 1000 * gameSpeed;
        lastUpdateTimeRef.current = currentTime;
        
        // Update unit states
        updateUnits(deltaTime, currentTime);
        
        // Update projectiles
        updateProjectiles(deltaTime, currentTime);
        
        // Update explosions
        updateExplosions(deltaTime);
      }
      
      renderGame();
      animationRef.current = requestAnimationFrame(updateGame);
    };
    
    const updateUnits = (deltaTime, currentTime) => {
      // Reset detection arrays
      unitsRef.current.forEach(unit => {
        unit.detected = [];
      });
      
      // Perform radar detection
      unitsRef.current.forEach(unit => {
        if (unit.radarActive) {
          unitsRef.current.forEach(target => {
            if (target.id !== unit.id && target.health > 0) {
              const distance = getDistance(unit, target);
              if (distance <= unit.radarRange) {
                unit.detected.push(target.id);
              }
            }
          });
        }
      });
      
      // Update unit behavior
      unitsRef.current.forEach(unit => {
        if (unit.health <= 0) return;
        
        // Process reload time
        if (unit.reload > 0) {
          unit.reload = Math.max(0, unit.reload - deltaTime * 1000);
        }
        
        // Decide state based on what unit detects
        if (unit.detected.length > 0 && unit.state !== 'evading') {
          // Find closest enemy
          let closestEnemy = null;
          let closestDistance = Infinity;
          
          unit.detected.forEach(detectedId => {
            const enemy = unitsRef.current.find(u => u.id === detectedId);
            if (enemy && enemy.team !== unit.team) {
              const distance = getDistance(unit, enemy);
              if (distance < closestDistance) {
                closestEnemy = enemy;
                closestDistance = distance;
              }
            }
          });
          
          if (closestEnemy) {
            unit.targetId = closestEnemy.id;
            if (closestDistance <= unit.firingRange) {
              unit.state = 'firing';
            } else {
              unit.state = 'pursuing';
            }
          }
        } else if (unit.health < 40 && Math.random() < 0.1) {
          // Low health, may decide to evade
          unit.state = 'evading';
          unit.targetId = null;
        } else if (!unit.targetId || Math.random() < 0.02) {
          // Random chance to return to patrolling
          unit.state = 'patrolling';
        }
        
        // Act based on current state
        switch (unit.state) {
          case 'patrolling':
            // Random movement
            if (Math.random() < 0.02) {
              unit.heading += (Math.random() - 0.5) * Math.PI / 2;
            }
            break;
            
          case 'pursuing':
            const target = unitsRef.current.find(u => u.id === unit.targetId);
            if (target) {
              // Turn toward target
              const targetAngle = Math.atan2(target.y - unit.y, target.x - unit.x);
              const angleDiff = normalizeAngle(targetAngle - unit.heading);
              unit.heading += angleDiff * Math.min(1, deltaTime * 2);
            }
            break;
            
          case 'evading':
            // Move away from all detected enemies
            let evadeAngle = unit.heading;
            if (unit.detected.length > 0) {
              let avgX = 0, avgY = 0;
              unit.detected.forEach(detectedId => {
                const enemy = unitsRef.current.find(u => u.id === detectedId);
                if (enemy && enemy.team !== unit.team) {
                  avgX += enemy.x;
                  avgY += enemy.y;
                }
              });
              
              avgX /= unit.detected.length;
              avgY /= unit.detected.length;
              
              // Move away from average enemy position
              const fleeAngle = Math.atan2(unit.y - avgY, unit.x - avgX);
              const turnSpeed = deltaTime * 3;
              const angleDiff = normalizeAngle(fleeAngle - unit.heading);
              unit.heading += angleDiff * Math.min(1, turnSpeed);
            }
            
            // Random evasive maneuvers
            if (Math.random() < 0.1) {
              unit.heading += (Math.random() - 0.5) * Math.PI / 4;
            }
            break;
            
          case 'firing':
            const enemy = unitsRef.current.find(u => u.id === unit.targetId);
            if (enemy && unit.reload === 0) {
              // Face target
              const targetAngle = Math.atan2(enemy.y - unit.y, enemy.x - unit.x);
              unit.heading = targetAngle;
              
              // Fire!
              fireProjectile(unit, enemy, currentTime);
              unit.reload = unit.reloadTime;
              unit.lastFired = currentTime;
            }
            break;
        }
        
        // Move unit
        const actualSpeed = unit.state === 'evading' ? unit.speed * 1.3 : unit.speed;
        unit.x += Math.cos(unit.heading) * actualSpeed * deltaTime;
        unit.y += Math.sin(unit.heading) * actualSpeed * deltaTime;
        
        // Keep units on map
        if (unit.x < 0) { unit.x = 0; unit.heading = Math.PI - unit.heading; }
        if (unit.x > mapSize) { unit.x = mapSize; unit.heading = Math.PI - unit.heading; }
        if (unit.y < 0) { unit.y = 0; unit.heading = -unit.heading; }
        if (unit.y > mapSize) { unit.y = mapSize; unit.heading = -unit.heading; }
      });
      
      // Check for teams eliminated
      const teamStatus = {};
      Object.keys(teams).forEach(team => {
        teamStatus[team] = 0;
      });
      
      unitsRef.current.forEach(unit => {
        if (unit.health > 0) {
          teamStatus[unit.team]++;
        }
      });
      
      Object.entries(teamStatus).forEach(([team, count]) => {
        if (count === 0 && teams[team] > 0) {
          battleLogRef.current.unshift(`Team ${team} has been eliminated!`);
        }
      });
    };
    
    const fireProjectile = (unit, target, currentTime) => {
      // Add slight inaccuracy
      const accuracy = 0.1;
      const inaccuracy = (Math.random() - 0.5) * accuracy;
      
      // Calculate lead for moving target
      const distance = getDistance(unit, target);
      const timeToImpact = distance / 300; // projectile speed is 300
      
      const predictedX = target.x + Math.cos(target.heading) * target.speed * timeToImpact;
      const predictedY = target.y + Math.sin(target.heading) * target.speed * timeToImpact;
      
      const angleToTarget = Math.atan2(predictedY - unit.y, predictedX - unit.x) + inaccuracy;
      
      projectilesRef.current.push({
        fromId: unit.id,
        fromTeam: unit.team,
        x: unit.x,
        y: unit.y,
        angle: angleToTarget,
        speed: 300,
        damage: 20 + Math.random() * 15,
        size: 3,
        timeCreated: currentTime
      });
    };
    
    const updateProjectiles = (deltaTime, currentTime) => {
      // Move projectiles
      projectilesRef.current.forEach(projectile => {
        projectile.x += Math.cos(projectile.angle) * projectile.speed * deltaTime;
        projectile.y += Math.sin(projectile.angle) * projectile.speed * deltaTime;
      });
      
      // Check for hits
      projectilesRef.current = projectilesRef.current.filter(projectile => {
        // Remove if too old (5 seconds)
        if (currentTime - projectile.timeCreated > 5000) {
          return false;
        }
        
        // Remove if outside map
        if (projectile.x < 0 || projectile.x > mapSize || 
            projectile.y < 0 || projectile.y > mapSize) {
          return false;
        }
        
        // Check for collision with units
        for (const unit of unitsRef.current) {
          if (unit.health <= 0) continue;
          if (unit.team === projectile.fromTeam) continue;
          
          const distance = getDistance(projectile, unit);
          if (distance < unit.size + projectile.size) {
            // Hit!
            unit.health -= projectile.damage;
            createExplosion(projectile.x, projectile.y, 1);
            
            if (unit.health <= 0) {
              createExplosion(unit.x, unit.y, 3);
              battleLogRef.current.unshift(`${projectile.fromTeam} destroyed ${unit.team} unit!`);
            }
            
            return false;
          }
        }
        
        return true;
      });
    };
    
    const updateExplosions = (deltaTime) => {
      // Update explosion lifetimes
      explosionsRef.current = explosionsRef.current.filter(explosion => {
        explosion.lifetime -= deltaTime;
        return explosion.lifetime > 0;
      });
    };
    
    const createExplosion = (x, y, size) => {
      explosionsRef.current.push({
        x,
        y,
        size: size * 5,
        lifetime: 0.5
      });
    };
    
    const renderGame = () => {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      const scale = width / mapSize;
      
      // Clear canvas
      ctx.fillStyle = '#111';
      ctx.fillRect(0, 0, width, height);
      
      // Draw grid
      ctx.strokeStyle = '#333';
      ctx.lineWidth = 1;
      
      for (let i = 0; i <= 10; i++) {
        const pos = (i / 10) * width;
        ctx.beginPath();
        ctx.moveTo(pos, 0);
        ctx.lineTo(pos, height);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, pos);
        ctx.lineTo(width, pos);
        ctx.stroke();
      }
      
      // Draw radar ranges for active units
      unitsRef.current.forEach(unit => {
        if (unit.health <= 0) return;
        if (unit.radarActive) {
          ctx.beginPath();
          ctx.arc(unit.x * scale, unit.y * scale, unit.radarRange * scale, 0, Math.PI * 2);
          ctx.strokeStyle = `${unit.team}`;
          ctx.globalAlpha = 0.15;
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      });
      
      // Draw projectiles
      projectilesRef.current.forEach(projectile => {
        ctx.beginPath();
        ctx.arc(projectile.x * scale, projectile.y * scale, projectile.size, 0, Math.PI * 2);
        ctx.fillStyle = getTeamColor(projectile.fromTeam);
        ctx.fill();
        
        // Draw trail
        ctx.beginPath();
        ctx.moveTo(projectile.x * scale, projectile.y * scale);
        ctx.lineTo(
          (projectile.x - Math.cos(projectile.angle) * 15) * scale,
          (projectile.y - Math.sin(projectile.angle) * 15) * scale
        );
        ctx.strokeStyle = getTeamColor(projectile.fromTeam);
        ctx.globalAlpha = 0.5;
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
      
      // Draw units
      unitsRef.current.forEach(unit => {
        if (unit.health <= 0) return;
        
        // Draw unit
        ctx.beginPath();
        ctx.arc(unit.x * scale, unit.y * scale, unit.size, 0, Math.PI * 2);
        ctx.fillStyle = getTeamColor(unit.team);
        ctx.fill();
        
        // Draw direction indicator
        ctx.beginPath();
        ctx.moveTo(unit.x * scale, unit.y * scale);
        ctx.lineTo(
          (unit.x + Math.cos(unit.heading) * 15) * scale,
          (unit.y + Math.sin(unit.heading) * 15) * scale
        );
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw detection lines
        unit.detected.forEach(detectedId => {
          const target = unitsRef.current.find(u => u.id === detectedId);
          if (target && target.team !== unit.team) {
            ctx.beginPath();
            ctx.moveTo(unit.x * scale, unit.y * scale);
            ctx.lineTo(target.x * scale, target.y * scale);
            ctx.strokeStyle = unit.state === 'firing' ? '#ff0' : '#0ff';
            ctx.globalAlpha = 0.3;
            ctx.lineWidth = 1;
            ctx.stroke();
            ctx.globalAlpha = 1;
          }
        });
        
        // Draw health bar
        const barWidth = 20;
        const barHeight = 3;
        const healthPercent = unit.health / 100;
        
        ctx.fillStyle = '#333';
        ctx.fillRect(
          (unit.x * scale) - barWidth/2,
          (unit.y * scale) + 10,
          barWidth,
          barHeight
        );
        
        ctx.fillStyle = healthPercent > 0.5 ? '#0f0' : healthPercent > 0.2 ? '#ff0' : '#f00';
        ctx.fillRect(
          (unit.x * scale) - barWidth/2,
          (unit.y * scale) + 10,
          barWidth * healthPercent,
          barHeight
        );
      });
      
      // Draw explosions
      explosionsRef.current.forEach(explosion => {
        const gradient = ctx.createRadialGradient(
          explosion.x * scale, explosion.y * scale, 0,
          explosion.x * scale, explosion.y * scale, explosion.size
        );
        
        gradient.addColorStop(0, 'rgba(255, 255, 0, 0.8)');
        gradient.addColorStop(0.5, 'rgba(255, 120, 0, 0.6)');
        gradient.addColorStop(1, 'rgba(255, 0, 0, 0)');
        
        ctx.beginPath();
        ctx.arc(explosion.x * scale, explosion.y * scale, explosion.size, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
      });
      
      // Draw battle log
      ctx.fillStyle = '#fff';
      ctx.font = '12px Arial';
      battleLogRef.current.slice(0, 5).forEach((log, i) => {
        ctx.fillText(log, 10, height - 10 - (i * 15));
      });
      
      // Draw team status
      const teamStatus = {};
      Object.keys(teams).forEach(team => {
        teamStatus[team] = { total: 0, alive: 0 };
      });
      
      unitsRef.current.forEach(unit => {
        teamStatus[unit.team].total += 1;
        if (unit.health > 0) {
          teamStatus[unit.team].alive += 1;
        }
      });
      
      let y = 20;
      Object.entries(teamStatus).forEach(([team, status]) => {
        ctx.fillStyle = getTeamColor(team);
        ctx.fillText(`${team}: ${status.alive}/${status.total}`, 10, y);
        y += 20;
      });
    };
    
    animationRef.current = requestAnimationFrame(updateGame);
    
    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isPaused, gameSpeed, mapSize, radarRange]);
  
  // Helper functions
  const getDistance = (obj1, obj2) => {
    const dx = obj1.x - obj2.x;
    const dy = obj1.y - obj2.y;
    return Math.sqrt(dx * dx + dy * dy);
  };
  
  const normalizeAngle = (angle) => {
    while (angle > Math.PI) angle -= Math.PI * 2;
    while (angle < -Math.PI) angle += Math.PI * 2;
    return angle;
  };
  
  const getTeamColor = (team) => {
    return team === 'red' ? '#f44' : 
           team === 'blue' ? '#48f' : 
           team === 'green' ? '#4d4' : '#dd2';
  };
  
  return (
    <div className="flex flex-col items-center bg-gray-800 p-4 rounded-lg shadow-lg w-full max-w-3xl">
      <h2 className="text-xl font-bold mb-4 text-gray-100">Tactical Radar Combat</h2>
      
      <div className="bg-black rounded shadow-inner mb-4">
        <canvas
          ref={canvasRef}
          width={600}
          height={600}
          className="rounded"
        />
      </div>
      
      <div className="w-full grid grid-cols-2 gap-4 mb-4">
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setIsPaused(!isPaused)}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 flex-1"
          >
            {isPaused ? 'Resume' : 'Pause'}
          </button>
          <button
            onClick={resetSimulation}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 flex-1"
          >
            Reset
          </button>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-1">
            Simulation Speed: {gameSpeed}x
          </label>
          <input
            type="range"
            min="0.5"
            max="3"
            step="0.5"
            value={gameSpeed}
            onChange={(e) => setGameSpeed(parseFloat(e.target.value))}
            className="w-full"
          />
        </div>
      </div>
      
      <div className="w-full grid grid-cols-3 gap-4">
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-1">
            Red Team: {teams.red}
          </label>
          <input
            type="range"
            min="0"
            max="5"
            step="1"
            value={teams.red}
            onChange={(e) => setTeams({...teams, red: parseInt(e.target.value)})}
            className="w-full"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-1">
            Blue Team: {teams.blue}
          </label>
          <input
            type="range"
            min="0"
            max="5"
            step="1"
            value={teams.blue}
            onChange={(e) => setTeams({...teams, blue: parseInt(e.target.value)})}
            className="w-full"
          />
        </div>
        
        <div>
          <label className="block text-sm font-medium text-gray-200 mb-1">
            Green Team: {teams.green}
          </label>
          <input
            type="range"
            min="0"
            max="5"
            step="1"
            value={teams.green}
            onChange={(e) => setTeams({...teams, green: parseInt(e.target.value)})}
            className="w-full"
          />
        </div>
      </div>
      
      <div className="mt-4 text-sm text-gray-300">
        <p>States: <span className="text-gray-400">Patrolling</span> • 
        <span className="text-cyan-400 ml-1">Pursuing</span> • 
        <span className="text-yellow-400 ml-1">Firing</span> • 
        <span className="text-red-400 ml-1">Evading</span></p>
      </div>
    </div>
  );
};

export default TacticalRadarSimulator;
