# SALP Research Reference
## Bio-Inspired Soft Underwater Robot Research

*Research tracking for implementing SALP-inspired agents in gymnasium environment*

---

## Project Overview

**SALP** (Salp-inspired Approach to Low-energy Propulsion) - Bio-inspired soft underwater robot that swims via jet propulsion, mimicking marine invertebrate salps.

**Source**: University of Pennsylvania, Sung Robotics Lab  
**Research URL**: https://sung.seas.upenn.edu/research/bio-inspired-soft-underwater-robot-that-swims-via-jet-propulsion/

---

## Target Behavior (What We Want to Implement)

### Core Locomotion Mechanism
- **Jet Propulsion**: Volume-based water displacement for thrust
- **Expansion Cycle**: Contract → suck water in → expand → expel water → thrust forward
- **Pulsing Motion**: Rhythmic expansion/contraction like jellyfish
- **Natural Spring-Back**: Passive return to expanded state after contraction

### Visual Morphing
- **Body Shape**: Smooth barrel/ellipsoid (not origami)
- **State Changes**: 
  - **Contracted**: More spherical, smaller volume
  - **Expanded**: Ellipsoidal, larger volume
- **Smooth Transitions**: Gradual morphing between states

---

## Key Research Insights

### Biological Inspiration
- **Salps**: Barrel-shaped marine invertebrates
- **Natural Behavior**: Rapid body cavity volume changes
- **Water Flow**: Front aperture intake → rear funnel expulsion
- **Multi-Agent**: Can form "salp chains" for coordinated swimming

### Performance Characteristics
- **Swimming Speed**: ~6.7 cm/s (0.2 body lengths/s)
- **Multi-Robot Benefits**: 9% velocity increase, 16% acceleration boost when coordinated
- **Energy Efficiency**: Cost of transport ~2.0
- **Bidirectional**: Forward and reverse propulsion possible

---

## Implementation Focus

### Essential Mechanics for Simulation
1. **Volume-Based Physics**
   - Body size changes drive water displacement
   - Larger volume change = more thrust
   - Simple expansion/contraction timing

2. **Propulsion Cycle**
   - **Phase 1**: Contract body (motor + tendon simulation)
   - **Phase 2**: Natural spring-back expansion
   - **Result**: Directional water jet → forward thrust

3. **Visual Representation**
   - Smooth ellipsoid body (no complex geometry)
   - Dynamic size scaling during cycles
   - Optional: translucent/transparent appearance

### Multi-Agent Possibilities
- **Chain Formation**: Physical connections between agents
- **Coordinated Pulsing**: Synchronized or asynchronous cycles
- **Collective Benefits**: Improved swimming performance

---

## Research Evolution Timeline

- **2021**: Original origami-inspired robot (magic ball pattern)
- **2023**: Drag coefficient characterization (0.64-1.26 range)
- **2024-2025**: Multi-robot coordination, bidirectional control, self-sensing

---

## Key Publications

1. **"Origami-inspired robot that swims via jet propulsion"** (2021)
   - IEEE Robotics and Automation Letters
   - Original concept and implementation

2. **"Effect of Jet Coordination on Underwater Propulsion with the Multi-Robot SALP System"** (2025)
   - Multi-robot coordination benefits
   - Performance improvements quantified

3. **"Drag coefficient characterization of the origami magic ball"** (2023)
   - Fluid dynamics analysis
   - Shape-dependent drag characteristics

---

## Implementation Notes

### Current Gymnasium Environment
- **Existing**: `squid_robot.py` - basic jet propulsion simulation
- **Enhancement Target**: Add SALP-inspired volume-based morphing
- **Key Addition**: Body size changes during propulsion cycles

### Simulation Enhancements
- Replace fixed-size robot with morphing ellipsoid
- Add expansion/contraction animation
- Implement volume-based thrust calculation
- Optional: Multi-agent chain formation

### Physics Considerations
- **Thrust Calculation**: Based on volume change rate
- **Body Dynamics**: Smooth interpolation between contracted/expanded states
- **Water Interaction**: Volume displacement drives propulsion force

---

## Video References
- **Multi-Robot Demo**: https://www.youtube.com/watch?v=mzd1QCXssCk
- **Research Channel**: Sung Robotics Group (YouTube)

---

## Next Steps for Implementation

1. **Modify Current Simulation**: Enhance `squid_robot.py` with morphing body
2. **Add Volume Physics**: Implement expansion/contraction thrust mechanics
3. **Visual Enhancement**: Smooth body size transitions
4. **Multi-Agent**: Explore chain formation capabilities
5. **RL Training**: Develop agents that learn optimal pulsing patterns

---

*Last Updated: January 2025*
*Focus: Simplified implementation for gymnasium environment experimentation*
