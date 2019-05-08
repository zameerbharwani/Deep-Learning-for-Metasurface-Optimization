# tf-Metasurface_Optimization
Optimization of single-element metasurface parameters using tensorflow and ~5600 Lumerical simulations as training data. Simulations performed under normally incident light. The features that define metasurface are the 1) length 2) width 3) height 4) x-direction periodicity 5) y-direction periodicity. The output is the phase spectrum around and across the visible at increments of 5 nm (450 nm - 800 nm)

For the powerpoint, there are animations so I recommend watching it in slide show mode. There are 

_**Scripts to follow soon**_

_Everything published in this repo has been done so with permission_

# Background

Metasurfaces are used to manipulate light in various manners for a plethora of applications. Current state-of-the-art methods to design these nanostructures are quite medieval and rely on a mere brute force method. That is, given a desired output, what combination of metasurface parameters can give us the values closest to what we seek? To answer this question, researchers rely on simulation software and perform thousands of parameter sweeps in hope that they find the best combination. The cost of the simulations are high, both in terms of time and computing power, especially in a research setting where many people are simultaneously working on the same cluster (see slide 18). 

In performing these vast parameter sweeps, researchers unknowingly built a powerful dataset. During my time at Harvard, I realized this and decided to aggregate some few thousand of my simulations and use it as a proof of concept that there exists a better and more efficient way to optimize metasurface parameters; deep learning. 

Attached is part of the presentation for the talk I gave to my colleagues at the Capasso Group in Harvard University's Applied Physics Department along with some results using the scatternet repo from MIT. The Capaso group is primarily comprised of physicists and thus a component of this presentation is on some of the basics of neural networks, which I think might be heplful to some people. Given that the talk was oral and the slides were merely visual aids, I've added some notes at the bottom of some slides for clarity. 

Due to the success with the scatternet repo (Nanophotonic Particle Simulation and Inverse Design), I have now built a better/simpler version designed for metasurface parameter optimization. Currently, there is only code for the forward design, though the inverse design is the next step. I will not be further developing my inverse design code due to time limitations and thus will not be posting it. I would highly recommend taking a look at the inverse design developed in the scatternet repo. Let it be noted that even though this repo is only for the forward design and thus does not alleviae the problem of brute force, it does offer two things:

1) Since deep learning is merely vectors/matrices, all computations are analytical and faster by up to 1200x compared to the simulation methods used currently, which are numerical

2) This code for the forward design sets the foundation for the inverse desgin

**Definitions**:

Forward Design: Given a set of input parameters that define a metasurface, predict the phase spectrum <br>
Inverse Design: Given a desired phase spectrum, what paramaters result in said phase spectrum?
