# Deep Learning for Metasurface Optimization

Optimization of single-element metasurface parameters using deep learning (non-linear regression) with tensorflow/keras and ~5600 Lumerical simulations as training data. Simulations performed under normally incident light. The features that define metasurface are the 1.length (L) 2.width (W) 3. height (H) 4.x-direction periodicity (Ux) 5. y-direction periodicity (Uy). The output is the phase spectrum around and across the visible at increments of 5 nm (450 nm - 800 nm).

For the powerpoint, there are animations so I recommend watching it in slide show mode.

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


# Results

![image](/Images/Summary.png)


Comparing the accuracy from the test set and the validation, we see similar values suggesting that we have likely not overfit our systems, which is positive. Ideally, the accuracy would be higher, but it should be noted that the dataset is excpetionally small, and wasn't originally designed for this purpose. Rather, I developed this dataset for a different project and this project was born as a result of my mere curiousity. Additionally, one of the challenges of designing in the metasurface realm is the issue of coupling and other offects that are very hard to predict. Based on my experience, a larger and better designed dataset could really help with this. Another reason the accuracy might not be as high can be explained by the fact that phase wraps around 2π, meaning that 0 is the same as 2π is the same as 12π. It is likely (verified by image below) that the network predicts a phase that when wrapped to 2π is the same value as the actual phase. 

![image](/Images/val_408.png)

One exciting result shown in the image above is from the validation set. The reason this is interesting is becaause the network predicted a phase of -2π when the FDTD software converged on 0. In reality these are the same phase values, but it appears that the network is able to learn that phase wraps, with it being explicitly put in the code. The way to justify it is via the non-linear regression process. For isotropic structures, we know the phsae is 0 across the entire spectrum. However, sometimes the FDTD software will provide 2π or 0 radians. This means in the training process, there are isotropic structures with 0 radians and 2π radians, it through the non-linear regression process, the network detected this pattern because it helped minimize the loss. 
